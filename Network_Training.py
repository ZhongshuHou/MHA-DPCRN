# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:32:17 2022

@author: Zhongshu.Hou & Qinwen.Hu

network training
"""
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
from Dataloader import Dataset, collate_fn
from Modules import DPCRN, MHAN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
from signal_processing import iSTFT_module_1_8
WINDOW = torch.sqrt(torch.hann_window(1200,device=device) + 1e-8)

#------------------------warm up strategy------------------------
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

#------------------------start training------------------------
def train(end_epoch = 100):

    '''Loss functions'''
    def Loss(y_pred, y_true):
        snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),(torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        snr_loss = 10 * torch.log10(snr + 1e-7)
        
        pred_stft = torch.stft(y_pred,1200,600,win_length=1200,window=WINDOW,center=True)
        true_stft = torch.stft(y_true,1200,600,win_length=1200,window=WINDOW,center=True)
        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(2/3))
        pred_imag_c = pred_stft_imag / (pred_mag**(2/3))
        true_real_c = true_stft_real / (true_mag**(2/3))
        true_imag_c = true_stft_imag / (true_mag**(2/3))
        real_loss = torch.mean((pred_real_c - true_real_c)**2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
        mag_loss = torch.mean((pred_mag**(1/3)-true_mag**(1/3))**2)

        return real_loss + imag_loss + mag_loss, snr_loss

    def Loss_mag_only(mask, noisy_mag, clean_mag):
        mask = mask[:,:,:,0]
        enh_mag = torch.mul(mask, noisy_mag) + 1e-12
        mag_loss = torch.mean((enh_mag**(1/3)-clean_mag**(1/3))**2)

        return mag_loss
    
    '''model'''
    model_DPCRN = DPCRN()
    model_MHA = MHAN(f_input=601, n_outp=601, d_model=256, n_blocks=5, n_heads=8, causal=True)

    ''' train from checkpoints'''
    # checkpoint_DPCRN = torch.load('',map_location=device)
    # model_DPCRN.load_state_dict(checkpoint_DPCRN['state_dict'])
    # checkpoint_MHA = torch.load('',map_location=device)
    # model_MHA.load_state_dict(checkpoint_MHA['state_dict'])
    model_DPCRN = model_DPCRN.to(device)
    model_MHA = model_MHA.to(device)

    '''optimizer & lr_scheduler'''

    optimizer_dpcrn = NoamOpt(model_size=333, factor=1., warmup=3000,
                    optimizer=torch.optim.Adam(model_DPCRN.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    optimizer_mha = NoamOpt(model_size=333, factor=1., warmup=3000,
                    optimizer=torch.optim.Adam(model_MHA.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))    

    '''load train data'''
    dataset = Dataset(length_in_seconds=8, random_start_point=True, train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=True)

    '''start train'''
    for epoch in range(end_epoch):
        train_loss = []
        asnr_loss = []
        model_DPCRN.train()
        model_MHA.train()
        dataset.train = True
        dataset.random_start_point = True
        idx = 0

        '''train'''
        print('epoch %s--training' %(epoch))
        for i, data in enumerate(tqdm(data_loader)):
            noisy, clean = data
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer_dpcrn.optimizer.zero_grad()
            optimizer_mha.optimizer.zero_grad()

            noisy_stft = torch.stft(noisy,1200,600,win_length=1200,window=WINDOW,center=True)
            clean_stft = torch.stft(clean,1200,600,win_length=1200,window=WINDOW,center=True) #(bs, F, T, 2)
            clean_mag = torch.sqrt(clean_stft[:, :, :, 0]**2 + clean_stft[:, :, :, 1]**2)
            noisy_mag = torch.sqrt(noisy_stft[:, :, :, 0]**2 + noisy_stft[:, :, :, 1]**2)
            mask = model_MHA(noisy_mag).unsqueeze(-1) #(bs, F, T, 1)
            mha_out = torch.mul(mask, noisy_stft)
            #----------------mapping based-----------------
            enh_stft = model_DPCRN(mha_out)
            #----------------------------------------------
            enh_s = iSTFT_module_1_8(n_fft=1200, hop_length=600, win_length=1200,window=WINDOW,center = True,length = noisy.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

            stft_loss, snr_loss = Loss(enh_s, clean, train=True, idx = idx, epoch = epoch) 
            mag_loss = Loss_mag_only(mask, noisy_mag, clean_mag)
            loss_overall = stft_loss + mag_loss
            loss_overall.backward()
            torch.nn.utils.clip_grad_norm_(model_DPCRN.parameters(), max_norm=3)
            torch.nn.utils.clip_grad_norm_(model_MHA.parameters(), max_norm=3)
            optimizer_dpcrn.step()
            optimizer_mha.step()
            train_loss.append(loss_overall.cpu().detach().numpy())
            idx += 1
        train_loss = np.mean(train_loss)
        '''eval'''
        valid_loss = []
        model_DPCRN.eval()
        model_MHA.eval()
        print('epoch %s--validating' %(epoch))
        dataset.train = False
        dataset.random_start_point = False
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):

                noisy, clean = data
                noisy = noisy.to(device)
                clean = clean.to(device)
                # Mag = Mag.to(device)
                noisy_stft = torch.stft(noisy,1200,600,win_length=1200,window=WINDOW,center=True)
                clean_stft = torch.stft(clean,1200,600,win_length=1200,window=WINDOW,center=True) #(bs, F, T, 2)
                clean_mag = torch.sqrt(clean_stft[:, :, :, 0]**2 + clean_stft[:, :, :, 1]**2)
                noisy_mag = torch.sqrt(noisy_stft[:, :, :, 0]**2 + noisy_stft[:, :, :, 1]**2)
                mask = model_MHA(noisy_mag).unsqueeze(-1) #(bs, F, T, 1)
                mha_out = torch.mul(mask, noisy_stft)
                #----------------mapping based-----------------
                enh_stft = model_DPCRN(mha_out)
                #----------------------------------------------
                enh_s = iSTFT_module_1_8(n_fft=1200, hop_length=600, win_length=1200,window=WINDOW,center = True,length = noisy.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

                stft_loss, snr_loss = Loss(enh_s, clean, train=True, idx = idx, epoch = epoch)
                mag_loss = Loss_mag_only(mask, noisy_mag, clean_mag)
                loss_overall = stft_loss + mag_loss
                valid_loss.append(loss_overall.cpu().detach().numpy())
                asnr_loss.append(snr_loss.cpu().detach().numpy())
            valid_loss = np.mean(valid_loss)
            asnr_loss = np.mean(asnr_loss)
        print('train loss: %s, valid loss %s, snr loss: %s' %(train_loss, valid_loss, asnr_loss))
        print('current step:{}, current lr:{}'.format(optimizer_dpcrn._step, optimizer_dpcrn._rate))
        print('current step:{}, current lr:{}'.format(optimizer_mha._step, optimizer_mha._rate))

        torch.save(
            {'epoch': epoch,
                'state_dict': model_MHA.state_dict(),
                'optimizer': optimizer_mha.optimizer.state_dict()},
            './mha/model_epoch_%s_trainloss_%s_validloss_%s_snr_loss_%s.pth' %(str(epoch), str(train_loss), str(valid_loss), str(asnr_loss)))

        torch.save(
            {'epoch': epoch,
                'state_dict': model_DPCRN.state_dict(),
                'optimizer': optimizer_dpcrn.optimizer.state_dict()},
            './dpcrn/model_epoch_%s_trainloss_%s_validloss_%s_snr_loss_%s.pth' %(str(epoch), str(train_loss), str(valid_loss), str(asnr_loss)))


if __name__ == '__main__':
    train(end_epoch=300)
