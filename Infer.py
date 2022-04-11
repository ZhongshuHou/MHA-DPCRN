# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:05:17 2022

@author: Zhongshu.Hou

Modules
"""
from Modules import DPCRN, MHAN
import torch
import soundfile as sf
import librosa 
from tqdm import tqdm
from signal_processing import iSTFT_module_1_8
device = torch.device("cuda:0")
torch.set_default_tensor_type(torch.FloatTensor)
# WINDOW = torch.sqrt(torch.hamming_window(1200,device=device))
WINDOW = torch.sqrt(torch.hann_window(1200,device=device) + 1e-8)
import argparse
from collections import OrderedDict
#%%
def infer(args):
    '''model'''
    model_DPCRN = DPCRN() # 定义模型
    model_MHA = MHAN(f_input=601, n_outp=601, d_model=256, n_blocks=5, n_heads=8, causal=True)

    ''' load checkpoints'''
    checkpoint_DPCRN = torch.load(args.check_dpcrn,map_location=device)
    model_DPCRN.load_state_dict(checkpoint_DPCRN['state_dict'])
    checkpoint_MHA = torch.load(args.check_mha,map_location=device)
    model_MHA.load_state_dict(checkpoint_MHA['state_dict']) 
    model_DPCRN = model_DPCRN.to(device)
    model_MHA = model_MHA.to(device)  
    model_DPCRN.eval()
    model_MHA.eval

    noisy_dir = args.noisy_dir
    noisy_list = librosa.util.find_files(noisy_dir, ext='wav')
    i = 0
    with torch.no_grad():
        for noisy_f in tqdm(noisy_list):
            
                
            noisy_s = sf.read(noisy_f)[0].astype('float32')
            noisy_s = torch.from_numpy(noisy_s.reshape((1,len(noisy_s)))).to(device)
            noisy_stft = torch.stft(noisy_s,1200,600,win_length=1200,window=WINDOW,center=True)
            noisy_mag = torch.sqrt(noisy_stft[:, :, :, 0]**2 + noisy_stft[:, :, :, 1]**2)
            mask = model_MHA(noisy_mag).unsqueeze(-1) #(bs, F, T, 1)
            mha_out = torch.mul(mask, noisy_stft)

            #----------------mapping based-----------------
            enh_stft = model_DPCRN(mha_out)
            #----------------------------------------------

            enh_s = iSTFT_module_1_8(n_fft=1200, hop_length=600, win_length=1200,window=WINDOW,center = True,length = noisy_s.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

            enh_s = enh_s[0,:].cpu().detach().numpy()

            enh_s = librosa.resample(enh_s, 48000, 16000)

            sf.write(args.saved_enhanced_dir + '/' + noisy_f.split('/')[-1], enh_s, 16000)
            i+=1


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_dpcrn", required=True, 
                        help='Path to DPCRN checkpoints')
    parser.add_argument("--check_mha", required=True, 
                        help='Path to MHAN checkpoints')
    parser.add_argument("--noisy_dir", required=True, 
                        help='Path to the dir containing noisy clips')
    parser.add_argument("--saved_enhanced_dir", required=True, 
                        help='Path to the dir saving enhanced clips')
    
    args = parser.parse_args()
    infer(args)

  