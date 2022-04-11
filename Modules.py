# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:32:17 2022

@author: Zhongshu.Hou & Qinwen.hu

Modules
"""
import torch
from torch import nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

'''
Import initialized SCM matrix
'''
Sc = np.load('./SpecCompress.npy').astype(np.float32)

'''
Encoder
'''
class Encoder(nn.Module):
    
    def __init__(self, auto_encoder = True):
        super(Encoder, self).__init__()
        
        self.F = 601
        self.F_c = 256
        self.F_low = 125
        self.auto_encoder = auto_encoder

        #---------------------------whole learnt-----------------------
        # self.flc = nn.Linear(self.F, self.F_c, bias=False)
        # self.flc.weight = nn.Parameter(torch.from_numpy(Sc), requires_grad=self.auto_encoder)   
        #--------------------------------------------------------------

        #---------------------------high learnt-----------------------
        self.flc_low = nn.Linear(self.F, self.F_low, bias=False)
        self.flc_low.weight = nn.Parameter(torch.from_numpy(Sc[:self.F_low, :]), requires_grad=False)

        self.flc_high = nn.Linear(self.F, self.F_c - self.F_low, bias=False)
        self.flc_high.weight = nn.Parameter(torch.from_numpy(Sc[self.F_low:, :]), requires_grad=True)
        #--------------------------------------------------------------

        self.conv_1 = nn.Conv2d(2,16,kernel_size=(2,5),stride=(1,2),padding=(1,1))
        self.bn_1 = nn.BatchNorm2d(16, eps=1e-8)
        self.act_1 = nn.PReLU(16)
        
        self.conv_2 = nn.Conv2d(16,32,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)
        
        self.conv_3 = nn.Conv2d(32,48,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_3 = nn.BatchNorm2d(48, eps=1e-8)
        self.act_3 = nn.PReLU(48)
        
        self.conv_4 = nn.Conv2d(48,64,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)
        
        self.conv_5 = nn.Conv2d(64,80,kernel_size=(1,2),stride=(1,1),padding=(0,1))
        self.bn_5 = nn.BatchNorm2d(80, eps=1e-8)
        self.act_5 = nn.PReLU(80)
        
    def forward(self,x):
        #x.shape = (Bs, F, T, 2)
        x = x.permute(0,3,2,1) #(Bs, 2, T, F)
        x = x.to(torch.float32)
        # x = self.flc(x)
        x_low = self.flc_low(x)
        x_high = self.flc_high(x)
        x = torch.cat([x_low, x_high], -1)
        x_1 = self.act_1(self.bn_1(self.conv_1(x)[:,:,:-1,:]))
        x_2 = self.act_2(self.bn_2(self.conv_2(x_1)[:,:,:-1,:]))
        x_3 = self.act_3(self.bn_3(self.conv_3(x_2)[:,:,:-1,:]))
        x_4 = self.act_4(self.bn_4(self.conv_4(x_3)[:,:,:-1,:]))
        x_5 = self.act_5(self.bn_5(self.conv_5(x_4)[:,:,:,:-1]))      
        
        return [x_1,x_2,x_3,x_4,x_5]

'''
DPRNN
'''
class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits
        
        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)
    
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)
        
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        
        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)

        self.width = width
        self.channel = channel
    
    def forward(self,x):
        # x.shape = (Bs, C, T, F)
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        x = x.permute(0,2,3,1) #(Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()   
        ## Intra RNN    
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0] #(Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel) #(Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = torch.add(x, intra_out)      
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0,2,1,3) #(Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1], inter_LSTM_input.shape[2], inter_LSTM_input.shape[3]) #(Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0,3,1,2)
        inter_out = inter_out.contiguous()
        
        return inter_out

'''
Decoder
'''
class Real_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Real_Decoder, self).__init__()
        self.F = 601
        self.F_c = 256
        self.F_low = 125
        self.auto_encoder = auto_encoder

        self.real_dconv_1 = nn.ConvTranspose2d(160, 64, kernel_size=(1,2), stride=(1,1))
        self.real_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.real_act_1 = nn.PReLU(64)
        
        self.real_dconv_2 = nn.ConvTranspose2d(128, 48, kernel_size=(2,3), stride=(1,1))
        self.real_bn_2 = nn.BatchNorm2d(48, eps=1e-8)
        self.real_act_2 = nn.PReLU(48)
        
        self.real_dconv_3 = nn.ConvTranspose2d(96, 32, kernel_size=(2,3), stride=(1,1))
        self.real_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_3 = nn.PReLU(32)
        
        self.real_dconv_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2,3), stride=(1,1))
        self.real_bn_4 = nn.BatchNorm2d(16, eps=1e-8)
        self.real_act_4 = nn.PReLU(16)
        
        self.real_dconv_5 = nn.ConvTranspose2d(32, 1, kernel_size=(2,5), stride=(1,2))
        self.real_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.real_act_5 = nn.PReLU(1)

        #---------------------random iSCM init------------------------
        self.inv_flc = nn.Linear(self.F_c, self.F, bias=False)
        #--------------------------------------------------------------

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1 = self.real_act_1(self.real_bn_1(self.real_dconv_1(skipcon_1)[:,:,:,:-1]))
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2 = self.real_act_2(self.real_bn_2(self.real_dconv_2(skipcon_2)[:,:,:-1,:-2]))
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3 = self.real_act_3(self.real_bn_3(self.real_dconv_3(skipcon_3)[:,:,:-1,:-2]))
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4 = self.real_act_4(self.real_bn_4(self.real_dconv_4(skipcon_4)[:,:,:-1,:-2]))
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5 = self.real_act_5(self.real_bn_5(self.real_dconv_5(skipcon_5)[:,:,:-1,:-1]))              
        outp = self.inv_flc(x_5)

        return outp

 

class Imag_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Imag_Decoder, self).__init__()

        self.F = 601
        self.F_c = 256
        self.F_low = 125
        self.auto_encoder = auto_encoder
        self.imag_dconv_1 = nn.ConvTranspose2d(160, 64, kernel_size=(1,2), stride=(1,1))
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)
        
        self.imag_dconv_2 = nn.ConvTranspose2d(128, 48, kernel_size=(2,3), stride=(1,1))
        self.imag_bn_2 = nn.BatchNorm2d(48, eps=1e-8)
        self.imag_act_2 = nn.PReLU(48)
        
        self.imag_dconv_3 = nn.ConvTranspose2d(96, 32, kernel_size=(2,3), stride=(1,1))
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)
        
        self.imag_dconv_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2,3), stride=(1,1))
        self.imag_bn_4 = nn.BatchNorm2d(16, eps=1e-8)
        self.imag_act_4 = nn.PReLU(16)
        
        self.imag_dconv_5 = nn.ConvTranspose2d(32, 1, kernel_size=(2,5), stride=(1,2))
        self.imag_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.imag_act_5 = nn.PReLU(1)

        #---------------------random iSCM init------------------------
        self.inv_flc = nn.Linear(self.F_c, self.F, bias=False)
        #--------------------------------------------------------------

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1 = self.imag_act_1(self.imag_bn_1(self.imag_dconv_1(skipcon_1)[:,:,:,:-1]))
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2 = self.imag_act_2(self.imag_bn_2(self.imag_dconv_2(skipcon_2)[:,:,:-1,:-2]))
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3 = self.imag_act_3(self.imag_bn_3(self.imag_dconv_3(skipcon_3)[:,:,:-1,:-2]))
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4 = self.imag_act_4(self.imag_bn_4(self.imag_dconv_4(skipcon_4)[:,:,:-1,:-2]))
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5 = self.imag_act_5(self.imag_bn_5(self.imag_dconv_5(skipcon_5)[:,:,:-1,:-1]))      
        outp = self.inv_flc(x_5)

        return outp

'''
DPCRN
'''
class DPCRN(nn.Module):
    #autoencoder = True
    def __init__(self):
        super(DPCRN,self).__init__()
        self.encoder = Encoder()
        self.dprnn = DPRNN(80,127,80)
        self.real_decoder = Real_Decoder()
        self.imag_decoder = Imag_Decoder()
        
    def forward(self, x):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        encoder_out = self.encoder(x) 
        dprnn_out = self.dprnn(encoder_out[4])
        enh_real = self.real_decoder(dprnn_out, encoder_out)
        enh_imag = self.imag_decoder(dprnn_out, encoder_out)
        enh_real = enh_real.permute(0,3,2,1)
        enh_imag = enh_imag.permute(0,3,2,1)
        enh_stft = torch.cat([enh_real, enh_imag], -1)
        
        return enh_stft



'''
MHAN
'''
class AttentionMaskV2(nn.Module):
    
    def __init__(self, causal, mask_value=-1e9):
        super(AttentionMaskV2, self).__init__()
        self.causal = causal
        
    def lower_triangular_mask(self, shape):
        '''
        

        Parameters
        ----------
        shape : a tuple of ints

        Returns
        -------
        a square Boolean tensor with the lower triangle being False

        '''
        row_index = torch.cumsum(torch.ones(size=shape), dim=-2)
        col_index = torch.cumsum(torch.ones(size=shape), dim=-1)
        return torch.lt(row_index, col_index)  # lower triangle:True, upper triangle:False
    
    def merge_masks(self, x, y):
        
        if x is None: return y
        if y is None: return x
        return torch.logical_and(x, y)
        
        
    def forward(self, inp):
        #input (bs, L, ...)
        max_seq_len = inp.shape[1]
        if self.causal ==True:
            causal_mask = self.lower_triangular_mask([max_seq_len, max_seq_len])      #(L, l)
            return causal_mask
        else:
            return torch.zeros(size=(max_seq_len, max_seq_len), dtype=torch.float32)

class MHAblockV2(nn.Module):
    
    def __init__(self, d_model, d_ff, n_heads):
        
        super(MHAblockV2, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        self.MHA = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, bias=False)
        self.norm_1 = nn.LayerNorm([self.d_model], eps=1e-6)
        
        self.fc_1 = nn.Conv1d(self.d_model, self.d_ff, 1)
        self.act = nn.ReLU()
        self.fc_2 = nn.Conv1d(self.d_ff, self.d_model, 1)
        self.norm_2 = nn.LayerNorm([self.d_model], eps=1e-6)
        
        
    def forward(self, x, att_mask):
        
        # x input: (bs, L, d_model)
        x = x.permute(1,0,2).contiguous() #(L, bs, d_model)
        layer_1,_ = self.MHA(x, x, x, attn_mask=att_mask, need_weights=False) #(L, bs, d_model)
        layer_1 = torch.add(x, layer_1).permute(1,0,2).contiguous() #(L, bs, d_model) ->  (bs, L, d_model)
        layer_1 = self.norm_1(layer_1) #(bs, L, d_model)
        
        layer_2 = self.fc_1(layer_1.permute(0,2,1).contiguous()) #(bs, d_ff, L) 
        layer_2 = self.act(layer_2) #(bs, d_ff, L) 
        layer_2 = self.fc_2(layer_2).permute(0,2,1).contiguous() #(bs, d_ff, L)  -> (bs, d_model, L) -> (bs, L, d_model)
        layer_2 = torch.add(layer_1, layer_2)
        layer_2 = self.norm_2(layer_2)
        return layer_2

class MHAN(nn.Module):
    '''
    Multi-head attention network with nn.MultiheadAttention API
    SDC: spectrum dimension compression
    SEC: spectrum energy compression
    '''
    def __init__(
          		self,
          		f_input,
          		n_outp,
          		d_model,
          		n_blocks,
          		n_heads,
          		causal
          		):        
        super(MHAN, self).__init__()
        
        self.n_outp = n_outp
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_ff = d_model*4
        self.d_k = self.d_model // self.n_heads 
        self.causal = causal
        self.F_low = 125

        #-----------------------high learnt-------------------------
        self.flc_low = nn.Linear(f_input, self.F_low, bias=False)
        self.flc_low.weight = nn.Parameter(torch.from_numpy(Sc[:self.F_low, :]), requires_grad=False)

        self.flc_high = nn.Linear(f_input, d_model - self.F_low, bias=False)
        self.flc_high.weight = nn.Parameter(torch.from_numpy(Sc[self.F_low:, :]), requires_grad=True)  
        #------------------------------------------------------------
        
        #input layer
        self.input_norm =  nn.LayerNorm([self.d_model], eps=1e-6)
        self.input_act = nn.ReLU()
        
        #attention block
        self.att_block_list = nn.ModuleList([MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.n_blocks)])
        
        #output_layer
        self.out_fc = nn.Linear(self.d_model, self.n_outp, bias=True)
        self.out_act = nn.Sigmoid()
        
    
    def forward(self, x):
        '''
        

        Parameters
        ----------
        x : tensor (bs, F, L)

        Returns
        -------
        x : tensor (bs, F, L)
        '''
        x = x.permute(0, 2, 1).contiguous() #(bs, L, F)
        
        #SDC
        # x = self.sdc(x) #(bs, L, d_model)
        x_low = self.flc_low(x)
        x_high = self.flc_high(x)
        x = torch.cat([x_low, x_high], -1)
        #attention mask
        att_mask = AttentionMaskV2(self.causal, -1.0e9)(x).to(device)
        
        #input layer
        x = self.input_norm(x) # (bs, L, d_model)
        x = self.input_act(x) #(bs, L, d_model)
        
        #attention block
        for att_block in self.att_block_list:
            x = att_block(x, att_mask) #(bs, L, d_model)
            
            
        #output layer
        x = self.out_fc(x) #(bs, L, d_model) -> (bs, L, F)
        x = self.out_act(x).permute(0, 2, 1).contiguous() #(bs, L, F) - > (bs, F, L)

        return x
