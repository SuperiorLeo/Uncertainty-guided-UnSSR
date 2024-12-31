# from turtle import forward
from termios import VT1
from tkinter import SEL
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class GLAF(nn.Module):
    def __init__(self,planes,mode='ds1'):
        super(GLAF,self).__init__()
        self.mode = mode
        self.conv0 = nn.Conv2d(planes, planes, 1, 1, 0, bias=False)
        self.conv1 = nn.Conv2d(planes, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        # 在定义的时候 planes 数量区分 否则会导致维度不对
        self.down_sample = nn.Conv2d(planes,planes*2,4,2,1,bias=False)
        self.up_sample = nn.ConvTranspose2d(planes,planes//2,4,2,1)

    def forward(self,x):
        b, c, h, w = x.size() #[1,3,64,64]
        # 引入非线性
        x = self.conv0(x) #[1, 3, 64, 64]
        input_x = x.clone()
        
        # channel attention
        input_x = input_x.view(b, c, h*w).unsqueeze(1) #([1, 1, 3, 4096]) 最前面加一个维度
        channel_x = self.conv1(x).view(b, 1, h*w) #[1, 1, 4096] 维度调整
        channel_x = self.softmax(channel_x).unsqueeze(-1) #[1, 1, 4096, 1] 最后面加一个维度
        
        # adaption augment
        aug_x = torch.matmul(input_x, channel_x).view(b, c,1,1) #[1, 3, 1, 1]
        output_x = x * aug_x.expand_as(x) # [1,3,64,64]

        # downsample or upsample
        if self.mode == 'ds':
            output_x = self.down_sample(output_x)
        elif self.mode == 'us':
            output_x = self.up_sample(output_x)

        # print('GLAF: {}'.format(output_x.shape))

        return output_x


class ReshapeBlock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(ReshapeBlock,self).__init__()
        # TODO 这里第一个卷积进行维度转变 还是 第二个卷积进行梯度转换 可以验证一下
        self.conv1 = Conv3x3(inplanes, outplanes, 3, 1)
        self.relu = nn.PReLU()
        self.conv2 = Conv3x3(outplanes, outplanes,3, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Conv3_1(nn.Module):
    def __init__(self,planes_3,planes_1):
        super(Conv3_1,self).__init__()
        self.conv31 = Conv3x3(planes_3, planes_1, 3, 1)
        self.relu = nn.PReLU()
        self.conv32 = Conv3x3(planes_1, planes_1, 3, 1)
        self.conv1 = nn.Conv2d(planes_1, planes_1, 1, 1, 0, bias=False)

    def forward(self,x):
        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.conv1(x)

        return x

class LRED(nn.Module):

    def __init__(self,inplanes):
        super(LRED,self).__init__()
        self.conv31_down = Conv3_1(inplanes,inplanes)
        self.conv31_mid = Conv3_1(inplanes,inplanes//2)
        self.conv31_up = Conv3_1(inplanes,inplanes//3)
        self.conv1 = nn.Conv2d(inplanes, inplanes, 1, 1, 0, bias=False)
        self.reshape = ReshapeBlock(inplanes*3//2,inplanes)

    def forward(self,x):
        b, c, h, w = x.size()
        # reshape operation and 1*1conv
        x_0 = self.conv31_down(x)
        x_1 = self.conv31_mid(x)
        x_2 = self.conv31_up(x)

        # down reshape transpose
        x_0_0 = x_0.view(b,c,h*w) 
        # mid reshape
        x_1_0 = x_1.view(b,c//2,h*w)
        x_1_1 = x_1_0.permute(0,2,1) #[b,h*w,c//2]
        # matric 01
        M1 = torch.matmul(x_0_0,x_1_1) #[b,c//2,c]
        M1_T = M1.permute(0,2,1) #[b,C//2,c]
        # Feature0_1
        F10 = torch.matmul(x_1_1,M1_T) #[b,h*w,c]
        F10 = F10.view(b,c,h,w)
        F10 += x_0
        # up reshape
        x_2_0 = x_2.view(b,c//3,h*w)
        x_2_1 = x_2_0.permute(0,2,1) #[b,h*w,C//3]
        # matric 12
        M2 = torch.matmul(x_1_0,x_2_1) #[C//3,C//2]
        M2_T = M2.permute(0,2,1)
        #Feature1_2
        F21 = torch.matmul(x_2_1,M2_T) #[b,h*w,c]
        F21 = F21.view(b,c//2,h,w)
        F21 += x_1

        # concat reshape add
        F_all = torch.cat((F10,F21),1)
        F_all = self.reshape(F_all)
        out = F_all + x
    
        return out

class LCAB(nn.Module):
    def __init__(self,inplanes):
        super(LCAB,self).__init__()
        self.conv31 = ReshapeBlock(inplanes,inplanes)
        self.LRED = LRED(inplanes)
        self.conv32 = Conv3_1(inplanes,inplanes)
        self.conv33 = ReshapeBlock(inplanes,inplanes)

    def forward(self,x):
        out = self.conv31(x)
        out_1 = self.conv32(out)
        out_2 = self.LRED(out)
        out = out_1 + out_2 + x
        out = self.conv33(out)
        out = out + x

        return out

class CTBlock(nn.Module):
    def __init__(self,inplanes):
        super(CTBlock,self).__init__()

        self.RDB1 = LCAB(inplanes)
        self.GLAF0 = GLAF(inplanes,mode='ds')
        self.TE0 = Transformer_E(inplanes*2,2,3,inplanes*2//3,inplanes*2)
        self.GLAF1 = GLAF(inplanes*2,mode='us')
        self.reshape = ReshapeBlock(inplanes*2,inplanes)

    def forward(self,x):
        x_conv1 = self.RDB1(x)
        trans_x_in1 = self.GLAF0(x)
        x_trans1 = self.TE0(trans_x_in1)
        trans_x_out1 = self.GLAF1(x_trans1)
        x_stage1 = torch.cat((trans_x_out1, x_conv1),1)
        x_stage1 = self.reshape(x_stage1)
        x_stage1 = x_stage1 + x

        return x_stage1

class SFFormer(nn.Module):
    def __init__(self, inplanes=3, planes=30, outplanes=31,n_CTblocks=8):
        super(SFFormer, self).__init__()

        self.reshape_in = ReshapeBlock(inplanes,planes)

        self.backbone = nn.ModuleList(
            [CTBlock(planes) for _ in range(n_CTblocks)]
        )

        self.reshape_out = ReshapeBlock(planes,outplanes)

        self.sort = Fore_Back_dis()
    

    def forward(self, x):

        out = self.FGSTBlock(x)

        idx = self.sort(out)

        return out, idx

    def FGSTBlock(self, x):
        out = self.reshape_in(x)

        for i, block in enumerate(self.backbone):
            out = block(out)
        out = self.reshape_out(out)

        return out

class Fore_Back_dis(nn.Module):
    def __init__(self):
        super(Fore_Back_dis,self).__init__()
        self.conv = nn.Conv2d(1, 1, 1, 1, 0, bias=False)
        self.softmax = nn.ReLU()

    def forward(self, hsi):
        B, C, H, W = hsi.shape
        delta_list = []
        for i in range(C):
            grey = hsi[:,i,:,:].unsqueeze(1)
            f1 = self.conv(grey)
            # f1 = self.softmax(f1)
            entropy = -f1 * torch.log(f1)
            mean = entropy.mean()
            entropy = entropy.view(B,1,H*W)
            score_high = int(0.5*H*W)
            # score_low = int(0.3*B*H*W)
            vals_high, indices = entropy.topk(k=score_high, dim=2, largest=True, sorted=True)
            mean_high = vals_high.mean()
            low_mean = (mean- 0.5*mean_high)/0.5
            delta = mean_high-low_mean
            delta_list.append(delta)

        a, idx1 = torch.sort(torch.Tensor(delta_list), descending=True)#descending为alse，升序，为True，降序
        idx = idx1[:3]
        # print(idx)

        return idx
    
# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        x = self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1)) # 这是为了后面sigmoid之后与v相乘的尺度匹配
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # print(self.num_heads * self.dim_head)
        b, h, w, c = x_in.shape
        
        x = x_in.reshape(b,h*w,c)
        
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        
        out_c = self.proj(x).view(b, h, w, c)

        out = out_c 

        return out

class Transformer_E(nn.Module):
    # def __init__(self, dim, depth=2, heads=3, dim_head=4, mlp_dim=12, sp_sz=16*16, num_channels = 48,dropout=0.):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))


    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.layers:
            # print('x: {}'.format(x.shape))
            x = attn(x)
            x = ff(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_tensor = torch.rand(6, 3, 256, 256)
    input_tensor = torch.rand(1, 3, 512, 512)
    
    '''
    Parameters number is  2734878
    Parameters number is 1787844.0; Flops: 353953710074.0
    '''
    model = SFFormer(3,48,31,3)
    # model = DLCD(3)
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    macs, params = profile(model, inputs=(input_tensor, ))
    # print(output_tensor.shape)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print('Parameters number is {}; Flops: {}'.format(params,macs))
    print(torch.__version__)

