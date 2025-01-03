import torch
import torch.nn as nn
# from SpaNet import BCR, denselayer
import numpy as np
import torch.nn.functional as f
# from spectralnorm import SpectralNorm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class My_Bn_1(nn.Module):
    def __init__(self):
        super(My_Bn_1,self).__init__()
    def forward(self,x):
        
        return x - torch.mean(x,dim = 1,keepdim=True)

class My_Bn_2(nn.Module):
    def __init__(self):
        super(My_Bn_2,self).__init__()
    def forward(self,x):

        return x - nn.AdaptiveAvgPool2d(1)(x)

class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False,spectralnorm = False,bias=False):
        super(BCR,self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding,bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding,bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                if spectralnorm:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        SpectralNorm(self.conv),
                        self.relu,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                )
        else:
            if BN:
                if spectralnorm:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        SpectralNorm(self.conv),
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class denselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True,BN=True,bias=True,spectralnorm=False):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=RELU,BN=BN,spectralnorm=spectralnorm,bias=bias)
        self.actlayer = BCR(kernel=3,cin=cout,cout=cout,group=cout,RELU=RELU,padding=1,BN=BN,spectralnorm=spectralnorm,bias=bias)
    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)

        return output

class stage(nn.Module):
    def __init__(self,cin,cout,final=False,extra=0,BN=False,linear=True,bias=False,spectralnorm=False):
        super(stage,self).__init__()
        self.Upconv = BCR(kernel = 3,cin = cin, cout = cout,stride= 1,padding=1,RELU=False,BN=False)
        if final == True:
            f_cout = cout +1
        else:
            f_cout = cout
        mid = cout*3
        self.denselayers = nn.ModuleList([
            denselayer(cin=1*cout,cout=cout*2,BN = BN,bias=bias,spectralnorm=spectralnorm),
            denselayer(cin=3*cout,cout=cout*2,BN = BN,bias=bias,spectralnorm=spectralnorm),
            denselayer(cin=5*cout,cout=cout*2,BN = BN,bias=bias,spectralnorm=spectralnorm),
            denselayer(cin=7*cout,cout=cout*2,BN = BN,bias=bias,spectralnorm=spectralnorm),
            denselayer(cin=9*cout,cout=f_cout,RELU=False,BN=False,bias=bias,spectralnorm=spectralnorm)])
        self.bn = My_Bn_1()
        self.linear = linear

    def forward(self,MSI,recon=None):
        MSI = self.Upconv(MSI)
        # print('-----------ERROR------------')
        # print(MSI.shape)
        if recon == None:
            x = [MSI]
            # print(MSI.shape)
            for layer in self.denselayers:
                x_ = layer(torch.cat(x,1))
                x.append(x_)
            if self.linear == True:
                return x[-1] + MSI
            else:
                return  x[-1]
        else:
            MSI = MSI + recon
            x = [MSI]
            # print(MSI.shape)
            for layer in self.denselayers:
                x_ = layer(torch.cat(x,1))
                x.append(x_)
            # if self.linear == True:
            #     return x[-1] + MSI
            # else:
            return  x[-1] + MSI




class reconnet(nn.Module):
    def __init__(self,extra=[0,0,0]):
        super(reconnet,self).__init__()

        self.stages = nn.ModuleList([
            stage(cin=3,cout=31,extra = extra[0],BN = False,linear=False,bias=False,spectralnorm=False),
            stage(cin=3,cout=31,extra = extra[1],BN = True,linear=False,bias=False,spectralnorm=False),
            stage(cin=3,cout=31,extra = extra[1],BN = True,linear=False,bias=False,spectralnorm=False),
            stage(cin=3,cout=31,extra = extra[2],BN = True,linear=False,bias=False,spectralnorm=False)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sort = Fore_Back_dis()

    # def degradation(self,resp1,HSI):

    #     MSI = torch.einsum('bji,bikm->bjkm',resp1,HSI)
    #     return MSI
    
    def forward(self,MSI,resp=None):

        ref = [np.array(range(8))*4, np.array(range(16))*2]
        ref[0][-1] = 30
        ref[1][-1] = 30
        recon_out = None
        MSI = [MSI]
        for index , stage in enumerate(self.stages):
            recon = stage(MSI[-1])
            if recon_out is None:
                recon_out = recon
            else:
                recon_out =  recon_out + recon
            # recon_out = nn.functional.leaky_relu(recon_out, negative_slope=0.01, inplace=True)
            # recon_out1 = nn.functional.relu(recon_out, inplace=True)
            # recon_out1 = nn.functional.sigmoid(recon_out)
            recon_out1 = recon_out
            # msi_ = MSI[0] - self.degradation(resp,recon_out1)
            # MSI.append(msi_)
        # recon_out1 = 
        recon_out1 = torch.nn.functional.leaky_relu(recon_out1, negative_slope=0.001) 
        idx = self.sort(recon_out1)
        return recon_out1 , idx

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

if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 这块可以模拟输入是patch时每一层的输入和输出
    # input_tensor = torch.rand(6, 3, 256, 256)
    input_tensor = torch.rand(1, 3, 512, 512)

    model = reconnet()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor,idx = model(input_tensor)
        print(output_tensor.shape) #torch.Size([1, 31, 512, 482])
    macs, params = profile(model, inputs=(input_tensor, ))
    # print(output_tensor.shape)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print('Parameters number is {}; Flops: {}'.format(params,macs))
    print(torch.__version__)
    '''
    Parameters number is  170997
    arameters number is 341993.0; Flops: 179310428160.0 
    '''
