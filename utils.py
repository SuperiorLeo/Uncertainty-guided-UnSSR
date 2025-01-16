from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
# import hdf5storage
import h5py
from math import exp
import torch.nn.functional as F
import scipy.io as scio 

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
            'epoch': epoch,
            'iter': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))


def save_matv73(mat_name, var_name, var):
    scio.savemat(mat_name,{var_name: var}) 
    # hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, hyper_loss, all_loss, test_mrae, test_rmse, test_psnr):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, hyper_loss, all_loss, test_mrae, test_rmse, test_psnr))
    loss_csv.flush()    
    loss_csv.close

def record_loss2(loss_csv,epoch, iteration, epoch_time, lr, train_loss, hyper_loss, per_Loss, all_loss, test_mrae, test_rmse, test_psnr):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, hyper_loss, per_Loss, all_loss, test_mrae, test_rmse, test_psnr))
    loss_csv.flush()    
    loss_csv.close

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)



class Loss_train(nn.Module):
    def __init__(self):
        super(Loss_train, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        # error = torch.abs(outputs - label)
        rrmse = torch.mean(error.view(-1))
        return rrmse

class Loss_train_spec(nn.Module):
    def __init__(self):
        super(Loss_train_spec, self).__init__()

    def forward(self, F0, output_list):
        const_pro = 0.1*self.erro(output_list[2],output_list[1]) + 0.5*(self.erro(output_list[2],F0)) + 0.1*(self.erro(output_list[1],output_list[0])) + 0.2*(self.erro(output_list[1],F0))

        return (1.0-const_pro)
    
    def erro(self, input1, input2):
        
        mae = torch.abs(input1 - input2)
        rmae = torch.mean(mae.contiguous().view(-1))

        return rmae

    
class Loss_train3(nn.Module):
    def __init__(self):
        super(Loss_train3, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) 
        # error = torch.abs(outputs - label)
        rrmse = torch.mean(error.view(-1))
        return rrmse


class Loss_valid(nn.Module):
    def __init__(self):
        super(Loss_valid, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        # error = torch.abs(outputs - label)
        mrae = torch.mean(error.view(-1))
        return mrae

class LossTrainCSS2(nn.Module):
    def __init__(self):
        super(LossTrainCSS2, self).__init__()
  
    def forward(self, outputs, label, rgb_label):
        filters = np.load("/home/lengyihong/workspace/HRNet_un/official_scoring_code/resources/cie_1964_w_gain.npz")['filters']

        filters = torch.Tensor(filters).cuda()
        shape1 = outputs.size() # ([1, 31, 64, 64])
        outputs = outputs - outputs.min()

        outputs_1 = outputs.reshape(shape1[0],shape1[1],-1) # ([1, 31, 4096])
        outputs_1 = outputs_1.permute(0,2,1) # ([1, 4096, 31])
        reRGB = torch.matmul(outputs_1,filters)  #torch.Size([1, 4096, 3])
        reRGB = reRGB.permute(0,2,1) # torch.Size([1, 3, 4096])
        reRGB = reRGB.reshape(shape1[0],3,shape1[2],shape1[3]) # torch.Size([1, 3, 64, 64])
        reRGB = normalize(reRGB, max_val=255., min_val=0.)

        rrmse = self.mrae_loss(reRGB, rgb_label)

        return rrmse

    def mrae_loss(self, outputs, label):
        error = torch.abs(outputs - label) 
        mrae = torch.mean(error)
        return mrae

class PerpetualCertain2(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualCertain2, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        # self.fore_back = Fore_Back_dis()

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, rgb, hsi, idx):
        # 相应波段
        B, C, H, W = rgb.shape

        # idx = self.fore_back(hsi)

        prgb = torch.rand(B, C, H, W).cuda()

        prgb[:,0,:,:] = hsi[:,int(idx[0]),:,:]
        prgb[:,1,:,:] = hsi[:,int(idx[1]),:,:]
        prgb[:,2,:,:] = hsi[:,int(idx[2]),:,:]

        loss = []
        dehaze_features = self.output_features(rgb)
        gt_features = self.output_features(prgb)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    # 这里记得加到.cuda()上面
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 

# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    # print(img1.size())
    # (_, _, height, width) = img1.size()
    # channel = 1
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
        
    # print(window)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
class Loss_ssim_hyper(nn.Module):
    def __init__(self):
        super(Loss_ssim_hyper, self).__init__()

    def forward(self,hyper):
        ssim_list = []
        # reRGB = normalize(reRGB, max_val=255., min_val=0.)
        for i in range(30):
            ssim_1 = ssim(hyper[:,i:i+1,:,:],hyper[:,i+1:i+2,:,:])
            ssim_list.append(ssim_1)
        ssim_tensor = torch.Tensor(ssim_list)
        ssim_all = torch.mean(ssim_tensor)
        loss_ssim_hyper = 1 - ssim_all
        return loss_ssim_hyper

# Classes to re-use window
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
