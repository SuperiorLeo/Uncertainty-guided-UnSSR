import torch
import os
import numpy as np
import cv2

# from architecture_spe.Restormer import Restormer
# from architecture_spe.MST_Plus_Plus import MST_Plus_Plus
# # from architecture_spe.ourNT2022 import AWAN
# from architecture_spe.AWAN import AWAN
# from architecture_spe.hrnet import SGN
# from architecture_spe.model2021 import reconnet
# from architecture_spe.SAUNet import SAUNet2
# from architecture_spe.SPECAT import SPECAT
from architecture_spe.UnSSR import SFFormer
from architecture_spe.HDNet import HDNet
from architecture_spe.LTRN import UDoubleTransformerCP3

import glob
import hdf5storage as hdf5
from utilsfold.utils import reconstruction_patch_image_gpu, save_matv73
import time
# from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# from skimage.measure import  compare_psnr,compare_ssim

def compute_MRAE(gt, rec):
    gt_hyper = gt
    rec_hyper = rec
    error = np.abs(rec_hyper - gt_hyper) / gt_hyper
    mrae = np.mean(error.reshape(-1))
    return mrae

def compute_RMSE(gt, rec):
    error = np.power(gt - rec, 2)
    rmse = np.sqrt(np.mean(error))
    return rmse

def get_reconstruction_gpu(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cuda()
    with torch.no_grad():
        start_time = time.time()
        # var_output1 = model(var_input[:,:,:-2,:])
        # var_output2 = model(var_input[:,:,2:,:])
        # var_output = torch.cat([var_output1, var_output2[:,:,-2:,:]], 2)
        var_output,before,after= model(var_input) #ours
        # var_output = model(var_input)
        end_time = time.time()

    return end_time-start_time, var_output.cpu()

def get_reconstruction_gpu2(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cuda()
    with torch.no_grad():
        start_time = time.time()
        # cave 数据集 512 512
        var_output, idx = model(var_input)
        # var_output,before,after = model(var_input)
        img_res = var_output.cpu().numpy() * 1.0
        img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
        img_res_limits = np.minimum(img_res, 1.0)
        img_res_limits = np.maximum(img_res_limits, 0)
        img_res = img_res_limits
        end_time = time.time()

    return end_time-start_time, img_res, idx


def compute_psnr(label,output):
    
    # assert self.label.ndim == 3 and self.output.ndim == 3

    img_c, img_w, img_h = label.shape
    ref = label.reshape(img_c, -1)
    tar = output.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max1 = np.max(ref, 1)

    psnrall = 10 * np.log10(1 / msr) # 这里是没开根号的rmse 所以相当于psnr公式log里面提出了一个1/2 
    out_mean = np.mean(psnrall)
    # return out_mean, max1
    return out_mean

def compute_ergas(self, scale=32):
    d = self.label - self.output
    ergasroot = 0
    for i in range(d.shape[0]):
        ergasroot = ergasroot + np.mean(d[i, :, :] ** 2) / np.mean(self.label[i, :, :]) ** 2
    ergas = (100 / scale) * np.sqrt(ergasroot/(d.shape[0]+1))
    return ergas

def compute_sam(label,output):
    # assert self.label.ndim == 3 and self.label.shape == self.label.shape

    c, w, h = label.shape
    x_true = label.reshape(c, -1)
    x_pred = output.reshape(c, -1)

    x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

    sam = np.arccos(sam) * 180 / np.pi
    # sam = np.arccos(sam)
    mSAM = sam.mean()
    var_sam = np.var(sam)
    # return mSAM, var_sam
    return mSAM

def compute_ssim(label,output):
    """
    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    # data_range=1
    # multidimension=False
    mssim =[]
    for i in range(label.shape[0]):
        mssim.append(structural_similarity(label[i, :, :], output[i, :, :]))
    # mssim = [
    #     structural_similarity(X=label[i, :, :], Y=output[i, :, :], data_range=data_range, multidimension=multidimension)
    #     for i in range(label.shape[0])]

    return np.mean(mssim)

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = '/home/lengyihong/workspace/SSR/ours/Tnnls2025/Tnnls_github/pth/cave_ours.pth'
result_path = '/home/lengyihong/workspace/SSR/ours/Tnnls2025/Tnnls_github/pth/results/'
img_path = '/home/data/SSROriDataset/CVAE/Valid_rgb/'
gt_path = '/home/data/SSROriDataset/CVAE/Valid_mat/'
var_name = 'rad'
# bands = hdf5.loadmat('/home/temp/ntire2022/NTIRE2022/Train_spectral/ARAD_1K_0002.mat')['bands']
# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)

# model = Restormer(3,31,48)
# model = AWAN(3,31,48,1)
model = SFFormer(3,48,31,3)
# model = HDNet(3,31)
# model = AWAN(3, 31, 90, 1)
# model = AWAN(3, 31, 90, 8)
# model = SGN(3, 31, 64)
# model = MST_Plus_Plus(3, 31, 31, 3)
# model = AWAN(3, 31, 100, 10) #drcr
# model = HSCNN_Plus(3, 31, 16)
# model = reconnet()
# model = resblock(conv_relu_res_relu_block, 16, 3,31)
# model = hslnet(3,31,90)

# model = UDoubleTransformerCP3(3, 128, 128, 32, 31, 4, 1, "RTGB3",3)

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()

img_path_name = glob.glob(os.path.join(img_path, '*.png'))
img_path_name.sort()
# gt_hyper_name = glob.glob(os.path.join(gt_path, '*.mat'))
# gt_hyper_name.sort()
mrae = []
rmse = []
psnr = []
ssim = []
sam = []
for i in range(len(img_path_name)):
    # load rgb images
    rgb = cv2.imread(img_path_name[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # rgb = np.float32(rgb) / 255
    rgb = normalize(np.float32(rgb), max_val=255., min_val=0.) # 与上面的是一样的意思和作用
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    # print(img_path_name[i].split('/')[-1])
    name = img_path_name[i].split('/')[-1]

    # _, img_res = reconstruction_patch_image_gpu(rgb, model, 128, 128)
    # _, img_res_overlap = reconstruction_patch_image_gpu(rgb[:, :, 128//2:, 128//2:], model, 128, 128)
    # img_res[128//2:, 128//2:, :] = (img_res[128//2:, 128//2:, :] + img_res_overlap) / 2.0

    # rgbFlip = np.flip(rgb, 2).copy()
    # _, img_resFlip = reconstruction_patch_image_gpu(rgbFlip, model, 128, 128)
    # _, img_res_overlapFlip = reconstruction_patch_image_gpu(rgbFlip[:, :, 128 // 2:, 128 // 2:], model, 128, 128)
    # img_resFlip[128 // 2:, 128 // 2:, :] = (img_resFlip[128 // 2:, 128 // 2:, :] + img_res_overlapFlip) / 2.0
    # img_resFlip = np.flip(img_resFlip, 0)
    # img_res = (img_res + img_resFlip) / 2
    # img_res = np.clip(img_res, 0, 1)
    
    # # type2
    rgb = torch.from_numpy(rgb).float()
    _,img_res, idx = get_reconstruction_gpu2(rgb,model)
    print('name: ' + name + ' idx: ' + str(idx))

    hyper_name = img_path_name[i].split('/')[-1].split('.png')[0] + '.mat'
    hyper_path = os.path.join(gt_path,hyper_name)
    gt_hyper = hdf5.loadmat(hyper_path)['rad']
    # print(hyper_path)
    # print(img_path_name[i])
    # gt_hyper = hdf5.loadmat(gt_hyper_name[i])['rad']
    gt_hyper = np.float32(np.array(gt_hyper)/(2**16-1))
    # gt_hyper = normalize(gt_hyper, max_val=1., min_val=0.)

    # mrae.append(compute_MRAE(gt_hyper, img_res))
    rmse.append(compute_RMSE(gt_hyper, img_res))
    gt_hyper1 = np.transpose(gt_hyper,[2,1,0])
    img_res1 = np.transpose(img_res,[2,1,0])
    psnr.append(compute_psnr(gt_hyper1, img_res1))
    ssim.append(compute_ssim(gt_hyper1, img_res1))
    sam.append(compute_sam(gt_hyper1, img_res1))

    mat_name = img_path_name[i].split('/')[-1][:-4] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name, img_res)
    # save_matv73(mat_dir, "bands", bands)
# print('mrae: '+str(sum(mrae)/len(mrae)))
print('rmse: '+str(sum(rmse)/len(rmse)))
print('psnr: '+str(sum(psnr)/len(psnr)))
print('ssim: '+str(sum(ssim)/len(ssim)))
print('sam: '+str(sum(sam)/len(sam)))


