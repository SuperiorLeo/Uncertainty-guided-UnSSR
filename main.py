import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
import random
from dataset2 import HyperDatasetValid, HyperDatasetTrain1 # Clean Data set

from architecture_spe.UnSSR import SFFormer

from architecture_spe.SAUNet import SAUNet2
from architecture_spe.SPECAT import SPECAT
from architecture_spe.HDNet import HDNet
from architecture_spe.MFormer import MFormer

# from AWAN import AWAN
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss2, Loss_RMSE, Loss_PSNR, PerpetualCertain2, LossTrainCSS2, Loss_valid,Loss_ssim_hyper
import cv2
import math
# utils
from tensorboardX import SummaryWriter
from torchvision.models import vgg16

writer = SummaryWriter()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 19 1e-4 pretrain 1e-5
parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--batchSize", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100+1, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--trade_off", type=float, default=0, help="trade_off")
parser.add_argument("--max_iter", type=float, default=94500, help="max_iter")  # patch48:380x450/32x100-534375; patch96:82x450/32x100-113906 18900/32
parser.add_argument("--outf", type=str, default="compare/HDNet", help='path log files')
opt = parser.parse_args()


def main():
    cudnn.benchmark = True
    # load dataset
    print("\nloading dataset ...")
    train_data1 = HyperDatasetTrain1(mode='train')
    print("Train set samples: ", len(train_data1))
    val_data = HyperDatasetValid(mode='valid')
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader1 = DataLoader(dataset=train_data1, batch_size=opt.batchSize, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    train_loader = [train_loader1]
    val_loader = DataLoader(dataset=val_data, batch_size=1,  shuffle=False, num_workers=2, pin_memory=True)

    # model
    # vgg16
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    
    
    print("\nbuilding models_baseline ...")
    # model = SFFormer(3,48,31,3)
    # model = MFormer(3, 31, 48, 1)
    model = HDNet(3, 31)
    

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    criterion_train = LossTrainCSS2()
    criterion_train_2 = Loss_ssim_hyper()
    criterion_train_per = PerpetualCertain2(vgg_model).cuda()
    criterion_valid_mrae = Loss_valid() #mrae
    criterion_valid_psnr = Loss_PSNR()
    criterion_valid_rmse = Loss_RMSE()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # batchsize integer times
        #
    if torch.cuda.is_available():
        model.cuda()
        criterion_train.cuda()
        criterion_train_2.cuda()
        criterion_train_per.cuda()
        criterion_valid_mrae.cuda()
        criterion_valid_psnr.cuda()
        criterion_valid_rmse.cuda()

    # Parameters, Loss and Optimizer
    start_epoch = 0
    iteration = 0
    record_val_loss = 1000
    # 原来是1e-8
    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # visualzation
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+')
    log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = ''
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    # pretrain_file = '/home/lengyihong/workspace/SSR/ours/new/train11/RealWorldResults_L1hySSIM19/1e-4/net_98epoch.pth'
    # if pretrain_file:
    #     if os.path.isfile(pretrain_file):
    #         print("=> loading checkpoint '{}'".format(pretrain_file))
    #         checkpoint = torch.load(pretrain_file)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    # start epoch
    for epoch in range(start_epoch+1, opt.end_epoch):
        start_time = time.time()
        train_loss, losses_hyper, loss_per, losses_all, iteration, lr = train(train_loader, model, criterion_train, criterion_train_2, criterion_train_per, optimizer, epoch, iteration, opt.init_lr, opt.decay_power, opt.trade_off)
        val_loss, val_rmse, val_psnr = validate(val_loader, model, criterion_valid_mrae, criterion_valid_rmse, criterion_valid_psnr)
        writer.add_scalars("loss",{"train_loss":train_loss,"losses_hyper":losses_hyper, "losses_all":losses_all},epoch+1)
        writer.add_scalars("val",{"val_loss":val_loss,"val_rmse":val_rmse, "val_psnr":val_psnr},epoch+1)

        # Save model  
        if torch.abs(val_loss - record_val_loss) < 0.0001 or val_loss < record_val_loss or epoch == (opt.end_epoch-1): 
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            if val_loss < record_val_loss:
                record_val_loss = val_loss
        # print loss
        end_time = time.time()
        epoch_time = end_time - start_time
        # TODO 这里要改记录函数
        print("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Hyper Loss: %.9f Perp Loss: %.9f All Loss: %.9f Test RMAE: %.9f Test RMSE: %.9f Test PSNR: %.9f"
              % (epoch, iteration, epoch_time, lr, train_loss, losses_hyper, loss_per, losses_all, val_loss, val_rmse, val_psnr))
        # save loss
        record_loss2(loss_csv,epoch, iteration, epoch_time, lr, train_loss, losses_hyper, loss_per, losses_all, val_loss, val_rmse, val_psnr)
        logger.info("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Hyper Loss: %.9f Perp Loss: %.9f All Loss: %.9f Test RMAE: %.9f Test RMSE: %.9f Test PSNR: %.9f "
              % (epoch, iteration, epoch_time, lr, train_loss, losses_hyper, loss_per, losses_all, val_loss, val_rmse, val_psnr))
        # 将每一个epoch的数据保存到train.log文件中
    writer.export_scalars_to_json(os.path.join(opt.outf,'train.json'))
    writer.close()

# Training
def train(train_loader, model, criterion, criterion_2, criterion_per, optimizer, epoch, iteration, init_lr, decay_power, trade_off):
    model.train()
    random.shuffle(train_loader)
    losses = AverageMeter()
    losses_hyper = AverageMeter()
    losses_per = AverageMeter()
    losses_all = AverageMeter()
    for k, train_data_loader in enumerate(train_loader):
        for i, (images, labels) in enumerate(train_data_loader):
            labels = labels.cuda()
            images = images.cuda()

            images = Variable(images)
            labels = Variable(labels)

            lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=opt.max_iter, power=decay_power)

            iteration = iteration + 1
            # Forward + Backward + Optimize
            output, idx = model(images)

            loss = criterion(output, labels, images)
            loss_hyper = criterion_2(output)
            loss_per = criterion_per(images, output, idx) 
            # loss_mask = criterion_3(before,after)

            loss_all = loss+ 0.3*loss_hyper + 0.001*loss_per
            # loss_all = loss+ 0.001*loss_hyper + 0.00001*loss_per
            # loss_all = loss+ 0.3*loss_hyper + 0.00001*loss_per
            optimizer.zero_grad()
            # scheduler.zero_grad()

            loss_all.backward()
            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            # scheduler.step()
            #  record loss
            losses.update(loss.data)
            losses_per.update(loss_per.data)
            losses_hyper.update(loss_hyper.data) #本质上还是对所有batch_size的损失取平均
            losses_all.update(loss_all.data)
            print('[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f,train_loss_hyper.avg=%.9f,train_loss_per.avg=%.9f, loss_all.avg=%.9f'
                  % (epoch, k+1, len(train_loader), iteration, lr, losses.avg, losses_hyper.avg, losses_per.avg, losses_all.avg))

    return losses.avg, losses_hyper.avg, losses_per.avg, losses_all.avg, iteration, lr


# Validate 在这里添加msae psnr两个评价指标，用于看psnr的情况
def validate(val_loader, model, criterion_mrae,criterion_rmse,criterion_psnr):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output, idx = model(input)
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)


    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
    print(torch.__version__)
