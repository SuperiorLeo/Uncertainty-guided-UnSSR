import torch
import torch.nn as nn
from math import sqrt

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=True)

class conv_relu_res_block(nn.Module):
    def __init__(self):
        super(conv_relu_res_block, self).__init__()
        self.conv1 = conv3x3(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.add(out,residual) 
        return out
    
class conv_relu_res_block_scale01(nn.Module):
    def __init__(self):
        super(conv_relu_res_block_scale01, self).__init__()
        self.conv1 = conv3x3(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.mul(out,0.1) 
        out = torch.add(out,residual) 
        return out
    
class conv_relu_res_relu_block(nn.Module):
    def __init__(self):
        super(conv_relu_res_relu_block, self).__init__()
        self.conv1 = conv3x3(64, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = torch.add(out,residual) 
        out = self.relu2(out)
        return out

class conv_relu_res_relu_block_scale01(nn.Module):
    def __init__(self):
        super(conv_relu_res_relu_block_scale01, self).__init__()
        self.conv1 = conv3x3(64, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = torch.mul(out,0.1) 
        out = torch.add(out,residual) 
        out = self.relu2(out)
        return out
    
class resblock(nn.Module):
    def __init__(self, block, block_num, input_channel, output_channel):
        super(resblock, self).__init__()

        self.in_channels = input_channel
        self.out_channels = output_channel
        self.input_conv = conv3x3(self.in_channels, out_channels=64)  
        self.conv_seq = self.make_layer(block, block_num)
        self.conv = conv3x3(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = conv3x3(in_channels=64,  out_channels=self.out_channels)
        self.sort = Fore_Back_dis()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,sqrt(2./n))# the devide  2./n  carefully  
                
    def make_layer(self,block,num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block()) # there is a () 
        return nn.Sequential(*layers)   
    
    def forward(self, x):
       
        out = self.input_conv(x)
        residual = out
        out = self.conv_seq(out)
        out = self.conv(out)
        out = torch.add(out,residual)  
        out = self.relu(out)
        out = self.output_conv(out)

        idx = self.sort(out)
        return out,idx

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