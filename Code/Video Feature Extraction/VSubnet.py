import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return x
    
class Convolution(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size = 3, padding=1, stride = 1, upsampling=False, downsampling=False, sampling_factor=2):
        super(Convolution, self).__init__()
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=kernel_size, padding=padding, stride=stride)
        self.upsample = nn.Upsample(scale_factor=sampling_factor, mode='nearest')
        self.downsample = nn.Conv2d(outChannel, outChannel, kernel_size=kernel_size, padding=padding, stride=sampling_factor)
        self.batchnorm = nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if(self.upsampling):
            x = self.upsample(x)
        x = self.relu(self.batchnorm(self.conv(x)))
        if(self.downsampling):
            x = self.downsample(x)
        return x
    
class VSubnet(nn.Module):
    def __init__(self):
        super(VSubnet, self).__init__()
        self.initial_layers = nn.Sequential(
            Convolution(3, 64),
            Convolution(64, 128, downsampling=True),
            Convolution(128, 256, downsampling=True),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        self.final_layers = nn.Sequential(
            Convolution(256, 128, upsampling=True),
            Convolution(128, 64, upsampling=True),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.res_blocks(x)
        x = self.final_layers(x)
        return x
