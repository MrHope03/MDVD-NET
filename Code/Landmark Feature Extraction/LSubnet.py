import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
class LSubnet(nn.Module):
    def __init__(self):
        super(LSubnet, self).__init__()
        self.layers = nn.Sequential(
            Convolution(3, 64),
            Convolution(64, 128, downsampling=True),
            Convolution(128, 256, downsampling=True),
            Convolution(256, 256, downsampling=True),
            Convolution(256, 256, downsampling=True),
            Convolution(256, 256, upsampling=True),
            Convolution(256, 256, upsampling=True),
            Convolution(256, 128, upsampling=True),
            Convolution(128, 64, upsampling=True),
        )

    def forward(self, x):
        return self.layers(x)
