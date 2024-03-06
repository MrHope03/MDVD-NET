import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 conv for matching dimensions

    def forward(self, x):
        # Residual connection
        skip = self.conv_skip(self.upsample(x))

        # Main path
        x = self.upsample(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x + skip  # Element-wise addition

class ASubnet(nn.Module):
    def __init__(self, input_size, initial_feature_dim):
        super(ASubnet, self).__init__()
        self.fc = nn.Linear(input_size, initial_feature_dim * 4 * 4)  # Adjust input_size as needed

        # Define the upsampling ResBlocks with increasing channels
        self.upres_blocks = nn.Sequential(
            UpsamplingResBlock(initial_feature_dim, initial_feature_dim // 2),
            UpsamplingResBlock(initial_feature_dim // 2, initial_feature_dim // 4),
            UpsamplingResBlock(initial_feature_dim // 4, initial_feature_dim // 8),
            UpsamplingResBlock(initial_feature_dim // 8, initial_feature_dim // 16),
            UpsamplingResBlock(initial_feature_dim // 16, initial_feature_dim // 16),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)  # Reshape from (batch_size, feature_dim) to (batch_size, channels, height, width)
        x = self.upres_blocks(x)
        return x

# Instantiate the A-subnet
# Assuming LSTM features size is 1024 and initial feature dimension after FC is 256
input_size = 1024
initial_feature_dim = 256
a_subnet = ASubnet(input_size, initial_feature_dim)

# Example tensor representing LSTM features
lstm_features = torch.randn(10, input_size)  # Batch size of 10
two_d_features = a_subnet(lstm_features)
