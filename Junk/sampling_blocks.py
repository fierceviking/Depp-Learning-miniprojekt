import torch.nn as nn
import torch.nn.functional as F


class DownSampling(nn.Module):
    def __init__(self, input_channels, kernel_size=3, padding=1):
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(input_channels, 64, kernel_size=kernel_size, padding=padding, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
    
class UpSampling(nn.Module):
    def __init__(self, input_channels, kernel_size=3, padding=1):
        super(UpSampling, self).__init__()

        self.conv = nn.Conv2d(input_channels, 64, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Use nearest neighbor interpolation (article)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # Scale_factor 2 to reconstruct image in original dimensions
        return self.relu(self.conv(x))
