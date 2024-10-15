import torch
import torch.nn as nn
import torch.nn.functional as F
from sampling_blocks import DownSampling, UpSampling

class DecomNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(DecomNet, self).__init__()

        # Define 5 conv layers (based on the R-Net article) and 2 layers without reLU
        self.de_conv_first = nn.Conv2d(4, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate') # 4 channels: R, G, B, and illumination

        self.de_conv_layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.ReLU()
        )

        self.de_conv_last = nn.Conv2d(64, 4, kernel_size=kernel_size, padding=padding, padding_mode='replicate')

    def forward(self, x):
        # First layer (without ReLU)
        c1 = self.de_conv_first(x)

        # Five layers with ReLU
        c2 = self.de_conv_layers(c1)

        # Last layer (without ReLU)
        c3 = self.de_conv_last(c2)
        #print(c3.shape) 
            # Output: [1, 4, 128, 128] with test tensor

        # Output: Reflectance (RGB) and Illumination
            # Recall tensor shape: (batch size, channels, height, width)
        R = torch.sigmoid(c3[:, 0:3, :, :]) # Only the 3 first channels
        I = torch.sigmoid(c3[:, 3:4, :, :]) # Only the last channel
        return R, I




class EnhanceNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1) -> None:
        super(EnhanceNet, self).__init__()

        self.downsample_1 = DownSampling(4)
        self.downsample_2 = DownSampling(64)
        self.downsample_3 = DownSampling(64)

        self.upsample_1 = UpSampling(64)
        self.upsample_2 = UpSampling(64)
        self.upsample_3 = UpSampling(64)

        self.conv1x1 = nn.Conv2d(64*3, 64, kernel_size=1) # Times 3 because we upscale three times
        self.illumination_map = nn.Conv2d(64, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        down1 = self.downsample_1(x) # 128x128 -> 64x64
        down2 = self.downsample_2(down1) # 64x64 -> 32x32
        down3 = self.downsample_3(down2) # 32x32 -> 16x16

        # Define upsampling and extracted feauture maps
        extracted_features = []
        up1 = self.upsample_1(down3) #16x16 -> 32x32
        extracted_features.append(up1)
        up2 = self.upsample_2(up1) # 32x32 -> 64x64
        extracted_features.append(up2)
        up3 = self.upsample_3(up2) # 64x64 -> 128x128
        extracted_features.append(up3)

        # Concat the features





class RetinexNet(nn.Module):
    pass


def main():
    # Random tensor for testing
    input_tensor = torch.rand(1, 4, 128, 128)

    decom = DecomNet()
    R, I = decom.forward(input_tensor)


if __name__ == '__main__':
    main()