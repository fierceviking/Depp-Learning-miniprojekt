import torch
import torch.nn as nn

class DecomNet(nn.Module):
    def __init__(self, kernel_size=3):
        super(DecomNet).__init__()
        padding = 1
        # Define 5 conv layers (based on the R-Net article) and 2 layers without reLU
        self.conv_first = nn.Conv2d(4, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate') # 4 channels: R, G, B, and illumination

        self.conv_layers = nn.Sequential(
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

        self.conv_last = nn.Conv2d(64, 4, kernel_size=kernel_size, padding=padding, padding_mode='replicate')

    def forward(self, input):
        # First layer (without ReLU)
        c1 = self.conv_first(input)

        # Five layers with ReLU
        c2 = self.conv_layers(c1)

        # Last layer (without ReLU)
        c3 = self.conv_last(c2)
        print(c3.shape)
        # Output: Reflectance (RGB) and Illumination
        # Recall tensor shape: (batch size, channels, height, width)
        #R = torch.sigmoid(c3())

class EnhanceNet(nn.Module):
    pass

class RetinexNet(nn.Module):
    pass


def main():
    # Random tensor for testing
    input_tensor = torch.rand(1, 4, 128, 128)

    decom = DecomNet()
    decom.forward(input_tensor)


if __name__ == '__main__':
    main()