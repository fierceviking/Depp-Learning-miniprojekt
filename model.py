import torch.nn as nn

class DecomNet(nn.Module):
    def __init__(self, kernel_size=3):
        super(DecomNet).__init__()
        padding = 1
        # Define 5 conv layers (based on the R-Net article) and 2 layers without reLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.conv5 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.conv6 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.conv7 = nn.Conv2d(64, 3, kernel_size=kernel_size, padding=padding, padding_mode='replicate')

        # Define ReLU
        self.relu = nn.ReLU()

    def forward(self, input):
        # First layer
        c1 = self.conv1(input)

        # Five layers with ReLU
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(c2))
        nn.ReLU()

class EnhanceNet(nn.Module):
    pass

class RetinexNet(nn.Module):
    pass

