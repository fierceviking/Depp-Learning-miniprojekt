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
        
        x = self.prepare_input(x)

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
    
    def prepare_input(self, input_image):
        """
        This function takes an input image (with channels=3) and converts it to channels=4
        """
        # Assume input_image is a PyTorch tensor with shape [B, C, H, W]
        input_max = torch.max(input_image, dim=1, keepdim=True)[0]  # Max across channels
        input_img = torch.cat((input_max, input_image), dim=1)  # Concatenate to create 4 channels
        return input_img





class EnhanceNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(EnhanceNet, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(4, channels, kernel_size, padding=1, padding_mode='replicate')

        # Downsampling convolutions
        self.down_conv1 = nn.Conv2d(channels, channels, kernel_size, stride=2, 
                                    padding=1, padding_mode='replicate')
        self.down_conv2 = nn.Conv2d(channels, channels, kernel_size, stride=2, 
                                    padding=1, padding_mode='replicate')
        self.down_conv3 = nn.Conv2d(channels, channels, kernel_size, stride=2, 
                                    padding=1, padding_mode='replicate')

        # Upsampling convolutions
        self.up_conv1 = nn.Conv2d(channels*2, channels, kernel_size, 
                                  padding=1, padding_mode='replicate')
        self.up_conv2 = nn.Conv2d(channels*2, channels, kernel_size, 
                                  padding=1, padding_mode='replicate')
        self.up_conv3 = nn.Conv2d(channels*2, channels, kernel_size, 
                                  padding=1, padding_mode='replicate')
        
        # Fusion (1x1 kernel) and output layers
        self.fusion = nn.Conv2d(channels*3, channels, kernel_size=1,
                                padding=1, padding_mode='replicate')
        self.output = nn.Conv2d(channels, 1, kernel_size=3)

        self.relu = nn.ReLU()

    
    def forward(self, R, I):
        # Combine Left and Right inputs
        x = torch.cat((R, I), dim=1)

        # Initial convolution
        c1 = self.relu(self.conv1(x)) # Shape: 128x128

        # Downsampling
        down1 = self.relu(self.down_conv1(c1)) # Shape: 64x64
        down2 = self.relu(self.down_conv2(down1)) # Shape: 32x32
        down3 = self.relu(self.down_conv3(down2)) # Shape: 16x16

        # Upsampling and concatenation
        up1 = F.interpolate(down3, size=down2.shape[2:]) # Upsample to 32x32
        up1_conv = self.relu(self.up_conv1(torch.cat((up1, down2), dim=1))) # Concat and apply conv
        
        up2 = F.interpolate(up1_conv, size=down1.shape[2:]) # Upsample to 64x64
        up2_conv = self.relu(self.up_conv2(torch.cat((up2, down1), dim=1)))

        up3 = F.interpolate(up2_conv, size=c1.shape[2:]) # Upsample to 128x128
        up3_conv = self.relu(self.up_conv3(torch.cat((up3, c1), dim=1)))

        # Prepare for final output
        up1_conv_resized = F.interpolate(up1_conv, size=R.shape[2:])
        up2_conv_resized = F.interpolate(up2_conv, size=R.shape[2:])

        # Fusion
        concat_features = torch.cat((up1_conv_resized, up2_conv_resized, up3_conv), dim=1)
        x = self.fusion(concat_features)

        # Output
        I_enhanced = self.output(x)
        return I_enhanced



class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.decom_net = DecomNet()
        self.enhance_net = EnhanceNet()
    
    def forward(self, input_low, input_high):
        # Convert the input to tensors (with float)
        #input_low = torch.from_numpy(input_low).float().cuda()
        #input_high = torch.from_numpy(input_high).float().cuda()

        # Get the Reflectance and Illumination map in both low and high images
        R_low, I_low = self.decom_net(input_low) 
        R_high, I_high = self.decom_net(input_high)

        # Enhance illumination component for low-images
        I_low_enhanced = self.enhance_net(R_low, I_low)

        # Concat illuminations (make it have 3 channels instead of 1 to match with Reflectance)
        I_low_concat = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_concat = torch.cat((I_high, I_high, I_high), dim=1)
        I_low_enhanced_concat = torch.cat((I_low_enhanced, I_low_enhanced, I_low_enhanced), dim=1)

        # Reconstruction loss
        self.recon_low = F.l1_loss(R_low * I_low_concat, input_low)
        self.recon_high = F.l1_loss(R_high * I_high_concat, input_high)

        # Mutual reconstruction loss (ensures consistent reflectance and illuminance)
        self.recon_mutual_low = F.l1_loss(R_high * I_low_concat, input_low)
        self.recon_mutual_high = F.l1_loss(R_low * I_high_concat, input_high)

        # Invariable reflectance loss
        self.invariable_reflectance_loss = F.l1_loss(R_low, R_high)

        # Adjustment loss
        self.adjustment_loss = F.l1_loss(R_low * I_low_enhanced_concat, input_high)


        # Smoothness loss
            # Note: We need to compute gradients (horizontal and vertical) for Illumination and Reflectance Map
        self.smooth_loss_low = self.smoothness_loss(I_low, R_low)
        self.smooth_loss_high = self.smoothness_loss(I_high, R_high)
        self.smooth_loss_enhanced = self.smoothness_loss(I_low_enhanced, R_low)

        # Decom net loss: L = L_reconstruction + lambda_ir * L_invariable_reflectance_loss + lambda_is * L_is
        self.decom_loss =   (self.recon_low + self.recon_high) + \
                            (0.001 * self.recon_mutual_low + 0.001 * self.recon_mutual_high) + \
                            (0.1 * self.smooth_loss_low + 0.1 * self.smooth_loss_high) + \
                            (0.01 * self.invariable_reflectance_loss)
        
        # Enhance net loss: L_recon + L_is
        self.enhance_loss = (self.adjustment_loss + self.smooth_loss_enhanced)
        return self.decom_loss, self.enhance_loss

            
    def compute_gradient(self, img):
        grad_x = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        grad_y = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        return grad_x, grad_y
    
    def smoothness_loss(self, I, R, lambda_g=10):
        grad_I_x, grad_I_y = self.compute_gradient(I)
        grad_R_x, grad_R_y = self.compute_gradient(R)

        exp_term_x = torch.exp(-lambda_g * grad_R_x)
        exp_term_y = torch.exp(-lambda_g * grad_R_y)

        smooth_x = torch.mean(torch.abs(grad_I_x * exp_term_x))
        smooth_y = torch.mean(torch.abs(grad_I_y * exp_term_y))

        smooth_loss = smooth_x + smooth_y

        return smooth_loss


def main():
    # Random tensor for testing
    input_tensor_low = torch.rand(1, 3, 128, 128)
    input_tensor_high = torch.rand(1, 3, 128, 128)

    decom = DecomNet()
    R_low, I_low = decom.forward(input_tensor_low)
    R_high, I_high = decom.forward(input_tensor_high)

    enhance = EnhanceNet()
    I_enhanced = enhance.forward(R_low, I_low)

    print(f"Reflectance low shape: {R_low.shape}, Illumination low shape: {I_low.shape}")
    print(f"Reflectance high shape: {R_high.shape}, Illumination high shape: {I_high.shape}")
    print(f"Enhanced Illumination shape: {I_enhanced.shape}")

    # Create the retinex model
    retinex = RetinexNet()
    input_low = torch.rand(1, 3, 128, 128)
    input_high = torch.rand(1, 3, 128, 128)
    
    decom_loss, enhance_loss = retinex(input_low, input_high)
    print(f"Decomposition Loss: {decom_loss}, Enhancement Loss: {enhance_loss}")


if __name__ == '__main__':
    main()
