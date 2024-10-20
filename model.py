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
        concat_features = torch.cat(extracted_features, dim=1)

        # Fuse the features using 1x1 kernel
        fused_features = self.conv1x1(concat_features)

        # Generate the illumination map (use sigmoid)
        I_enhanced = torch.sigmoid(self.illumination_map(fused_features))

        return I_enhanced


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.decom_net = DecomNet()
        self.enhance_net = EnhanceNet()
    
    def forward(self, input_low, input_high):
        # Convert the input to tensors (with float)
        input_low = torch.from_numpy(input_low).float().cuda()
        input_high = torch.from_numpy(input_high).float().cuda()


        # Get the Reflectance and Illumination map in both low and high images
        R_low, I_low = self.decom_net(input_low) 
        R_high, I_high = self.decom_net(input_high)

        # Enhance illumination component for low-images
        I_low_enhanced = self.enhance_net(I_low)

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
        self.smooth_loss_enhanced = self.smoothness_loss(I_low_enhanced, I_low_enhanced_concat)

        # Decom net loss: L = L_reconstruction + lambda_ir * L_invariable_reflectance_loss + lambda_is * L_is
        self.decom_loss =   (self.recon_low + self.recon_high) + \
                            (0.001 * self.recon_mutual_low + 0.001 * self.recon_mutual_high) + \
                            (0.1 * self.smooth_loss_low + 0.1 * self.smooth_loss_high) + \
                            (0.01 * self.invariable_reflectance_loss)
        
        # Enhance net loss: L_recon + L_is
        self.enhance_loss = (self.adjustment_loss + self.smooth_loss_enhanced)

            
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
    input_tensor_low = torch.rand(1, 4, 128, 128)
    input_tensor_high = torch.rand(1, 4, 128, 128)

    retinex_model = RetinexNet().cuda()  # Make sure to move the model to GPU if available

    # Test forward pass
    R_low, I_low = retinex_model.decom_net(input_tensor_low)
    R_high, I_high = retinex_model.decom_net(input_tensor_high)
    I_enhanced = retinex_model.enhance_net(I_low)

    print(f"Reflectance low shape: {R_low.shape}, Illumination low shape: {I_low.shape}")
    print(f"Reflectance high shape: {R_high.shape}, Illumination high shape: {I_high.shape}")
    print(f"Enhanced Illumination shape: {I_enhanced.shape}")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()