"""
Placeholder for old lines of code
"""

""" 
Old forward for Retinex
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

        """

"""
Old loss
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

        return self.decom_loss, self.enhance_loss """


"""
For testing during development
    R_high = torch.rand(1, 3, 128, 128)
    I_high = torch.rand(1, 1, 128, 128)
    input_high = torch.rand(1, 3, 128, 128)

    R_low = torch.rand(1, 3, 128, 128)
    I_low = torch.rand(1, 1, 128, 128)
    input_low = torch.rand(1, 3, 128, 128)
    enhanced_image = torch.rand(1, 3, 128, 128)

    decom = compute_decom_loss(input_low, input_high, R_low, I_low, R_high, I_high)
    print(f"Reconstruction high: {decom}")

    enhance = compute_enhance_loss(input_high, R_low, I_low, enhanced_image)
    print(f"Enhance loss: {enhance}")

"""
# Old loss functions smooth
# def compute_gradient(img):
#     grad_x = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
#     grad_y = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
#     return grad_x, grad_y

# def smoothness_loss(I, R, lambda_g=10):
#     grad_I_x, grad_I_y = compute_gradient(I)
#     grad_R_x, grad_R_y = compute_gradient(R)

#     exp_term_x = torch.exp(-lambda_g * grad_R_x)
#     exp_term_y = torch.exp(-lambda_g * grad_R_y)

#     smooth_x = torch.mean(torch.abs(grad_I_x * exp_term_x))
#     smooth_y = torch.mean(torch.abs(grad_I_y * exp_term_y))

#     smooth_loss = smooth_x + smooth_y

#     return smooth_loss



import os
train_high_names = os.listdir('train_data/high')
print("Length of train data: ", len(train_high_names))

val_high_names = os.listdir('vali_data/high')
print("Length of validation data: ", len(val_high_names))