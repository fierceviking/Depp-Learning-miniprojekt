"""
Placeholder for old lines of code
"""

""" def forward(self, input_low, input_high):
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

"""self.recon_low = F.l1_loss(R_low * I_low_concat, input_low)
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