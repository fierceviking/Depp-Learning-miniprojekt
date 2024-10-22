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