import torch
from model import RetinexNet
import torch.nn.functional as F

import wandb

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


def train(model, device, train_loader, optimizer, scheduler, epoch):
    # Enable model training mode
    model.train()

    for batch_idx, (low_light, high_light) in enumerate(train_loader):
        low_light, high_light = low_light.to(device), high_light.to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass: Decomposing and enhancing the low-light image
        enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)

        # Decompose the high-light image as well (for comparison)
        R_high, I_high = model.decom_net(high_light)


        # Compute losses
        decom_loss = compute_decom_loss(low_light, high_light, 
                                        R_low, I_low, 
                                        R_high, I_high)
        enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

        loss = decom_loss + enhance_loss

        # Compute gradients with backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        if scheduler:
            scheduler.step()

        if batch_idx % 100:
            print(f"Train Epoch: {epoch}, Iteration: {batch_idx}, Train Loss: {loss.item()}")
            wandb.log({"Train loss": loss.item()})
            wandb.log({"Decom loss": decom_loss.item(), "Enhance loss": enhance_loss.item()})



def compute_gradient(img):
    grad_x = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    grad_y = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    return grad_x, grad_y

def smoothness_loss(I, R, lambda_g=10):
    grad_I_x, grad_I_y = compute_gradient(I)
    grad_R_x, grad_R_y = compute_gradient(R)

    exp_term_x = torch.exp(-lambda_g * grad_R_x)
    exp_term_y = torch.exp(-lambda_g * grad_R_y)

    smooth_x = torch.mean(torch.abs(grad_I_x * exp_term_x))
    smooth_y = torch.mean(torch.abs(grad_I_y * exp_term_y))

    smooth_loss = smooth_x + smooth_y

    return smooth_loss


def compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high):
    #I_low_concat = torch.cat((I_low, I_low, I_low), dim=1)
    #I_high_concat = torch.cat((I_high, I_high, I_high), dim=1)
    
    # Comptute losses
    reconstruction_low = F.l1_loss(R_low * I_low, low_light)
    reconstruction_high = F.l1_loss(R_high * I_high, high_light)
    recon_mutual_low = F.l1_loss(R_high * I_low, low_light)
    recon_mutual_high = F.l1_loss(R_low * I_high, high_light)

    invariable_reflectance_loss = F.l1_loss(R_low, R_high)
    smooth_loss_low = smoothness_loss(R=R_low, I=I_low)
    smooth_loss_high = smoothness_loss(R=R_high, I=I_high)

    lambda_ir = 0.001
    lambda_is = 0.1

    decom_loss = (reconstruction_low + reconstruction_high) + \
                 (lambda_ir*recon_mutual_low + lambda_ir*recon_mutual_high) + \
                 (lambda_is*smooth_loss_low + lambda_is*smooth_loss_high) + \
                 (0.01 * invariable_reflectance_loss)

    return decom_loss

def compute_enhance_loss(high_light, R_low, I_low_enhanced):
    reconstuction_loss = F.l1_loss(R_low * I_low_enhanced, high_light)
    smooth_loss_enhanced = smoothness_loss(R=R_low, I=I_low_enhanced)

    enhance_loss = reconstuction_loss + smooth_loss_enhanced
    return enhance_loss


def main():
    torch.manual_seed(42)

    R_high = torch.rand(1, 3, 128, 128)
    I_high = torch.rand(1, 1, 128, 128)
    input_high = torch.rand(1, 3, 128, 128)

    R_low = torch.rand(1, 3, 128, 128)
    I_low = torch.rand(1, 1, 128, 128)
    input_low = torch.rand(1, 3, 128, 128)

    decom = compute_decom_loss(input_low, input_high, R_low, I_low, R_high, I_high)
    print(f"Reconstruction high: {decom}")

    enhance = compute_enhance_loss(input_high, R_low, I_low)
    print(f"Enhance loss: {enhance}")

if __name__ == '__main__':
    main()

