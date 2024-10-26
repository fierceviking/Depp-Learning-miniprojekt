import torch
import torch.nn.functional as F

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

    enhance_loss = reconstuction_loss + 3 * smooth_loss_enhanced
    return enhance_loss