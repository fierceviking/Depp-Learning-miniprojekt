import torch
import torch.nn.functional as F


# Define the gradient kernels as PyTorch tensors
smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).reshape(1, 1, 2, 2)
smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)  # Transpose for y-direction

def gradient(input_tensor, direction):
    if direction == "x":
        kernel = smooth_kernel_x.to(input_tensor.device)
    elif direction == "y":
        kernel = smooth_kernel_y.to(input_tensor.device)
    else:
        raise ValueError("Direction must be either 'x' or 'y'")
    
    # Apply convolution to calculate gradient
    gradient_output = F.conv2d(input_tensor, kernel, stride=1, padding=1)
    return torch.abs(gradient_output)

def ave_gradient(input_tensor, direction):
    # Compute the gradient in the specified direction
    gradient_output = gradient(input_tensor, direction)
    # Apply average pooling (equivalent to TensorFlow's average_pooling2d)
    return F.avg_pool2d(gradient_output, kernel_size=3, stride=1, padding=1)

def smooth(I, R):
    # Convert input_R to grayscale by taking the mean across the color channels
    input_R_gray = torch.mean(R, dim=1, keepdim=True)
    
    # Calculate gradients for input_I and input_R_gray in both x and y directions
    grad_I_x = gradient(I, "x")
    grad_I_y = gradient(I, "y")
    ave_grad_R_x = ave_gradient(input_R_gray, "x")
    ave_grad_R_y = ave_gradient(input_R_gray, "y")
    
    # Calculate smoothness loss using exponential decay factor
    smooth_x = grad_I_x * torch.exp(-10 * ave_grad_R_x)
    smooth_y = grad_I_y * torch.exp(-10 * ave_grad_R_y)
    
    # Take the mean across all dimensions to get the smoothness value
    smoothness_loss = torch.mean(smooth_x + smooth_y)
    return smoothness_loss

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


def compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high):
    #I_low_concat = torch.cat((I_low, I_low, I_low), dim=1)
    #I_high_concat = torch.cat((I_high, I_high, I_high), dim=1)
    
    # Comptute losses
    reconstruction_low = F.l1_loss(R_low * I_low, low_light)
    reconstruction_high = F.l1_loss(R_high * I_high, high_light)
    recon_mutual_low = F.l1_loss(R_high * I_low, low_light)
    recon_mutual_high = F.l1_loss(R_low * I_high, high_light)

    invariable_reflectance_loss = F.l1_loss(R_low, R_high)
    smooth_loss_low = smooth(R=R_low, I=I_low)
    smooth_loss_high = smooth(R=R_high, I=I_high)

    lambda_ir = 0.001
    lambda_is = 0.1

    decom_loss = (reconstruction_low + reconstruction_high) + \
                 (lambda_ir*recon_mutual_low + lambda_ir*recon_mutual_high) + \
                 (lambda_is*smooth_loss_low + lambda_is*smooth_loss_high) + \
                 (0.01 * invariable_reflectance_loss)

    return decom_loss

def compute_enhance_loss(high_light, R_low, I_low_enhanced):
    reconstuction_loss = F.l1_loss(R_low * I_low_enhanced, high_light)
    smooth_loss_enhanced = smooth(R=R_low, I=I_low_enhanced)

    enhance_loss = reconstuction_loss + 3 * smooth_loss_enhanced
    return enhance_loss