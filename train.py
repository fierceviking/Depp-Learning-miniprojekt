import torch
from model import RetinexNet
from loss_functions import compute_enhance_loss, compute_decom_loss
import torch.nn.functional as F


import wandb



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
        enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced, enhanced_image)

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


def validate(model, device, vali_loader):
    model.eval()
    val_loss = 0

    # Skip calculating gradients
    with torch.no_grad():
        for batch_idx, (low_light, high_light) in enumerate(vali_loader):
            low_light, high_light = low_light.to(device), high_light.to(device)
            
            # Forward pass: Decomposing and enhancing the low-light image
            enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)

            # Decompose the high-light image as well (for comparison)
            R_high, I_high = model.decom_net(high_light)


            # Compute losses
            decom_loss = compute_decom_loss(low_light, high_light, 
                                            R_low, I_low, 
                                            R_high, I_high)
            enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced, enhanced_image)

            loss = decom_loss + enhance_loss
            val_loss += loss
        
        avg_val_loss = val_loss / len(vali_loader)
        print(f"\nAverage validation loss: {avg_val_loss}")
        wandb.log({"Average validation loss": avg_val_loss})

        # Add PSNR and SSIM

        return avg_val_loss


def main():
    torch.manual_seed(42)



if __name__ == '__main__':
    main()

"""
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

