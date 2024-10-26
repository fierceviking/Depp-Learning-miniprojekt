import torch
from model import RetinexNet
from loss_functions import compute_enhance_loss, compute_decom_loss
import torch.nn.functional as F
from torchvision import transforms
from data_loader import LOLDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

import wandb



def train(model, device, train_loader, optimizer, epoch):
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

        #scheduler.step()

        if batch_idx % 100 == 0:
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
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(vali_loader)
        print(f"\nAverage validation loss: {avg_val_loss}")
        wandb.log({"Average validation loss": avg_val_loss})


def main():
    torch.manual_seed(42)
    # Create directory if it doesn't exist
    os.makedirs("models/job_2", exist_ok=True)

    num_epochs = 100
    batch_size = 16
    learning_rate = 1e-3

    # Set compute environment
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Start new wandb run
    wandb.login()
    wandb.init(project="DL_miniproject")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the same size
        transforms.ToTensor(),          # Convert to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std deviation
    ])

    # Define training and validation set
    train_low_dir = 'train_data/low'
    train_high_dir = 'train_data/high'
    vali_low_dir = 'vali_data/low'
    vali_high_dir = 'vali_data/high'
    patch_size = (96,96)

    train_data = LOLDataset(train_low_dir, train_high_dir, transform=transform, patch_size=patch_size)
    vali_data = LOLDataset(vali_low_dir, vali_high_dir, transform=transform, patch_size=patch_size)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    vali_loader = DataLoader(vali_data, batch_size=batch_size, shuffle=False)

    # Define model
    model = RetinexNet().to(device)

    # Define optimizers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    # Train and test
    for epoch in range(0, num_epochs+1):
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr}")
        train(model, device, train_loader, optimizer, epoch)
        validate(model, device, vali_loader)

        wandb.log({'Epoch': epoch})
        wandb.log({'Learning_rate': current_lr})
        scheduler.step()

        # Save model per 20 epoch
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"models/job_2/RetinexNet_epoch{epoch}.pt")

if __name__ == '__main__':
    main()

