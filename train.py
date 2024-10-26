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



def train_decom_net(model, device, train_loader, optimizer, epoch):
    # Enable model training mode
    model.train()

    for batch_idx, (low_light, high_light) in enumerate(train_loader):
        low_light, high_light = low_light.to(device), high_light.to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass: Decomposing the images
        R_low, I_low = model(low_light)
        R_high, I_high = model(high_light)


        # Compute loss
        decom_loss = compute_decom_loss(low_light, high_light, 
                                        R_low, I_low, 
                                        R_high, I_high)

        # Backpropagation + weight update
        decom_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch}, Iteration: {batch_idx}, Decomposition loss: {decom_loss.item()}")
            wandb.log({"Decom loss": decom_loss.item()})


def train_enhance_net(model, device, train_loader, optimizer, epoch):
    # Freeze DecomNet parameters
    for param in model.decom_net.parameters():
        param.requires_grad = False
    

    # Enable model training mode
    model.train()

    for batch_idx, (low_light, high_light) in enumerate(train_loader):
        low_light, high_light = low_light.to(device), high_light.to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass: Decomposing and enhancing the low-light image
        R_low, I_low, I_low_enhanced = model(low_light)

        enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

        # Backpropagation + weight update
        enhance_loss.backward()
        optimizer.step()


        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch}, Iteration: {batch_idx}, Enhance Loss: {enhance_loss.item()}")
            wandb.log({"Enhance loss": enhance_loss.item()})

def fine_tune(model, device, train_loader, optimizer, epoch):   
    # Enable model training mode
    model.train()

    for batch_idx, (low_light, high_light) in enumerate(train_loader):
        low_light, high_light = low_light.to(device), high_light.to(device)

        # Zero out gradients
        optimizer.zero_grad()

    
        # Forward pass: Decomposing and enhancing the low-light image
        enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)
        R_high, I_high = model.decom_net(high_light)

        enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)
        decom_loss = compute_decom_loss(low_light, high_light, 
                                        R_low, I_low, 
                                        R_high, I_high)
        loss = decom_loss + enhance_loss

        # Backpropagation + weight update
        loss.backward()
        optimizer.step()


        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch}, Iteration: {batch_idx}, Train Loss: {loss.item()}")
            wandb.log({"Train loss": loss.item()})

def validate(model, device, vali_loader, stage='both'):
    model.eval()
    val_loss = 0

    # Skip calculating gradients
    with torch.no_grad():
        for batch_idx, (low_light, high_light) in enumerate(vali_loader):
            low_light, high_light = low_light.to(device), high_light.to(device)
            
            if stage == 'decom':
                # Forward pass: Only Decomposing the low-light image
                R_low, I_low = model.decom_net(low_light)
                R_high, I_high = model.decom_net(high_light)

                # Compute losses
                decom_loss = compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high)
                val_loss += decom_loss.item()

            elif stage == 'enhance':
                # Forward pass: Decomposing and enhancing the low-light image
                enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)

                # Decompose the high-light image as well (for comparison)
                R_high, I_high = model.decom_net(high_light)

                # Compute enhance loss
                enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)
                val_loss += enhance_loss.item()

            elif stage == 'both':
                # Forward pass: Decomposing and enhancing the low-light image
                enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)

                # Decompose the high-light image as well (for comparison)
                R_high, I_high = model.decom_net(high_light)

                # Compute both losses
                decom_loss = compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high)
                enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

                loss = decom_loss + enhance_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(vali_loader)
        print(f"\nAverage validation loss: {avg_val_loss}")
        wandb.log({"Average validation loss": avg_val_loss})


def main():
    torch.manual_seed(42)
    # Create directory if it doesn't exist
    os.makedirs("models/DecomNet", exist_ok=True)
    os.makedirs("models/EnhanceNet", exist_ok=True)
    os.makedirs("models/FineTuning", exist_ok=True)

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

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the same size
        transforms.ToTensor(), # Convert to PyTorch tensor
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std deviation
    ])

    vali_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define training and validation set
    train_low_dir = 'train_data/low'
    train_high_dir = 'train_data/high'
    vali_low_dir = 'vali_data/low'
    vali_high_dir = 'vali_data/high'
    patch_size = (96,96)

    train_data = LOLDataset(train_low_dir, train_high_dir, transform=train_transform, patch_size=patch_size)
    vali_data = LOLDataset(vali_low_dir, vali_high_dir, transform=vali_transform, patch_size=patch_size)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    vali_loader = DataLoader(vali_data, batch_size=batch_size, shuffle=False)

    # Stage 1: Train DecomNet
    model = RetinexNet(train_decom_only=True).to(device)

    # Define optimizers and schedulers
    optimizer_decom = optim.Adam(model.decom_net.parameters(), lr=learning_rate)
    optimizer_enhance = optim.Adam(model.enhance_net.parameters(), lr=learning_rate)
    optimizer_both = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8, weight_decay=0.0001)

    # scheduler_decom = StepLR(optimizer_decom, step_size=10, gamma=0.5)
    # scheduler_enhance = StepLR(optimizer_enhance, step_size=10, gamma=0.5)
    # scheduler_both = StepLR(optimizer_both, step_size=10, gamma=0.5)

    for epoch in range(num_epochs+1):
        # current_lr = scheduler_decom.get_last_lr()[0]
        # print(f"Learning rate: {current_lr}")
        train_decom_net(model, device, train_loader, optimizer_decom, epoch)
        validate(model, device, vali_loader, stage='decom')
        wandb.log({'Epoch': epoch})
        # wandb.log({'Learning_rate': current_lr})
        # scheduler_decom.step()

        if epoch % 10 == 0:
            # Save DecomNet's parameters
            torch.save(model.decom_net.state_dict(), f'models/DecomNet/DecomNet_trained_2_{epoch}.pt')


    # Stage 2: Train EnhanceNet
    # Load DecomNet's parameters
    model = RetinexNet(train_decom_only=False, train_enhance_only=True).to(device)
    model.decom_net.load_state_dict(torch.load(f'models/DecomNet/DecomNet_trained_2_{num_epochs}.pt')) 

    for epoch in range(num_epochs+1):
        # current_lr = scheduler_enhance.get_last_lr()[0]
        # print(f"Learning rate: {current_lr}")
        train_enhance_net(model, device, train_loader, optimizer_enhance, epoch)
        validate(model, device, vali_loader, stage='enhance')
        wandb.log({'Epoch': epoch})
        # wandb.log({'Learning_rate': current_lr})
        # scheduler_enhance.step()

        if epoch % 10 == 0:
            # Save EnhanceNet's parameters
            torch.save(model.enhance_net.state_dict(), f'models/EnhanceNet/EnhanceNet_trained_2_{epoch}.pt')

    # Stage 3: Fine-tuning
    model = RetinexNet().to(device)
    model.decom_net.load_state_dict(torch.load(f'models/DecomNet/DecomNet_trained_2_{num_epochs}.pt')) 
    model.enhance_net.load_state_dict(torch.load(f'models/EnhanceNet/EnhanceNet_trained_2_{num_epochs}.pt'))

    for epoch in range(num_epochs+1):
        # current_lr = scheduler_enhance.get_last_lr()[0]
        # print(f"Learning rate: {current_lr}")
        fine_tune(model, device, train_loader, optimizer_both, epoch)
        validate(model, device, vali_loader, stage='both')
        wandb.log({'Epoch': epoch})
        # wandb.log({'Learning_rate': current_lr})
        # scheduler_both.step()

        if epoch % 10 == 0:
            # Save EnhanceNet's parameters
            torch.save(model.state_dict(), f'models/FineTuning/FineTuning_2_{epoch}.pt')


if __name__ == '__main__':
    main()

