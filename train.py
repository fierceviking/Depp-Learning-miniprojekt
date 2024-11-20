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


def train_models(model, device, train_loader, optimizer, epoch, stage):
    model.train()

    if stage == 'enhance':
        for param in model.decom_net.parameters():
            param.requires_grad = False

    for batch_idx, (low_light, high_light) in enumerate(train_loader):
        low_light, high_light = low_light.to(device), high_light.to(device)

        optimizer.zero_grad()

        # Specify training stages
        if stage == 'decom':
            # Forward pass: Decomposing the images
            R_low, I_low = model(low_light)
            R_high, I_high = model(high_light)

            loss = compute_decom_loss(low_light, high_light, 
                                            R_low, I_low, 
                                            R_high, I_high)
        elif stage == 'enhance':
            # Forward pass: Enhance the low-light image
            R_low, I_low, I_low_enhanced = model(low_light)

            loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

        elif stage == 'finetune':
            # Forward pass: Decomposing and enhancing the low-light image
            enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)
            R_high, I_high = model.decom_net(high_light)

            enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)
            decom_loss = compute_decom_loss(low_light, high_light, 
                                            R_low, I_low, 
                                            R_high, I_high)

            weight = 0.1
            loss = decom_loss + (weight*enhance_loss)
        
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch}, Iteration: {batch_idx}, {stage.title()} Loss: {loss.item()}")
            wandb.log({f"{stage.title()} loss": loss.item()})


def validate(model, device, vali_loader, stage='finetune'):
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

                decom_loss = compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high)
                val_loss += decom_loss.item()

            elif stage == 'enhance':
                # Forward pass: Decomposing and enhancing the low-light image
                R_low, I_low, I_low_enhanced = model(low_light)

                # Decompose the high-light image as well (for comparison)
                R_high, I_high = model.decom_net(high_light)

                enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)
                val_loss += enhance_loss.item()

            elif stage == 'finetune':
                # Forward pass: Decomposing and enhancing the low-light image
                enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)

                # Decompose the high-light image as well (for comparison)
                R_high, I_high = model.decom_net(high_light)

                decom_loss = compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high)
                enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

                weight = 0.1
                loss = decom_loss + (weight*enhance_loss)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(vali_loader)
        print(f"\nAverage validation loss: {avg_val_loss}")
        wandb.log({"Average validation loss": avg_val_loss})


def main():
    torch.manual_seed(42)

    # Specify stages
    stage = 'decom'
    #stage = 'enhance'
    #stage = 'finetune'

    # Specify num test
    num_test = 8 # Only used for file-naming


    # Create directory if it doesn't exist (for snapshots)
    os.makedirs(f"snapshots/DecomNet/Job_{num_test}", exist_ok=True)
    os.makedirs(f"snapshots/EnhanceNet/Job_{num_test}", exist_ok=True)
    os.makedirs(f"snapshots/FineTuning/Job_{num_test}", exist_ok=True)

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
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    vali_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Define training and validation path
    train_low_dir = 'data/train_data/low'
    train_high_dir = 'data/train_data/high'

    vali_low_dir = 'data/vali_data/low'
    vali_high_dir = 'data/vali_data/high'


    patch_size = (96,96)

    train_data = LOLDataset(train_low_dir, train_high_dir, transform=train_transform, patch_size=patch_size)
    vali_data = LOLDataset(vali_low_dir, vali_high_dir, transform=vali_transform, patch_size=patch_size)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    vali_loader = DataLoader(vali_data, batch_size=batch_size, shuffle=False)


    if stage == 'decom':
        model = RetinexNet(train_decom_only=True, train_enhance_only=False).to(device)

    elif stage == 'enhance':
        model = RetinexNet(train_decom_only=False, train_enhance_only=True).to(device)
        model.decom_net.load_state_dict(torch.load(f'snapshots/DecomNet/Job_{num_test}/DecomNet_trained_{num_test}_{num_epochs}.pt')) 
        
    elif stage == 'finetune':
        model = RetinexNet(train_decom_only=False, train_enhance_only=False).to(device)
        model.decom_net.load_state_dict(torch.load(f'snapshots/DecomNet/Job_{num_test}/DecomNet_trained_{num_test}_{num_epochs}.pt')) 
        model.enhance_net.load_state_dict(torch.load(f'snapshots/EnhanceNet/Job_{num_test}/EnhanceNet_trained_{num_test}_{num_epochs}.pt'))

    # Define optimizers and schedulers
    optimizers = {
        'decom': optim.Adam(model.decom_net.parameters(), lr=learning_rate),
        'enhance': optim.Adam(model.enhance_net.parameters(), lr=learning_rate),
        'finetune': optim.Adam(model.parameters(), lr=learning_rate)
    }

    schedulers = {
        'decom': StepLR(optimizers['decom'], step_size=20, gamma=0.5),
        'enhance': StepLR(optimizers['enhance'], step_size=20, gamma=0.5),
        'finetune': StepLR(optimizers['finetune'], step_size=20, gamma=0.5)
    }

   
    # Define training loop
    for epoch in range(num_epochs+1):
        current_lr = schedulers[stage].get_last_lr()[0]
        print(f"Learning rate: {current_lr}")
        train_models(model, device, train_loader, optimizers[stage], epoch, stage)
        validate(model, device, vali_loader, stage) 
        wandb.log({'Epoch': epoch})
        wandb.log({'Learning_rate': current_lr})
        schedulers[stage].step()
        
        if epoch % 10 == 0: # Changed from 20 to 10 since finetuning only gets 50 epochs.
            if stage == 'decom':
                torch.save(model.decom_net.state_dict(), f'snapshots/DecomNet/Job_{num_test}/DecomNet_trained_{num_test}_{epoch}.pt')
            elif stage == 'enhance':
                torch.save(model.enhance_net.state_dict(), f'snapshots/EnhanceNet/Job_{num_test}/EnhanceNet_trained_{num_test}_{epoch}.pt')
            elif stage == 'finetune':
                torch.save(model.state_dict(), f'snapshots/FineTuning/Job_{num_test}/FineTuning_{num_test}_{epoch}.pt')

   
if __name__ == '__main__':
    main()

