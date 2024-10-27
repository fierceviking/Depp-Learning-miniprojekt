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

        # Zero out gradients
        optimizer.zero_grad()

        if stage == 'decom':
            # Forward pass: Decomposing the images
            R_low, I_low = model(low_light)
            R_high, I_high = model(high_light)

            # Compute loss
            loss = compute_decom_loss(low_light, high_light, 
                                            R_low, I_low, 
                                            R_high, I_high)
        elif stage == 'enhance':
            # Forward pass: Enhance the low-light image
            R_low, I_low, I_low_enhanced = model(low_light)

            loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

        elif stage == 'finetune':
            # Fine tuning
            # Forward pass: Decomposing and enhancing the low-light image
            enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)
            R_high, I_high = model.decom_net(high_light)

            enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)
            decom_loss = compute_decom_loss(low_light, high_light, 
                                            R_low, I_low, 
                                            R_high, I_high)
            weight = 0.1
            loss = decom_loss + weight*enhance_loss
        
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

                # Compute losses
                decom_loss = compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high)
                val_loss += decom_loss.item()

            elif stage == 'enhance':
                # Forward pass: Decomposing and enhancing the low-light image
                R_low, I_low, I_low_enhanced = model(low_light)

                # Decompose the high-light image as well (for comparison)
                R_high, I_high = model.decom_net(high_light)

                # Compute enhance loss
                enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)
                val_loss += enhance_loss.item()

            elif stage == 'finetune':
                # Forward pass: Decomposing and enhancing the low-light image
                enhanced_image, R_low, I_low, I_low_enhanced = model(low_light)

                # Decompose the high-light image as well (for comparison)
                R_high, I_high = model.decom_net(high_light)

                # Compute both losses
                decom_loss = compute_decom_loss(low_light, high_light, R_low, I_low, R_high, I_high)
                enhance_loss = compute_enhance_loss(high_light, R_low, I_low_enhanced)

                weight = 0.1
                loss = decom_loss + weight*enhance_loss
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
    num_test = 9


    # Create directory if it doesn't exist
    os.makedirs(f"models/DecomNet/Job_{num_test}", exist_ok=True)
    os.makedirs(f"models/EnhanceNet/Job_{num_test}", exist_ok=True)
    os.makedirs(f"models/FineTuning/Job_{num_test}", exist_ok=True)

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
        transforms.Normalize((0, 0, 0), (1, 1, 1))  # Normalize min/max scaling
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std deviation
    ])

    vali_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Define training and validation set
    # train_low_dir = 'train_data/low'
    # train_high_dir = 'train_data/high'
    train_low_dir = 'full_train/low'
    train_high_dir = 'full_train/high'

    vali_low_dir = 'vali_data/low'
    vali_high_dir = 'vali_data/high'
    patch_size = (96,96)

    train_data = LOLDataset(train_low_dir, train_high_dir, transform=train_transform, patch_size=patch_size)
    # vali_data = LOLDataset(vali_low_dir, vali_high_dir, transform=vali_transform, patch_size=patch_size)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    # vali_loader = DataLoader(vali_data, batch_size=batch_size, shuffle=False)


    if stage == 'decom':
        model = RetinexNet(train_decom_only=True, train_enhance_only=False).to(device)

    elif stage == 'enhance':
        model = RetinexNet(train_decom_only=False, train_enhance_only=True).to(device)
        model.decom_net.load_state_dict(torch.load(f'models/DecomNet/Job_{num_test}/DecomNet_trained_{num_test}_{num_epochs}.pt')) 
        
    elif stage == 'finetune':
        model = RetinexNet(train_decom_only=False, train_enhance_only=False).to(device)
        model.decom_net.load_state_dict(torch.load(f'models/DecomNet/Job_{num_test}/DecomNet_trained_{num_test}_{num_epochs}.pt')) 
        model.enhance_net.load_state_dict(torch.load(f'models/EnhanceNet/Job_{num_test}/EnhanceNet_trained_{num_test}_{num_epochs}.pt'))

    # Define optimizers and schedulers
    optimizers = {
        'decom': optim.Adam(model.decom_net.parameters(), lr=learning_rate),
        'enhance': optim.Adam(model.enhance_net.parameters(), lr=learning_rate),
        # 'finetune': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8, weight_decay=0.0001)
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
        #validate(model, device, vali_loader, stage) #Commented for final training
        wandb.log({'Epoch': epoch})
        wandb.log({'Learning_rate': current_lr})
        schedulers[stage].step()
        
        if epoch % 10 == 0: # Changed from 20 to 10 since finetuning only gets 50 epochs.
            if stage == 'decom':
                torch.save(model.decom_net.state_dict(), f'models/DecomNet/Job_{num_test}/DecomNet_trained_{num_test}_{epoch}.pt')
            elif stage == 'enhance':
                torch.save(model.enhance_net.state_dict(), f'models/EnhanceNet/Job_{num_test}/EnhanceNet_trained_{num_test}_{epoch}.pt')
            elif stage == 'finetune':
                torch.save(model.state_dict(), f'models/FineTuning/Job_{num_test}/FineTuning_{num_test}_{epoch}.pt')

   
if __name__ == '__main__':
    main()

