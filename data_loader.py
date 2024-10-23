import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        self.transform = transform

    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, index):
        low_path = os.path.join(self.low_dir, self.low_images[index])
        high_path = os.path.join(self.high_dir, self.high_images[index])

        low_image = Image.open(low_path)
        high_image = Image.open(high_path)

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        return low_image, high_image
    

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the same size
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std deviation
])

def main():
    train_low_dir = 'train_data/low'
    train_high_dir = 'train_data/high'

    train_dataset = LOLDataset(train_low_dir, train_high_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)


if __name__ == '__main__':
    main()