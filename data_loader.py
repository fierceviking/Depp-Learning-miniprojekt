import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, patch_size=(96,96)):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        self.transform = transform
        self.patch_size = patch_size

        assert len(self.low_images) == len(self.high_images), "Number of low and high images must match."

        for low_image, high_image in zip(self.low_images, self.high_images):
            assert low_image == high_image, f"Filename mismatch: {low_image} and {high_image} do not match."

    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, index):
        low_path = os.path.join(self.low_dir, self.low_images[index])
        high_path = os.path.join(self.high_dir, self.high_images[index])

        low_image = Image.open(low_path)
        high_image = Image.open(high_path)


        # Randomly select a patch from the iamage
        if low_image.size[0] > self.patch_size[0] and low_image.size[1] > self.patch_size[1]:
            x = random.randint(0, low_image.size[0] - self.patch_size[0])
            y = random.randint(0, low_image.size[1] - self.patch_size[1])
            low_image = low_image.crop((x, y, x + self.patch_size[0], y + self.patch_size[1]))
            high_image = high_image.crop((x, y, x + self.patch_size[0], y + self.patch_size[1]))

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
    # print(train_dataset.__len__)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(train_loader))

if __name__ == '__main__':
    main()