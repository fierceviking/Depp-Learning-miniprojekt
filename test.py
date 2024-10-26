import torch
from model import RetinexNet
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    # Transformation
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the same size
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std deviation
    ])

    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    return image


def main():
    if torch.cuda_is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    path = "models/job_1/RetinexNet_epoch100.pt"
    model = RetinexNet().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    # Load image
    image_path = 'test_data/low/1.png'
    image = load_image(image_path).to(device)

    # Run inference
    with torch.no_grad():
        enhanced_image, reflectance, illumination, enhanced_illumination = RetinexNet(image)

    # Save and Display image
    enhanced_image = enhanced_image.squeeze(0).cpu()
    enhanced_image = transforms.ToPILImage()(enhanced_image)

    enhanced_image.save("enhanced_image1.png")
    enhanced_image.show()
