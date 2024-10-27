import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import os
from model import RetinexNet

def load_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # Resize and normalize the image
    image = cv2.resize(image, (256, 256))  # Resize to the required size
    image = image.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Change to CHW format and add batch dimension

def calculate_PSNR(enhanced_img, ground_truth):
    mse = F.mse_loss(enhanced_img, ground_truth).item()
    assert mse != 0, "Cannot divide by zero!"
    max_pixel_value = 1.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_SSIM(enhanced_img, ground_truth):
    ssim_metric = SSIM(data_range=1.0)
    ssim_value = ssim_metric(enhanced_img, ground_truth)
    return ssim_value.item()


def main():
    torch.manual_seed(42)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    weights = "models/FineTuning/Job_8/FineTuning_8_50.pt"
    model = RetinexNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # Test data image path
    gt_high_dir = 'data/test_data/high'
    gt_low_dir = 'data/test_data/low'

    psnr_values = []
    ssim_values = []

    for image_name in os.listdir(gt_high_dir):
        gt_high_path = os.path.join(gt_high_dir, image_name)
        gt_low_path = os.path.join(gt_low_dir, image_name)

        # print(gt_image_path)
        gt_high_image = load_image(gt_high_path).to(device)
        gt_low_image = load_image(gt_low_path).to(device)

        with torch.no_grad():
            enhanced_image, _, _, _ = model(gt_low_image)

        psnr = calculate_PSNR(enhanced_image, gt_high_image)
        ssim = calculate_SSIM(enhanced_image, gt_high_image)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    print(f'Mean PSNR: {round(mean_psnr, 2)}')
    print(f'Mean SSIM: {round(mean_ssim, 2)}')

    with open('results.txt_6', 'w') as f:
        f.write(f"Used weights: {weights}\n")
        f.write(f"Mean PSNR on test set: {round(mean_psnr,2)}\n")
        f.write(f"Mean SSIM on test set: {round(mean_ssim,2)}")


if __name__ == '__main__':
    main()