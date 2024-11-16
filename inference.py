import torch
from model import RetinexNet
import cv2
import numpy as np

def load_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # Resize and normalize the image
    image = cv2.resize(image, (256, 256))  # Resize to the required size
    image = image.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Change to CHW format and add batch dimension

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    path = "snapshots/FineTuning/Job_8/FineTuning_8_50.pt"
    model = RetinexNet().to(device)

    # Load the model state_dict with map_location
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # Load image
    image_path = 'data/test_data/high/22.png'
    image = load_image(image_path).to(device)

    # Run inference
    with torch.no_grad():
        enhanced_image, reflectance, illumination, enhanced_illumination = model(image)

    # Save and display image using OpenCV
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0)  # Change dimensions to HxWxC
    enhanced_image = np.clip(enhanced_image.cpu().numpy(), 0, 1)
    enhanced_image = (enhanced_image * 255).astype('uint8')  # Convert to uint8 format

    # Save the enhanced image
    # cv2.imwrite("22_LLIE.png", cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

    # Original image: Save original image
    image = image.squeeze(0).permute(1,2,0)
    image = np.clip(image.cpu().numpy(), 0, 1)
    image = (image * 255).astype('uint8')
    # cv2.imwrite("22_original_high.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Display the enhanced image
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
