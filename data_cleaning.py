from PIL import Image
import glob
import os
import numpy as np

def load_images(img_file):
    im = Image.open(img_file)
    resize = (128,128)
    img_resize = im.resize(resize)
    return img_resize

def save_images(image_list, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for _, image_path in enumerate(image_list):
        img = load_images(image_path)

        # Get original name
        img_name = os.path.basename(image_path)

        # Save the resized_image
        img.save(os.path.join(folder_path, img_name))



def main():
    # Define the path to the images
    train_high_names = glob.glob(os.path.join(os.path.dirname(__file__), 'train_data/high/*.png'))
    train_low_names = glob.glob(os.path.join(os.path.dirname(__file__), 'train_data/low/*.png'))

    # Sort the names
    train_high_names.sort()
    train_low_names.sort()

    # Create a list of names
    train_high = []
    train_low = []

    # Define the resized path
    resized_high_folder = os.path.join(os.path.dirname(__file__), 'train_data_resized/high')
    resized_low_folder = os.path.join(os.path.dirname(__file__), 'train_data_resized/low')

    # Save images
    save_images(train_high_names, resized_high_folder)
    save_images(train_low_names, resized_low_folder)





if __name__ == '__main__':
    main()