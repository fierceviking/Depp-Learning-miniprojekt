import os
from sklearn.model_selection import train_test_split
import shutil

def main():
    # Define the path to the images
    train_high_names = os.listdir('train_data/high')
    train_low_names = os.listdir('train_data/low')

    # Sort the names
    train_high_names.sort()
    train_low_names.sort()

    validation_high_folder = os.path.join(os.path.dirname(__file__), 'vali_data/high')
    validation_low_folder = os.path.join(os.path.dirname(__file__), 'vali_data/low')

    os.makedirs(validation_high_folder, exist_ok=True)
    os.makedirs(validation_low_folder, exist_ok=True)


    # Define split
    train_high, val_high, train_low, val_low = train_test_split(
        train_high_names, train_low_names, test_size=.15, random_state=42
    )

    # Move validation images to new folder
    for img in val_high:
        shutil.move(os.path.join('train_data/high', img), os.path.join(validation_high_folder, img))


    for img in val_low:
        shutil.move(os.path.join('train_data/low', img), os.path.join(validation_low_folder, img))


if __name__ == '__main__':
    main()