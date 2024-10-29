import os
import random
import shutil

def train_val_split(org_folder_path, val_split):
    # Creates a list of the pictures in the original traning set
    # This is possible because the names in the "high" and "low" folders are the same per pair
    org_pic_path = os.path.join(org_folder_path, 'high')
    pic_list = os.listdir(org_pic_path)

    # Shuffles the order of the pictures and make a split
    random.seed(42)
    random.shuffle(pic_list)
    split_point = int((1 - val_split) * len(pic_list))

    # Creates two new lists, for training and validation respectively
    train_pics = pic_list[:split_point]
    val_pics = pic_list[split_point:]

    return train_pics, val_pics


def add_split_to_folder(org_folder_path, split_pics, split_path, high_low):
    # Makes the folder for "split_path" with a high and a low folder within
    os.makedirs(split_path, exist_ok=True)
    for type in high_low:
        file_path = os.path.join(split_path, type)
        os.makedirs(file_path, exist_ok=True)

        # Adds images from high/low path to the new "split_path"
        for img in split_pics:
            shutil.copy(os.path.join(org_folder_path, type, img), os.path.join(split_path, type))

def main():
    # The different variables
    org_folder_path = r'data/org_training_data'
    train_split_path = r'data/training'
    val_split_path = r'data/val'
    high_low = ["high", "low"]
    val_split = 0.15

    # A split is made to create a new training and validation set
    train_pics, val_pics = train_val_split(org_folder_path, val_split)

    # The new folders are created and the images are added
    add_split_to_folder(org_folder_path, train_pics, train_split_path, high_low)
    add_split_to_folder(org_folder_path, val_pics, val_split_path, high_low)


if __name__ == '__main__':
    main()
