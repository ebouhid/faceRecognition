import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# TODO: fix errors when not using albumentations
# RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[8, 112, 112, 3] to have 3 channels, but got 112 channels instead

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir # post-processed folder
        self.transforms = transforms
        self.celeb_folders = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.celeb_folders)
    
    def __getitem__(self, idx):
        celebrity_folder = self.celeb_folders[idx]
        celebrity_folder_path = os.path.join(self.root_dir, celebrity_folder)
        celebrity_images = os.listdir(celebrity_folder_path)

        # Restriction: we need at least 2 images for each folder
        if len(celebrity_images) < 2:
            return self.__getitem__(random.randint(0, len(self)-1))
    
        # get anchor
        anchor_image_name = random.choice(celebrity_images)
        anchor_image_path = os.path.join(celebrity_folder_path, anchor_image_name)

        # get positive (making sure it has different name from anchor)
        positive_image_name = random.choice(celebrity_images)
        while positive_image_name == anchor_image_name:
            positive_image_name = random.choice(celebrity_images)
        positive_image_path = os.path.join(celebrity_folder_path, positive_image_name)

        # get negative (making sure it is from a different celebrity)
        negative_celebrity_folder = random.choice(self.celeb_folders)
        while negative_celebrity_folder == celebrity_folder:
            negative_celebrity_folder = random.choice(self.celeb_folders)
        negative_celebrity_folder_path = os.path.join(self.root_dir, negative_celebrity_folder)
        negative_images = os.listdir(negative_celebrity_folder_path)
        negative_image_name = random.choice(negative_images)
        negative_image_path = os.path.join(negative_celebrity_folder_path, negative_image_name)

        anchor_image = Image.open(anchor_image_path)
        positive_image = Image.open(positive_image_path)
        negative_image = Image.open(negative_image_path)

        if self.transforms:
            anchor_image = (self.transforms(image=np.array(anchor_image))['image'])
            positive_image = (self.transforms(image=np.array(positive_image))['image'])
            negative_image = (self.transforms(image=np.array(negative_image))['image'])

        anchor_image = np.asarray(anchor_image)
        positive_image = np.asarray(positive_image)
        negative_image = np.asarray(negative_image)

        # convert images to float32
        anchor_image = anchor_image.astype(np.float32) / 255.
        positive_image = positive_image.astype(np.float32) / 255.
        negative_image = negative_image.astype(np.float32) / 255.

        return anchor_image, positive_image, negative_image