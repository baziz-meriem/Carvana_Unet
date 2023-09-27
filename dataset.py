import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self): # returns the number of samples in the dataset.
        return len(self.images)

    def __getitem__(self, index): # loads and returns a sample from the dataset at the given index

        img_path = os.path.join(self.image_dir, self.images[index]) 
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")) 
        image = np.array(Image.open(img_path)) #cnvert to  numpy cause augmentation lib expects it
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)#masks are grayscale
        
        mask[mask == 255.0] = 1.0 #converting the grayscale image to a binary mask where 1.0 represents the target class or region, and 0.0 represents everything else

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)#dict of augmented data
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask