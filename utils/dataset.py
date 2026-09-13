import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class BreastMultiPhaseDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        self.data_path = os.path.join(data_path, mode)
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(self.data_path, 'images', '*.png')))
        self.masks = sorted(glob.glob(os.path.join(self.data_path, 'masks', '*.png')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
