import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class BreastMultiPhaseDataset(Dataset):
    """
    Dataset class for multi-phase DCE-MRI breast tumor segmentation.
    Includes synchronized spatial data augmentations for both images and masks.
    """
    def __init__(self, base_path, mode='train', augment=True):
        super().__init__()
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        folder_mode = 'train' if mode == 'train' else ('val' if mode == 'val' else 'test')
        root_dir = os.path.join(base_path, "Dataset_Final_Multi_Noisy", folder_mode)

        if not os.path.exists(root_dir):
            root_dir = os.path.join(base_path, folder_mode)

        self.img_paths = []
        self.mask_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    if 'images' in root.lower():
                        self.img_paths.append(full_path)
                    elif 'labels' in root.lower() or 'masks' in root.lower():
                        self.mask_paths.append(full_path)

        self.img_paths.sort()
        self.mask_paths.sort()

    def _apply_synchronized_transforms(self, image, mask):
        # Resize images to standard model input dimensions
        image = TF.resize(image, [256, 256])
        mask = TF.resize(mask, [256, 256], interpolation=transforms.InterpolationMode.NEAREST)

       
        if self.augment:
            
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

           
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

           
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

       
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

      
        if image.shape[0] == 3:
            image = torch.cat([image, image, image], dim=0)


        mask = (mask > 0.5).float()

        return image, mask

    def __len__(self):
        return min(len(self.img_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        return self._apply_synchronized_transforms(image, mask)
