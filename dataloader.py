import torch
import os
from PIL import Image
import numpy as np
from visuvalisation import show_images
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

def create_transforms(augment=True):
    """
    Create a set of transforms for image preprocessing
    
    Args:
        augment (bool): Whether to apply augmentations
    """
    transforms = [
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
    ] if augment else [
        T.Resize((128, 128)),
        T.ToTensor(),
    ]
    
    return T.Compose(transforms)

class BrainMRIDataset(Dataset):
    def __init__(self, folder, label, transform=None):
        self.folder = folder
        self.label = label
        self.transform = transform
        self.image_files = os.listdir(folder)
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(1 if self.label == "tumor" else 0)
        }

# Paths (adjust to your lab PC setup)
base_path = "Brain MRI Images for Brain Tumor Detection"
transform = create_transforms(augment=True)

healthy_dataset = BrainMRIDataset(
    os.path.join(base_path, "no"), 
    "healthy",
    transform=transform
)
tumor_dataset = BrainMRIDataset(
    os.path.join(base_path, "yes"), 
    "tumor", 
    transform=transform
)

healthy_loader = DataLoader(healthy_dataset, batch_size=32, shuffle=True)
tumor_loader = DataLoader(tumor_dataset, batch_size=32, shuffle=True)

#