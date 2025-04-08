import os
import random
import cv2
import matplotlib.pyplot as plt
import torch
#display images
# def show_images(folder, title, num_images=4):
#     if not os.path.exists(folder):
#         raise FileNotFoundError(f"The folder '{folder}' does not exist.")
#     all_images = os.listdir(folder)
#     if len(all_images) < num_images:
#         raise ValueError(f"Not enough images in the folder '{folder}'. Required: {num_images}, Found: {len(all_images)}")
#     images = random.sample(all_images, num_images)
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
#     for i, img_name in enumerate(images):
#         img_path = os.path.join(folder, img_name)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
#         axes[i].imshow(img)
#         axes[i].axis("off")
#         axes[i].set_title(title)
    
#     plt.show()

def show_images(data, title, num_images=4):
    """
    Display images from either a folder path or a tensor of images
    
    Args:
        data: str (folder path) or torch.Tensor (B, C, H, W)
        title: str, title for the subplot
        num_images: int, number of images to display
    """
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if isinstance(data, str):  # If data is a folder path
        if not os.path.exists(data):
            raise FileNotFoundError(f"The folder '{data}' does not exist.")
        all_images = os.listdir(data)
        if len(all_images) < num_images:
            raise ValueError(f"Not enough images in the folder. Required: {num_images}, Found: {len(all_images)}")
        
        images = random.sample(all_images, num_images)
        for i, img_name in enumerate(images):
            img_path = os.path.join(data, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(title)
            
    elif isinstance(data, torch.Tensor):  # If data is a tensor
        if num_images > len(data):
            raise ValueError(f"Not enough images in tensor. Required: {num_images}, Found: {len(data)}")
        
        indices = random.sample(range(len(data)), num_images)
        for i, idx in enumerate(indices):
            img = data[idx]
            if img.dim() == 3:  # If image has channel dimension
                img = img.squeeze(0)  # Remove channel dim if present
            axes[i].imshow(img.cpu().numpy(), cmap='gray')
            axes[i].axis("off")
            axes[i].set_title(title)
    
    else:
        raise TypeError("Data must be either a folder path (str) or a tensor")
    
    plt.show()