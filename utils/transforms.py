# utils/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentations(img_size=512):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(), ToTensorV2()
    ])

def get_val_augmentations(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(), ToTensorV2()
    ])
