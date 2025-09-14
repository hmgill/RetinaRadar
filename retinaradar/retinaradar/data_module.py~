import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

#
from handler_dataset import HandlerDataset




class RetinaRadarPLDataModule(pl.LightningDataModule):

    def __init__(
            self,
            retina_radar_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=32,
            seed=42
    ):
        super().__init__()
        self.retina_radar_dataset = retina_radar_dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.seed = seed

        # Define the standard imagenet mean and std
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        # Define Albumentations training transforms
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        
        # Define Albumentations validation/test transforms
        self.val_test_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        
    def setup(self, stage: str):
        # Create instance of dataset
        handler_dataset = HandlerDataset(self.retina_radar_dataset, transform=None)
        
        # Calculate split sizes
        total_size = len(handler_dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size

        # split the dataset 
        generator = torch.Generator().manual_seed(self.seed)
        self.train_split, self.val_split, self.test_split = random_split(
            handler_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        
    def train_dataloader(self):
        # set the appropriate transform 
        self.train_split.dataset.transform = self.train_transform
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True, num_workers=4)

    
    def val_dataloader(self):
        # set the appropriate transform        
        self.val_split.dataset.transform = self.val_test_transform
        return DataLoader(self.val_split, batch_size=self.batch_size, num_workers=4)

    
    def test_dataloader(self):
        # set the appropriate transform        
        self.test_split.dataset.transform = self.val_test_transform
        return DataLoader(self.test_split, batch_size=self.batch_size, num_workers=4)
