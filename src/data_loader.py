"""
Dataset Loader
Supports loading and preprocessing of CIFAR-10, CIFAR-100, ImageNet datasets
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Unified dataset loader"""
    
    def __init__(self, dataset_name: str = Config.DEFAULT_DATASET):
        """
        Initialize dataset loader
        
        Args:
            dataset_name: Dataset name ('cifar10', 'cifar100', 'imagenet')
        """
        if dataset_name not in Config.AVAILABLE_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.dataset_name = dataset_name
        self.config = Config.DATASET_PARAMS[dataset_name]
        self.data_root = os.path.join(Config.DATA_ROOT, dataset_name)
        
        # Create dataset directory
        os.makedirs(self.data_root, exist_ok=True)
        
        logger.info(f"Initializing dataset loader: {dataset_name}")
        
    def get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """
        Get data preprocessing transforms
        
        Args:
            is_training: Whether in training mode
            
        Returns:
            transforms.Compose: Preprocessing transforms
        """
        if self.dataset_name in ['cifar10', 'cifar100']:
            if is_training:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
        
        elif self.dataset_name == 'imagenet':
            if is_training:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        
        return transform
    
    def load_dataset(self, train: bool = True, download: bool = True) -> Dataset:
        """
        Load dataset
        
        Args:
            train: Whether to load training set
            download: Whether to automatically download dataset
            
        Returns:
            Dataset: Dataset object
        """
        transform = self.get_transforms(is_training=train)
        
        if self.dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root=self.data_root,
                train=train,
                transform=transform,
                download=download
            )
        
        elif self.dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(
                root=self.data_root,
                train=train,
                transform=transform,
                download=download
            )
        
        elif self.dataset_name == 'imagenet':
            # ImageNet requires manual download, check if data exists
            split = 'train' if train else 'val'
            imagenet_path = os.path.join(self.data_root, split)
            
            if not os.path.exists(imagenet_path):
                raise FileNotFoundError(
                    f"ImageNet dataset not found: {imagenet_path}\n"
                    f"Please follow these steps to manually download:\n"
                    f"1. Visit http://www.image-net.org/download\n"
                    f"2. Register account and apply for download permission\n"
                    f"3. Download ILSVRC2012 dataset\n"
                    f"4. Extract to {self.data_root}/ folder"
                )
            
            dataset = torchvision.datasets.ImageFolder(
                root=imagenet_path,
                transform=transform
            )
        
        # If training set size limit is configured, create subset
        if train and self.config['train_size'] is not None:
            indices = torch.randperm(len(dataset))[:self.config['train_size']]
            dataset = Subset(dataset, indices)
            logger.info(f"Using training subset, size: {len(dataset)}")
        
        return dataset
    
    def get_data_loaders(self, 
                        batch_size: Optional[int] = None,
                        num_workers: Optional[int] = None,
                        pin_memory: Optional[bool] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and test data loaders
        
        Args:
            batch_size: Batch size, None uses config default
            num_workers: Number of worker processes, None uses config default
            pin_memory: Whether to pin memory, None uses config default
            
        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
        """
        # Use default config or user specified config
        batch_size = batch_size or self.config['batch_size']
        num_workers = num_workers or Config.NUM_WORKERS
        pin_memory = pin_memory if pin_memory is not None else Config.PIN_MEMORY
        
        # Load datasets
        train_dataset = self.load_dataset(train=True, download=True)
        test_dataset = self.load_dataset(train=False, download=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        logger.info(f"  Batch size: {batch_size}")
        
        return train_loader, test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information
        
        Returns:
            Dict[str, Any]: Dataset information dictionary
        """
        return {
            'name': self.dataset_name,
            'num_classes': self.config['num_classes'],
            'image_size': self.config['image_size'],
            'channels': self.config['channels'],
            'batch_size': self.config['batch_size'],
            'train_size': self.config['train_size']
        }


class UnlearnableDataset(Dataset):
    """Unlearnable examples dataset"""
    
    def __init__(self, clean_images: torch.Tensor, 
                 unlearnable_images: torch.Tensor,
                 targets: torch.Tensor,
                 perturbations: torch.Tensor):
        """
        Initialize unlearnable dataset
        
        Args:
            clean_images: Clean images
            unlearnable_images: Unlearnable images
            targets: Labels
            perturbations: Perturbations
        """
        self.clean_images = clean_images
        self.unlearnable_images = unlearnable_images
        self.targets = targets
        self.perturbations = perturbations
        
        assert len(clean_images) == len(unlearnable_images) == len(targets)
        
    def __len__(self) -> int:
        return len(self.clean_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.clean_images[idx],
            self.unlearnable_images[idx], 
            self.targets[idx],
            self.perturbations[idx]
        )


def create_data_loader(dataset_name: str) -> DatasetLoader:
    """
    Convenience function to create data loader
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        DatasetLoader: Data loader instance
    """
    return DatasetLoader(dataset_name)


def get_available_datasets() -> Dict[str, str]:
    """
    Get available dataset list and descriptions
    
    Returns:
        Dict[str, str]: Mapping from dataset name to description
    """
    return {
        name: info['description'] 
        for name, info in Config.DATASET_URLS.items()
    }


def check_dataset_availability(dataset_name: str) -> bool:
    """
    Check if dataset is available
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        bool: Whether dataset is available
    """
    if dataset_name not in Config.AVAILABLE_DATASETS:
        return False
    
    data_root = os.path.join(Config.DATA_ROOT, dataset_name)
    
    if dataset_name in ['cifar10', 'cifar100']:
        # CIFAR datasets will auto-download, just check directory
        return True
    
    elif dataset_name == 'imagenet':
        # ImageNet requires manual download, check if exists
        train_path = os.path.join(data_root, 'train')
        val_path = os.path.join(data_root, 'val')
        return os.path.exists(train_path) and os.path.exists(val_path)
    
    return False


def print_dataset_info():
    """Print all dataset information"""
    print("\n" + "=" * 60)
    print("Available Dataset Information")
    print("=" * 60)
    
    for dataset_name in Config.AVAILABLE_DATASETS:
        config = Config.DATASET_PARAMS[dataset_name]
        url_info = Config.DATASET_URLS[dataset_name]
        available = check_dataset_availability(dataset_name)
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Description: {url_info['description']}")
        print(f"  Classes: {config['num_classes']}")
        print(f"  Image size: {config['image_size']}")
        print(f"  Channels: {config['channels']}")
        print(f"  Default batch size: {config['batch_size']}")
        print(f"  Available: {'Yes' if available else 'No'}")
        
        if not url_info['auto_download']:
            print("  Download steps:")
            for step in url_info.get('manual_instructions', []):
                print(f"    {step}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Print dataset information
    print_dataset_info()
    
    # Test data loader
    print("\nTesting data loader...")
    
    for dataset_name in ['cifar10']:  # Only test CIFAR-10 as others may require manual download
        if check_dataset_availability(dataset_name):
            print(f"\nTesting {dataset_name}...")
            
            try:
                loader = create_data_loader(dataset_name)
                train_loader, test_loader = loader.get_data_loaders()
                
                # Get one batch of data
                for images, targets in train_loader:
                    print(f"  Batch shape: {images.shape}")
                    print(f"  Label shape: {targets.shape}")
                    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
                    break
                    
                print(f"  {dataset_name} test successful!")
                
            except Exception as e:
                print(f"  {dataset_name} test failed: {e}")
        else:
            print(f"\n{dataset_name} unavailable, skipping test")