import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import Config

def get_data_augmentation():
    """Data augmentation for training"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD)
    ])

def get_test_transform():
    """Transform for validation/test"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD)
    ])

def load_cifar10_data():
    """Load and prepare CIFAR-10 dataset"""
    # Download and transform
    train_dataset = datasets.CIFAR10(
        root=Config.DATA_PATH,
        train=True,
        download=True,
        transform=get_data_augmentation()
    )
    
    test_dataset = datasets.CIFAR10(
        root=Config.DATA_PATH,
        train=False,
        download=True,
        transform=get_test_transform()
    )
    
    # Split training data into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader