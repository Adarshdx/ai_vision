import torch
from pathlib import Path

class Config:
    # Paths
    BASE_PATH = Path(__file__).parent
    DATA_PATH = BASE_PATH / 'data'
    MODEL_PATH = BASE_PATH / 'models'
    OUTPUT_PATH = BASE_PATH / 'outputs'
    
    # Create directories
    DATA_PATH.mkdir(exist_ok=True)
    MODEL_PATH.mkdir(exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Model
    NUM_CLASSES = 10
    PRETRAINED = True
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Data augmentation
    IMG_SIZE = 224  # ResNet input size
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CIFAR-10 classes
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Random seed for reproducibility
    SEED = 42