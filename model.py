import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class CIFAR10Classifier(nn.Module):
    """ResNet18 based classifier for CIFAR-10"""
    
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(CIFAR10Classifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first conv layer for CIFAR-10 (32x32 images vs 224x224)
        # ResNet expects 224x224, but we resize to 224x224, so keep original
        # But we can use a smaller kernel for better performance
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Modify final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for modified layers"""
        for m in self.backbone.conv1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classifier"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

def create_model():
    """Create and initialize the model"""
    model = CIFAR10Classifier(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {Config.DEVICE}")
    
    return model