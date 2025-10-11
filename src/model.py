import torch.nn as nn
from torchvision import models

class FineTuneResNet18(nn.Module):
    def __init__(self, num_classes):
        super(FineTuneResNet18, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # keep only feature extractor
        
        self.extra_layers = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)            
        x = x.view(x.size(0), 512, 1, 1) 
        x = self.extra_layers(x)        
        x = x.view(x.size(0), -1)       
        x = self.classifier(x)
        return x