import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from PIL import Image
import os

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        valid_samples = []
        for path, label in self.samples:
            try:
                img = Image.open(path).convert("RGB")
                valid_samples.append((path, label))
            except OSError:
                print(f"Skipping corrupted image: {path}")
        self.samples = valid_samples
        self.targets = [s[1] for s in valid_samples]

def get_data_loaders(transform, batch_size=32):
    dataset = SafeImageFolder(root="/content/data/Mushrooms", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

class ResNetTrainer:
    def __init__(self, num_classes=9, lr=1e-4, use_dropout=True, optimizer_type='adamw'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if use_dropout:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(self.model.fc.in_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, num_classes)
            )
        else:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        running_corrects = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        return running_loss / len(train_loader.dataset), running_corrects.float() / len(train_loader.dataset)
    
    def validate(self, val_loader):
        self.model.eval()
        running_corrects = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        return running_loss / len(val_loader.dataset), running_corrects.float() / len(val_loader.dataset)
    
    def train(self, train_loader, val_loader, epochs=5):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()
        return val_acc.item()  # Return final validation accuracy
    
def run_ablation_study():
    experiments = [
        {"name": "Baseline (Aug + Dropout + AdamW)", 
         "transform": transforms.Compose([
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.ColorJitter(0.1,0.1,0.1,0.1),
             transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
         ]),
         "dropout": True,
         "optimizer": "adamw"
        },
        {"name": "No Augmentation", 
         "transform": transforms.Compose([
             transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
         ]),
         "dropout": True,
         "optimizer": "adamw"
        },
        {"name": "No Dropout", 
         "transform": transforms.Compose([
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.ColorJitter(0.1,0.1,0.1,0.1),
             transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
         ]),
         "dropout": False,
         "optimizer": "adamw"
        },
        {"name": "SGD Optimizer", 
         "transform": transforms.Compose([
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.ColorJitter(0.1,0.1,0.1,0.1),
             transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
         ]),
         "dropout": True,
         "optimizer": "sgd"
        }
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\nRunning Experiment: {exp['name']}")
        train_loader, val_loader = get_data_loaders(exp["transform"])
        trainer = ResNetTrainer(num_classes=9, lr=1e-4, use_dropout=exp["dropout"], optimizer_type=exp["optimizer"])
        val_acc = trainer.train(train_loader, val_loader, epochs=5)
        print(f"Validation Accuracy for {exp['name']}: {val_acc:.4f}")
        results.append((exp['name'], val_acc))
    
    # Save summary
    with open("ablation_summary.txt", "w") as f:
        f.write("Ablation Study Summary:\n")
        f.write("======================\n")
        for name, acc in results:
            f.write(f"{name}: {acc:.4f}\n")
    print("\nAblation summary saved to ablation_summary.txt")

if __name__ == "__main__":
    run_ablation_study()
