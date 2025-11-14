import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

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

def load_mushroom_data(data_dir="/content/data/Mushrooms", batch_size=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SafeImageFolder(root=data_dir, transform=transform)

    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {dataset.classes}")

    return train_loader, val_loader, dataset.classes

class ResNetTrainer:
    def __init__(self, num_classes=10, lr=1e-4):
        # Device selection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace FC with Dropout + two layers
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, train_loader, val_loader, num_epochs=20):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():  # Mixed precision
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_corrects.float() / len(train_loader.dataset)

            self.model.eval()
            running_loss = 0.0
            running_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.cuda.amp.autocast():  # Mixed precision
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_corrects.float() / len(val_loader.dataset)

            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            self.scheduler.step()  # update learning rate

    def get_model(self):
        return self.model

if __name__ == "__main__":
    train_loader, val_loader, classes = load_mushroom_data(batch_size=32)
    num_classes = len(classes)

    trainer = ResNetTrainer(num_classes=num_classes, lr=1e-4)
    trainer.train(train_loader, val_loader, num_epochs=20)

    model = trainer.get_model()
    torch.save(model.state_dict(), "resnet18_mushrooms_finetuned.pth")
    print("Model saved to resnet18_mushrooms_finetuned.pth")
