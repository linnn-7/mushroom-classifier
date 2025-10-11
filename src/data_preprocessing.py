import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

Image.LOAD_TRUNCATED_IMAGES = True  

def safe_loader(path):
    """Load image, skip if truncated/corrupted."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except OSError:
        print(f"Warning: truncated/corrupted image replaced: {path}")
        # Return a blank RGB image as placeholder
        return Image.new("RGB", (224, 224), (0, 0, 0))

def load_data(batch_size=32):
    DATA_DIR = "../data/Mushrooms"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform, loader=safe_loader)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Total valid images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {dataset.classes}")

    return train_loader, val_loader, dataset.classes


if __name__ == "__main__":
    load_data()

