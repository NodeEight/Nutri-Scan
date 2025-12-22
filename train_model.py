import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuration
BATCH_SIZE = 16 # Reduced batch size slightly as we have multiple runs
LEARNING_RATE = 0.001
EPOCHS = 100
IMG_SIZE = 224
BODY_PARTS = ['back', 'body', 'finger', 'head', 'leg', 'muac', 'side']
MODELS_DIR = "models"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create models directory if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

class NutriDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label

def load_data_for_part(body_part):
    """
    Crawls the 'malnourished/{body_part}' and 'normal/{body_part}' directories.
    """
    
    # Define classes
    # 0: Malnourished
    # 1: Normal
    
    malnourished_dir = Path("malnourished") / body_part
    normal_dir = Path("normal") / body_part
    
    # Collect all image paths (recursively in case of subfolders)
    malnourished_images = list(malnourished_dir.rglob("*.jpg")) + \
                          list(malnourished_dir.rglob("*.jpeg")) + \
                          list(malnourished_dir.rglob("*.png"))
                          
    normal_images = list(normal_dir.rglob("*.jpg")) + \
                    list(normal_dir.rglob("*.jpeg")) + \
                    list(normal_dir.rglob("*.png"))

    print(f"[{body_part.upper()}] Found {len(malnourished_images)} malnourished images.")
    print(f"[{body_part.upper()}] Found {len(normal_images)} normal images.")
    
    all_paths = malnourished_images + normal_images
    # Labels: 0 for malnourished, 1 for normal
    all_labels = [0] * len(malnourished_images) + [1] * len(normal_images)
    
    return all_paths, all_labels

def train_model_for_part(body_part):
    print(f"\n{'='*20} Training Model for: {body_part.upper()} {'='*20}")
    
    # 1. Prepare Data
    paths, labels = load_data_for_part(body_part)
    
    if not paths:
        print(f"Skipping {body_part}: No images found.")
        return

    # Check for class balance/existence
    if len(set(labels)) < 2:
        print(f"Skipping {body_part}: Needs both Malnourished and Normal classes.")
        return

    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NutriDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = NutriDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model
    # Use modern weights API
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_correct / val_total
        
        print(f"[{body_part}] Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save Best Model Only
        if val_epoch_acc >= best_acc:
            best_acc = val_epoch_acc
            save_path = os.path.join(MODELS_DIR, f"nutriscan_model_{body_part}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saved new best model for {body_part} (Acc: {best_acc:.4f})")

def main():
    print("Starting multi-model training...")
    for part in BODY_PARTS:
        train_model_for_part(part)
    print("\nAll training completed.")

if __name__ == "__main__":
    main()
