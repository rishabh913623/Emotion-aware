"""
Advanced Training Script for Facial Emotion Recognition
Week 2: PyTorch implementation with improved architecture and data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
import warnings
warnings.filterwarnings('ignore')

from advanced_model import AdvancedEmotionCNN

class FER2013Dataset(Dataset):
    """Custom dataset for FER2013 data"""
    
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get pixel data and emotion label
        pixels = self.data.iloc[idx]['pixels']
        emotion = self.data.iloc[idx]['emotion']
        
        # Convert pixel string to image
        pixel_array = np.array([int(x) for x in pixels.split()]).reshape(48, 48)
        image = Image.fromarray(pixel_array.astype(np.uint8), mode='L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, emotion

class DAiSEEDataset(Dataset):
    """Custom dataset for DAiSEE data (folder structure)"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Emotion mapping for DAiSEE (adjust based on actual dataset structure)
        self.emotion_map = {
            'Boredom': 6,      # neutral
            'Engagement': 3,   # happy
            'Confusion': 2,    # fear
            'Frustration': 0   # angry
        }
        
        # Load all samples
        for emotion_name, emotion_idx in self.emotion_map.items():
            emotion_dir = os.path.join(root_dir, emotion_name)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_name)
                        self.samples.append((img_path, emotion_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, emotion = self.samples[idx]
        
        # Load and convert image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Return a blank image if loading fails
            image = np.zeros((48, 48), dtype=np.uint8)
        
        image = cv2.resize(image, (48, 48))
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, emotion

class EmotionTrainer:
    """Advanced training class for emotion recognition"""
    
    def __init__(self, model, device, num_classes=7):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{val_loss/(pbar.n+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        
        return val_loss, val_acc, all_preds, all_targets
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def get_data_transforms(augmentation=True):
    """Get data transforms for training and validation"""
    
    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.RandomCrop((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform

def create_data_loaders(dataset_type, data_path, batch_size, num_workers=4, augmentation=True):
    """Create data loaders for training and validation"""
    
    train_transform, val_transform = get_data_transforms(augmentation)
    
    if dataset_type == 'fer2013':
        # FER2013 dataset
        dataset = FER2013Dataset(data_path, transform=train_transform)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Apply validation transform to validation set
        val_dataset.dataset.transform = val_transform
        
    elif dataset_type == 'daisee':
        # DAiSEE dataset
        train_dataset = DAiSEEDataset(
            os.path.join(data_path, 'train'), transform=train_transform
        )
        val_dataset = DAiSEEDataset(
            os.path.join(data_path, 'val'), transform=val_transform
        )
        
    elif dataset_type == 'folder':
        # Generic folder structure
        train_dataset = ImageFolder(
            os.path.join(data_path, 'train'), transform=train_transform
        )
        val_dataset = ImageFolder(
            os.path.join(data_path, 'val'), transform=val_transform
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def train_model(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = AdvancedEmotionCNN(num_classes=args.num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.dataset_type, args.data_path, args.batch_size, 
        args.num_workers, args.augmentation
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    trainer = EmotionTrainer(model, device, args.num_classes)
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\\nStarting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training
        train_loss, train_acc = trainer.train_epoch(
            train_loader, optimizer, criterion, epoch
        )
        
        # Validation
        val_loss, val_acc, val_preds, val_targets = trainer.validate(
            val_loader, criterion, epoch
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"\\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_accuracies': trainer.train_accuracies,
                'val_accuracies': trainer.val_accuracies
            }
            
            torch.save(checkpoint, args.save_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
            
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\\nEarly stopping triggered after {args.patience} epochs without improvement")
            break
    
    total_time = time.time() - start_time
    print(f"\\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    trainer.plot_training_history('training_history_advanced.png')
    
    # Generate classification report
    print("\\nValidation Classification Report:")
    print(classification_report(
        val_targets, val_preds, 
        target_names=trainer.emotion_labels[:args.num_classes]
    ))
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(
        val_targets, val_preds, 'confusion_matrix_advanced.png'
    )
    
    # Save training configuration
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['total_training_time'] = total_time
    config['device_used'] = str(device)
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"\\n🎉 Training completed! Model saved to {args.save_path}")
    return model, best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Advanced Facial Emotion Recognition Training')
    
    # Dataset arguments
    parser.add_argument('--dataset_type', type=str, choices=['fer2013', 'daisee', 'folder'], 
                       default='fer2013', help='Type of dataset')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to dataset')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of emotion classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=15,
                       help='Patience for early stopping')
    
    # Technical arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--save_path', type=str, default='best_emotion_model.pth',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    print("🚀 Advanced Facial Emotion Recognition Training")
    print("="*50)
    print(f"Dataset: {args.dataset_type}")
    print(f"Data path: {args.data_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data augmentation: {args.augmentation}")
    print("="*50)
    
    # Start training
    model, best_acc = train_model(args)
    
    if best_acc >= 70.0:
        print(f"\\n🎯 Target accuracy achieved: {best_acc:.2f}% ≥ 70%")
    else:
        print(f"\\n⚠️  Target accuracy not reached: {best_acc:.2f}% < 70%")
        print("Consider:")
        print("- Training for more epochs")
        print("- Adjusting hyperparameters")
        print("- Using more data")
        print("- Trying different augmentation strategies")

if __name__ == "__main__":
    main()