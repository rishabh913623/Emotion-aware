"""
Audio Emotion Recognition Training Script
Week 3: Train classifier using RAVDESS/IEMOCAP datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import librosa
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
import pickle
from pathlib import Path

from audio_emotion_model import AudioEmotionCNN, AudioFeatureExtractor

class AudioEmotionDataset(Dataset):
    """Dataset for audio emotion recognition"""
    
    def __init__(self, data_path, dataset_type='ravdess', sample_rate=16000, max_length=5.0):
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type.lower()
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        
        # Emotion mappings for different datasets
        if self.dataset_type == 'ravdess':
            self.emotion_map = {
                1: 'neutral',  # neutral
                2: 'neutral',  # calm
                3: 'happy',    # happy
                4: 'sad',      # sad
                5: 'angry',    # angry
                6: 'fear',     # fearful
                7: 'disgust',  # disgust
                8: 'surprise'  # surprised (map to happy)
            }
        elif self.dataset_type == 'iemocap':
            self.emotion_map = {
                'neu': 'neutral',
                'hap': 'happy',
                'sad': 'sad',
                'ang': 'angry',
                'fea': 'fear',
                'dis': 'disgust',
                'exc': 'happy',  # excitement -> happy
                'fru': 'angry'   # frustration -> angry
            }
        
        # Target emotions for our model
        self.target_emotions = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.target_emotions)
        
        # Load dataset
        self.audio_files, self.labels = self._load_dataset()
        print(f"Loaded {len(self.audio_files)} audio files")
        
    def _load_dataset(self):
        """Load audio files and labels based on dataset type"""
        audio_files = []
        labels = []
        
        if self.dataset_type == 'ravdess':
            return self._load_ravdess()
        elif self.dataset_type == 'iemocap':
            return self._load_iemocap()
        else:
            return self._load_generic()
    
    def _load_ravdess(self):
        """Load RAVDESS dataset"""
        audio_files = []
        labels = []
        
        for actor_dir in self.data_path.glob('Actor_*'):
            if actor_dir.is_dir():
                for audio_file in actor_dir.glob('*.wav'):
                    # RAVDESS filename format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
                    filename_parts = audio_file.stem.split('-')
                    if len(filename_parts) >= 3:
                        emotion_code = int(filename_parts[2])
                        if emotion_code in self.emotion_map:
                            emotion = self.emotion_map[emotion_code]
                            if emotion in self.target_emotions:
                                audio_files.append(str(audio_file))
                                labels.append(emotion)
        
        return audio_files, labels
    
    def _load_iemocap(self):
        """Load IEMOCAP dataset"""
        audio_files = []
        labels = []
        
        # Look for session directories
        for session_dir in self.data_path.glob('Session*'):
            if session_dir.is_dir():
                # Look for audio files
                audio_dir = session_dir / 'sentences' / 'wav'
                if audio_dir.exists():
                    for audio_file in audio_dir.glob('*.wav'):
                        # Try to extract emotion from filename or use annotation files
                        emotion = self._extract_iemocap_emotion(audio_file)
                        if emotion and emotion in self.target_emotions:
                            audio_files.append(str(audio_file))
                            labels.append(emotion)
        
        return audio_files, labels
    
    def _extract_iemocap_emotion(self, audio_file):
        """Extract emotion label for IEMOCAP file"""
        # This is a simplified version - in practice, you'd use the annotation files
        filename = audio_file.stem.lower()
        
        # Simple heuristic based on filename patterns
        if 'neu' in filename:
            return 'neutral'
        elif 'hap' in filename or 'exc' in filename:
            return 'happy'
        elif 'sad' in filename:
            return 'sad'
        elif 'ang' in filename or 'fru' in filename:
            return 'angry'
        elif 'fea' in filename:
            return 'fear'
        elif 'dis' in filename:
            return 'disgust'
        
        return None
    
    def _load_generic(self):
        """Load generic folder-based dataset"""
        audio_files = []
        labels = []
        
        for emotion_dir in self.data_path.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name.lower()
                if emotion in self.target_emotions:
                    for audio_file in emotion_dir.glob('*.wav'):
                        audio_files.append(str(audio_file))
                        labels.append(emotion)
        
        return audio_files, labels
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Trim or pad audio to max_length
            max_samples = int(self.max_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
            
            # Extract features
            features = self.feature_extractor.extract_all_features(audio)
            feature_vector = self._combine_features(features)
            
            # Encode label
            label_encoded = self.label_encoder.transform([label])[0]
            
            return torch.FloatTensor(feature_vector), torch.LongTensor([label_encoded])[0]
            
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return zero features for failed loading
            return torch.zeros(93), torch.LongTensor([0])[0]
    
    def _combine_features(self, features):
        """Combine all audio features into a single feature vector"""
        combined_features = []
        
        # MFCC features (52 dimensions)
        combined_features.extend(features['mfcc_mean'])
        combined_features.extend(features['mfcc_std'])
        combined_features.extend(features['mfcc_delta'])
        combined_features.extend(features['mfcc_delta2'])
        
        # Pitch features (3 dimensions)
        combined_features.extend([
            features['pitch_mean'],
            features['pitch_std'],
            features['voicing_probability']
        ])
        
        # Spectral features (8 dimensions)
        spectral_keys = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'zero_crossing_rate_mean', 'zero_crossing_rate_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std'
        ]
        combined_features.extend([features[key] for key in spectral_keys])
        
        # Chroma features (24 dimensions)
        combined_features.extend(features['chroma_mean'])
        combined_features.extend(features['chroma_std'])
        
        # Tempo features (2 dimensions)
        combined_features.extend([features['tempo'], features['beat_strength']])
        
        # Energy features (4 dimensions)
        combined_features.extend([
            features['energy_mean'],
            features['energy_std'],
            features['audio_length'],
            features['rms_energy']
        ])
        
        # Ensure we have exactly 93 features
        feature_array = np.array(combined_features[:93])
        if len(feature_array) < 93:
            feature_array = np.pad(feature_array, (0, 93 - len(feature_array)), 'constant')
        
        # Handle NaN and infinite values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_array

class AudioEmotionTrainer:
    """Trainer class for audio emotion recognition"""
    
    def __init__(self, model, device, num_classes=6):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust']
        
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
    
    def plot_training_history(self, save_path='audio_training_history.png'):
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
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='audio_confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Audio Emotion Recognition - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_data_loaders(dataset_path, dataset_type, batch_size, train_split=0.8, num_workers=4):
    """Create data loaders for training and validation"""
    
    # Create dataset
    dataset = AudioEmotionDataset(dataset_path, dataset_type=dataset_type)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, dataset.emotion_labels

def train_audio_model(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, emotion_labels = create_data_loaders(
        args.data_path, args.dataset_type, args.batch_size, 
        args.train_split, args.num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Emotion labels: {emotion_labels}")
    
    # Create model
    model = AudioEmotionCNN(input_dim=93, num_classes=len(emotion_labels))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    trainer = AudioEmotionTrainer(model, device, len(emotion_labels))
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\\nStarting audio emotion training...")
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
        scheduler.step(val_acc)
        
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
                'emotion_labels': emotion_labels,
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
    trainer.plot_training_history('audio_training_history.png')
    
    # Generate classification report
    print("\\nValidation Classification Report:")
    print(classification_report(
        val_targets, val_preds, 
        target_names=emotion_labels
    ))
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(
        val_targets, val_preds, 'audio_confusion_matrix.png'
    )
    
    return model, best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Recognition Training')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to audio emotion dataset')
    parser.add_argument('--dataset_type', type=str, choices=['ravdess', 'iemocap', 'generic'],
                       default='ravdess', help='Type of dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    
    # Technical arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default='best_audio_emotion_model.pth',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    print("🎵 Audio Emotion Recognition Training")
    print("="*50)
    print(f"Dataset: {args.dataset_type}")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*50)
    
    # Start training
    model, best_acc = train_audio_model(args)
    
    print(f"\\n🎯 Audio emotion recognition training completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()