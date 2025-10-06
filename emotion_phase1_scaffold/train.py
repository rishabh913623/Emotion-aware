#!/usr/bin/env python3
"""
Emotion-Aware Virtual Classroom Assistant - Phase 1 Training Script
Trains a CNN model for facial emotion detection using FER2013 or custom datasets.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
import pickle

# Emotion labels for FER2013 dataset
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionCNN:
    """CNN model for emotion detection"""

    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self):
        """Build the CNN architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Fully connected layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_callbacks(self, model_save_path='best_emotion_model.h5'):
        """Get training callbacks"""
        callbacks_list = [
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        return callbacks_list

def load_fer2013_data(csv_path):
    """Load and preprocess FER2013 dataset from CSV file"""
    print(f"Loading FER2013 data from {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"Error: FER2013 CSV file not found at {csv_path}")
        print("Please download fer2013.csv from Kaggle and place it in the data directory.")
        return None, None, None, None

    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Extract pixels and emotions
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].tolist()

    # Convert pixel strings to numpy arrays
    X = []
    for pixel_string in tqdm(pixels, desc="Processing images"):
        pixel_values = [int(pixel) for pixel in pixel_string.split()]
        pixel_array = np.array(pixel_values).reshape(48, 48, 1)
        X.append(pixel_array)

    X = np.array(X, dtype='float32')
    y = np.array(emotions)

    # Normalize pixel values
    X = X / 255.0

    # Convert labels to categorical
    y_categorical = keras.utils.to_categorical(y, num_classes=7)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test

def load_folder_data(data_dir):
    """Load emotion data from folder structure (emotion_name/image_files)"""
    print(f"Loading data from folder structure: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return None, None, None, None

    X = []
    y = []
    emotion_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    emotion_to_label = {emotion: idx for idx, emotion in enumerate(sorted(emotion_folders))}

    print(f"Found emotion categories: {list(emotion_to_label.keys())}")

    for emotion_name, label in emotion_to_label.items():
        emotion_path = os.path.join(data_dir, emotion_name)
        image_files = [f for f in os.listdir(emotion_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Loading {len(image_files)} images for {emotion_name}...")

        for image_file in tqdm(image_files, desc=f"Processing {emotion_name}"):
            image_path = os.path.join(emotion_path, image_file)

            # Load and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                img = img.reshape(48, 48, 1)
                X.append(img)
                y.append(label)

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)

    # Convert labels to categorical
    y_categorical = keras.utils.to_categorical(y, num_classes=len(emotion_to_label))

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test

def create_data_generators(X_train, y_train, batch_size=32):
    """Create data generators for training with augmentation"""

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # No augmentation for validation
    val_datagen = ImageDataGenerator()

    # Split training data into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    train_generator = train_datagen.flow(
        X_train_split, y_train_split, batch_size=batch_size
    )

    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=batch_size
    )

    return train_generator, val_generator, X_val, y_val

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history plot saved to {save_path}")

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model on test set...")

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Calculate accuracy
    test_accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes,
                              target_names=EMOTION_LABELS))

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(EMOTION_LABELS))
    plt.xticks(tick_marks, EMOTION_LABELS, rotation=45)
    plt.yticks(tick_marks, EMOTION_LABELS)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    return test_accuracy

def train_emotion_model(data_source, data_path, epochs=50, batch_size=32,
                       learning_rate=0.001, model_save_path='emotion_model.h5'):
    """Main training function"""

    print("=" * 60)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)

    # Load data based on source type
    if data_source == 'fer2013':
        X_train, X_test, y_train, y_test = load_fer2013_data(data_path)
    elif data_source == 'folder':
        X_train, X_test, y_train, y_test = load_folder_data(data_path)
    else:
        raise ValueError("data_source must be 'fer2013' or 'folder'")

    if X_train is None:
        print("Failed to load data. Exiting...")
        return None

    # Create model
    print("\nBuilding CNN model...")
    emotion_cnn = EmotionCNN(input_shape=(48, 48, 1), num_classes=y_train.shape[1])
    model = emotion_cnn.build_model()
    emotion_cnn.compile_model(learning_rate=learning_rate)

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen, X_val, y_val = create_data_generators(
        X_train, y_train, batch_size=batch_size
    )

    # Get callbacks
    callbacks_list = emotion_cnn.get_callbacks(model_save_path)

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    print(f"\nTraining Configuration:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Validation steps: {validation_steps}")

    # Train the model
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot training history
    plot_training_history(history)

    # Load best model for evaluation
    print(f"\nLoading best model from {model_save_path}...")
    best_model = keras.models.load_model(model_save_path)

    # Evaluate on test set
    test_accuracy = evaluate_model(best_model, X_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best model saved to: {model_save_path}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("=" * 60)

    return best_model, history

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Train CNN for emotion detection')

    parser.add_argument('--data_source', type=str, choices=['fer2013', 'folder'],
                       default='fer2013', help='Data source type')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to FER2013 CSV file or folder containing emotion data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--model_save_path', type=str, default='emotion_model.h5',
                       help='Path to save the trained model')

    args = parser.parse_args()

    # Set up GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Using GPU for training.")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU for training.")

    # Train the model
    try:
        model, history = train_emotion_model(
            data_source=args.data_source,
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_save_path=args.model_save_path
        )

        if model is not None:
            print("\nTraining completed successfully!")
            print(f"Model saved to: {args.model_save_path}")
            print("\nTo use the model for inference, load it with:")
            print(f"model = tf.keras.models.load_model('{args.model_save_path}')")

    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()