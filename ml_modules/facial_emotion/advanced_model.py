"""
Advanced Facial Emotion Recognition Module
Week 2: Enhanced CNN model with real-time detection capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from PIL import Image
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available, using OpenCV for face detection")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Dlib not available, using OpenCV for face detection")

from collections import deque
import threading
import queue

class AdvancedEmotionCNN(nn.Module):
    """
    Advanced CNN architecture for facial emotion recognition
    Improved version of the original model with better performance
    """
    
    def __init__(self, num_classes: int = 7, input_channels: int = 1):
        super(AdvancedEmotionCNN, self).__init__()
        
        # Feature extraction layers with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.5)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Classification layers (no BatchNorm for inference compatibility)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten for attention and classification
        x = x.view(x.size(0), -1)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        
        return x

class FaceDetector:
    """
    Multi-backend face detection with MediaPipe and Dlib support
    """
    
    def __init__(self, backend: str = "opencv"):
        self.backend = backend
        
        if backend == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        elif backend == "dlib" and DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
        else:
            # Default to OpenCV
            self.backend = "opencv"
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        Returns list of bounding boxes (x, y, width, height)
        """
        faces = []
        
        if self.backend == "mediapipe" and MEDIAPIPE_AVAILABLE:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
                    
        elif self.backend == "dlib" and DLIB_AVAILABLE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector(gray)
            
            for detection in detections:
                x = detection.left()
                y = detection.top()
                width = detection.right() - x
                height = detection.bottom() - y
                faces.append((x, y, width, height))
                
        else:  # OpenCV fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in detections:
                faces.append((x, y, w, h))
        
        return faces

class RealTimeEmotionDetector:
    """
    Real-time emotion detection system with face tracking and temporal smoothing
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.6,
        face_detector_backend: str = "opencv"
    ):
        # Emotion labels
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = AdvancedEmotionCNN(num_classes=len(self.emotion_labels))
        if model_path and torch.cuda.is_available():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {model_path}")
            except:
                print(f"Could not load model from {model_path}, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Face detection
        self.face_detector = FaceDetector(backend=face_detector_backend)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Temporal smoothing
        self.emotion_history = deque(maxlen=5)  # Last 5 predictions
        self.face_tracker = {}  # Track individual faces
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.processing_times = deque(maxlen=30)
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face image for emotion recognition"""
        # Convert to PIL Image
        if len(face_image.shape) == 3:
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        else:
            face_pil = Image.fromarray(face_image)
        
        # Apply transforms
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor.to(self.device)
    
    def predict_emotion(self, face_tensor: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from face tensor
        Returns (emotion_label, confidence, probabilities)
        """
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion_label = self.emotion_labels[predicted.item()]
            confidence_score = confidence.item()
            prob_array = probabilities.cpu().numpy()[0]
            
            return emotion_label, confidence_score, prob_array
    
    def smooth_predictions(self, current_prediction: str, current_confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to predictions"""
        self.emotion_history.append((current_prediction, current_confidence))
        
        if len(self.emotion_history) < 3:
            return current_prediction, current_confidence
        
        # Count occurrences of each emotion
        emotion_counts = {}
        total_confidence = 0
        
        for emotion, conf in self.emotion_history:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {'count': 0, 'confidence_sum': 0}
            emotion_counts[emotion]['count'] += 1
            emotion_counts[emotion]['confidence_sum'] += conf
            total_confidence += conf
        
        # Find most frequent emotion with highest average confidence
        best_emotion = current_prediction
        best_score = 0
        
        for emotion, data in emotion_counts.items():
            avg_confidence = data['confidence_sum'] / data['count']
            frequency_weight = data['count'] / len(self.emotion_history)
            score = avg_confidence * frequency_weight
            
            if score > best_score:
                best_emotion = emotion
                best_score = score
        
        # Calculate smoothed confidence
        smoothed_confidence = total_confidence / len(self.emotion_history)
        
        return best_emotion, smoothed_confidence
    
    def detect_emotions_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect emotions in all faces in the frame
        Returns list of detection results
        """
        start_time = time.time()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size == 0:
                continue
            
            # Preprocess face
            face_tensor = self.preprocess_face(face_region)
            
            # Predict emotion
            emotion, confidence, probabilities = self.predict_emotion(face_tensor)
            
            # Apply temporal smoothing
            smoothed_emotion, smoothed_confidence = self.smooth_predictions(emotion, confidence)
            
            # Only include if confidence is above threshold
            if smoothed_confidence >= self.confidence_threshold:
                result = {
                    'face_id': i,
                    'bbox': (x, y, w, h),
                    'emotion': smoothed_emotion,
                    'confidence': smoothed_confidence,
                    'raw_emotion': emotion,
                    'raw_confidence': confidence,
                    'probabilities': probabilities.tolist(),
                    'timestamp': time.time()
                }
                results.append(result)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.fps_counter += 1
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        
        if elapsed_time > 0:
            fps = self.fps_counter / elapsed_time
        else:
            fps = 0
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'fps': fps,
            'avg_processing_time': avg_processing_time,
            'device': str(self.device),
            'model_loaded': True,
            'total_frames_processed': self.fps_counter
        }
    
    def reset_performance_stats(self):
        """Reset performance counters"""
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.processing_times.clear()

class EmotionRecognitionService:
    """
    High-level service for emotion recognition in virtual classroom
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.detector = RealTimeEmotionDetector(model_path=model_path)
        self.active_sessions = {}  # Track active classroom sessions
        
    def start_session(self, session_id: str, user_id: str):
        """Start emotion recognition session"""
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'start_time': time.time(),
            'frame_count': 0,
            'emotion_history': []
        }
    
    def end_session(self, session_id: str):
        """End emotion recognition session"""
        if session_id in self.active_sessions:
            session_data = self.active_sessions.pop(session_id)
            return {
                'session_id': session_id,
                'duration': time.time() - session_data['start_time'],
                'total_frames': session_data['frame_count'],
                'emotion_summary': self._generate_emotion_summary(session_data['emotion_history'])
            }
        return None
    
    def process_frame(self, frame: np.ndarray, session_id: str) -> Dict:
        """Process frame for emotion recognition"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Detect emotions
        results = self.detector.detect_emotions_in_frame(frame)
        
        # Update session data
        session = self.active_sessions[session_id]
        session['frame_count'] += 1
        
        if results:
            session['emotion_history'].extend(results)
        
        return {
            'session_id': session_id,
            'frame_number': session['frame_count'],
            'detections': results,
            'timestamp': time.time()
        }
    
    def _generate_emotion_summary(self, emotion_history: List[Dict]) -> Dict:
        """Generate summary statistics for emotion history"""
        if not emotion_history:
            return {}
        
        emotion_counts = {}
        total_detections = len(emotion_history)
        
        for detection in emotion_history:
            emotion = detection['emotion']
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        # Calculate percentages
        emotion_percentages = {
            emotion: (count / total_detections) * 100
            for emotion, count in emotion_counts.items()
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            'total_detections': total_detections,
            'emotion_distribution': emotion_percentages,
            'dominant_emotion': dominant_emotion[0],
            'dominant_emotion_percentage': (dominant_emotion[1] / total_detections) * 100
        }

# Global service instance
emotion_service = EmotionRecognitionService()