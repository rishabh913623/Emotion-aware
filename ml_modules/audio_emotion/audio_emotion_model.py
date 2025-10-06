"""
Audio Emotion Recognition Module
Week 3: Real-time speech emotion analysis using MFCC, pitch, and tone features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Dict, Optional, Union
import io
import time
from collections import deque
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract audio features for emotion recognition"""
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        try:
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return mfccs
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return np.zeros((self.n_mfcc, 1))
    
    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch (F0) and voicing probability"""
        try:
            pitches, magnitudes = librosa.core.piptrack(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Get the pitch with highest magnitude at each time
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                pitch_values.append(pitch)
            
            pitch_array = np.array(pitch_values)
            
            # Calculate voicing probability (non-zero pitch ratio)
            voicing_prob = np.mean(pitch_array > 0)
            
            return pitch_array, voicing_prob
        except Exception as e:
            print(f"Pitch extraction error: {e}")
            return np.zeros(1), 0.0
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features"""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            
            # Zero crossing rate
            zero_crossings = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            
            return {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossings),
                'zero_crossing_rate_std': np.std(zero_crossings),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth)
            }
        except Exception as e:
            print(f"Spectral feature extraction error: {e}")
            return {key: 0.0 for key in [
                'spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_rolloff_mean', 'spectral_rolloff_std',
                'zero_crossing_rate_mean', 'zero_crossing_rate_std',
                'spectral_bandwidth_mean', 'spectral_bandwidth_std'
            ]}
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features"""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            return chroma
        except Exception as e:
            print(f"Chroma extraction error: {e}")
            return np.zeros((12, 1))
    
    def extract_tempo_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract tempo and rhythm features"""
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # Beat strength
            beat_strength = np.mean(librosa.util.normalize(
                librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            ))
            
            return {
                'tempo': float(tempo),
                'beat_strength': float(beat_strength)
            }
        except Exception as e:
            print(f"Tempo extraction error: {e}")
            return {'tempo': 0.0, 'beat_strength': 0.0}
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Extract comprehensive feature set"""
        
        # Ensure audio is normalized and has correct type
        if len(audio) == 0:
            return self._get_empty_features()
        
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
        
        features = {}
        
        # MFCC features
        mfccs = self.extract_mfcc(audio)
        features['mfcc'] = mfccs
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)
        features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
        
        # Pitch features
        pitch, voicing_prob = self.extract_pitch(audio)
        features['pitch_mean'] = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0.0
        features['pitch_std'] = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0.0
        features['voicing_probability'] = voicing_prob
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # Chroma features
        chroma = self.extract_chroma_features(audio)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # Tempo features
        tempo_features = self.extract_tempo_features(audio)
        features.update(tempo_features)
        
        # Energy features
        features['energy_mean'] = np.mean(audio ** 2)
        features['energy_std'] = np.std(audio ** 2)
        
        # Statistical features
        features['audio_length'] = len(audio) / self.sample_rate
        features['rms_energy'] = np.sqrt(np.mean(audio ** 2))
        
        return features
    
    def _get_empty_features(self) -> Dict[str, Union[np.ndarray, float]]:
        """Return empty feature set for silent audio"""
        return {
            'mfcc': np.zeros((self.n_mfcc, 1)),
            'mfcc_mean': np.zeros(self.n_mfcc),
            'mfcc_std': np.zeros(self.n_mfcc),
            'mfcc_delta': np.zeros(self.n_mfcc),
            'mfcc_delta2': np.zeros(self.n_mfcc),
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'voicing_probability': 0.0,
            'spectral_centroid_mean': 0.0,
            'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0,
            'spectral_rolloff_std': 0.0,
            'zero_crossing_rate_mean': 0.0,
            'zero_crossing_rate_std': 0.0,
            'spectral_bandwidth_mean': 0.0,
            'spectral_bandwidth_std': 0.0,
            'chroma_mean': np.zeros(12),
            'chroma_std': np.zeros(12),
            'tempo': 0.0,
            'beat_strength': 0.0,
            'energy_mean': 0.0,
            'energy_std': 0.0,
            'audio_length': 0.0,
            'rms_energy': 0.0
        }

class AudioEmotionCNN(nn.Module):
    """CNN model for audio emotion recognition"""
    
    def __init__(self, input_dim: int = 88, num_classes: int = 6):
        super(AudioEmotionCNN, self).__init__()
        
        # Feature dimension calculation
        # MFCC: 13 * 4 (mean, std, delta, delta2) = 52
        # Pitch: 3 (mean, std, voicing_prob) = 3
        # Spectral: 8 features = 8
        # Chroma: 12 * 2 (mean, std) = 24
        # Tempo: 2 features = 2
        # Energy: 4 features = 4
        # Total: 52 + 3 + 8 + 24 + 2 + 4 = 93 features
        
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Feature processing
        features = self.feature_layers(x)
        
        # Apply attention (optional - can be disabled for simpler model)
        # attention_weights = self.attention(features)
        # attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(features)
        
        return output

class RealTimeAudioEmotionDetector:
    """Real-time audio emotion detection system"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        sample_rate: int = 16000
    ):
        # Emotion labels for audio
        self.emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust']
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        
        # Model setup
        self.model = AudioEmotionCNN(input_dim=93, num_classes=len(self.emotion_labels))
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded audio emotion model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}")
                print("Using randomly initialized model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        
        # Temporal smoothing
        self.emotion_history = deque(maxlen=5)
        
        # Performance monitoring
        self.processing_times = deque(maxlen=30)
        
    def preprocess_audio(self, audio_data: Union[np.ndarray, bytes]) -> torch.Tensor:
        """Preprocess audio data for emotion recognition"""
        
        # Convert bytes to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio_array = audio_data.astype(np.float32)
        
        # Resample if needed (basic resampling)
        if len(audio_array) == 0:
            return torch.zeros(1, 93).to(self.device)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio_array)
        
        # Combine all features into a single vector
        feature_vector = self._combine_features(features)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
        
        return feature_tensor
    
    def _combine_features(self, features: Dict[str, Union[np.ndarray, float]]) -> np.ndarray:
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
        
        # Ensure we have the correct number of features
        feature_array = np.array(combined_features[:93])  # Take first 93 features
        
        # Pad with zeros if we don't have enough features
        if len(feature_array) < 93:
            feature_array = np.pad(feature_array, (0, 93 - len(feature_array)), 'constant')
        
        # Handle NaN and infinite values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_array
    
    def predict_emotion(self, feature_tensor: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from audio features
        Returns (emotion_label, confidence, probabilities)
        """
        with torch.no_grad():
            outputs = self.model(feature_tensor)
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
        
        # Count occurrences and calculate weighted confidence
        emotion_votes = {}
        total_weight = 0
        
        for i, (emotion, conf) in enumerate(self.emotion_history):
            weight = (i + 1) / len(self.emotion_history)  # Recent predictions have higher weight
            if emotion not in emotion_votes:
                emotion_votes[emotion] = {'weight': 0, 'confidence_sum': 0}
            emotion_votes[emotion]['weight'] += weight
            emotion_votes[emotion]['confidence_sum'] += conf * weight
            total_weight += weight
        
        # Find best emotion based on weighted voting
        best_emotion = current_prediction
        best_score = 0
        
        for emotion, data in emotion_votes.items():
            avg_confidence = data['confidence_sum'] / data['weight']
            vote_strength = data['weight'] / total_weight
            score = avg_confidence * vote_strength
            
            if score > best_score:
                best_emotion = emotion
                best_score = score
        
        # Calculate smoothed confidence
        smoothed_confidence = min(best_score * 1.2, 1.0)  # Boost confidence for stable predictions
        
        return best_emotion, smoothed_confidence
    
    def detect_emotion_from_audio(self, audio_data: Union[np.ndarray, bytes]) -> Dict:
        """
        Main emotion detection function
        """
        start_time = time.time()
        
        try:
            # Preprocess audio
            feature_tensor = self.preprocess_audio(audio_data)
            
            # Predict emotion
            emotion, confidence, probabilities = self.predict_emotion(feature_tensor)
            
            # Apply temporal smoothing
            smoothed_emotion, smoothed_confidence = self.smooth_predictions(emotion, confidence)
            
            # Only return result if confidence is above threshold
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            result = {
                'emotion': smoothed_emotion if smoothed_confidence >= self.confidence_threshold else 'neutral',
                'confidence': smoothed_confidence,
                'raw_emotion': emotion,
                'raw_confidence': confidence,
                'probabilities': {
                    label: float(prob) for label, prob in zip(self.emotion_labels, probabilities)
                },
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"Audio emotion detection error: {e}")
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'raw_emotion': 'error',
                'raw_confidence': 0.0,
                'probabilities': {label: 0.0 for label in self.emotion_labels},
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'avg_processing_time': avg_processing_time,
            'device': str(self.device),
            'model_loaded': True,
            'total_processed': len(self.processing_times),
            'sample_rate': self.sample_rate
        }
    
    def reset_performance_stats(self):
        """Reset performance counters"""
        self.processing_times.clear()
        self.emotion_history.clear()

class AudioEmotionService:
    """High-level service for audio emotion recognition in virtual classroom"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.detector = RealTimeAudioEmotionDetector(model_path=model_path)
        self.active_sessions = {}  # Track active audio sessions
        
    def start_session(self, session_id: str, user_id: str):
        """Start audio emotion recognition session"""
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'start_time': time.time(),
            'audio_chunks_processed': 0,
            'emotion_history': []
        }
    
    def end_session(self, session_id: str):
        """End audio emotion recognition session"""
        if session_id in self.active_sessions:
            session_data = self.active_sessions.pop(session_id)
            return {
                'session_id': session_id,
                'duration': time.time() - session_data['start_time'],
                'total_chunks': session_data['audio_chunks_processed'],
                'emotion_summary': self._generate_emotion_summary(session_data['emotion_history'])
            }
        return None
    
    def process_audio_chunk(self, audio_data: Union[np.ndarray, bytes], session_id: str) -> Dict:
        """Process audio chunk for emotion recognition"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Detect emotion
        result = self.detector.detect_emotion_from_audio(audio_data)
        
        # Update session data
        session = self.active_sessions[session_id]
        session['audio_chunks_processed'] += 1
        
        if result['emotion'] != 'error':
            session['emotion_history'].append(result)
        
        result['session_id'] = session_id
        result['chunk_number'] = session['audio_chunks_processed']
        
        return result
    
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
        
        # Calculate average confidence
        avg_confidence = np.mean([d['confidence'] for d in emotion_history])
        
        return {
            'total_detections': total_detections,
            'emotion_distribution': emotion_percentages,
            'dominant_emotion': dominant_emotion[0],
            'dominant_emotion_percentage': (dominant_emotion[1] / total_detections) * 100,
            'average_confidence': float(avg_confidence)
        }

# Global service instance
audio_emotion_service = AudioEmotionService()