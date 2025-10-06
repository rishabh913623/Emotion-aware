"""
Core configuration settings for the Emotion-Aware Virtual Classroom
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # App info
    app_name: str = "Emotion-Aware Virtual Classroom"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite:///./emotion_classroom.db"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # ML Models
    models_path: str = "ml_modules/models"
    facial_model_path: str = "ml_modules/models/facial_emotion_model.pth"
    audio_model_path: str = "ml_modules/models/audio_emotion_model.pth"
    text_model_path: str = "ml_modules/models/text_sentiment_model"
    fusion_model_path: str = "ml_modules/models/multimodal_fusion_model.pth"
    
    # WebRTC
    stun_servers: list = ["stun:stun.l.google.com:19302"]
    turn_servers: list = []
    
    # File uploads
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".mp4", ".wav", ".mp3"]
    
    # Privacy & Consent
    gdpr_compliance: bool = True
    data_retention_days: int = 365
    anonymization_enabled: bool = True
    
    # Monitoring
    prometheus_enabled: bool = False
    sentry_dsn: str = ""
    
    # Cloud settings
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "emotion-classroom-data"
    
    # Email (for notifications)
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return Settings()

# Emotion labels and mappings
EMOTION_LABELS = {
    'facial': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
    'audio': ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust'],
    'text': ['positive', 'negative', 'neutral']
}

LEARNING_STATES = {
    'engaged': 'Student is actively participating and focused',
    'confused': 'Student appears to be having difficulty understanding',
    'bored': 'Student shows signs of disengagement or boredom',
    'frustrated': 'Student appears frustrated or stressed',
    'curious': 'Student shows interest and curiosity',
    'neutral': 'Student emotional state is neutral/baseline'
}

# WebRTC Configuration
WEBRTC_CONFIG = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"}
    ],
    "iceCandidatePoolSize": 10
}

# ML Model Configuration
MODEL_CONFIG = {
    "facial_emotion": {
        "input_size": (48, 48, 1),
        "num_classes": 7,
        "confidence_threshold": 0.6
    },
    "audio_emotion": {
        "sample_rate": 16000,
        "n_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512,
        "confidence_threshold": 0.5
    },
    "text_sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "max_length": 512,
        "confidence_threshold": 0.7
    },
    "fusion": {
        "weights": {
            "facial": 0.4,
            "audio": 0.3,
            "text": 0.3
        },
        "confidence_threshold": 0.6
    }
}