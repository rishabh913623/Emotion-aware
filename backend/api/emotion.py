"""
Emotion Recognition API endpoints
Week 2: Facial emotion recognition integrated
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import base64
import cv2
import numpy as np
from PIL import Image
import io
from pydantic import BaseModel
import sys
import os

# Add ml_modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_modules'))

from backend.core.database import get_db, EmotionData, User
from backend.api.auth import get_current_active_user

# Import emotion recognition models - temporarily disabled for demo
facial_service = None
audio_service = None
text_service = None
print("Warning: ML emotion recognition services disabled for demo")

router = APIRouter()

# Pydantic models
class EmotionResult(BaseModel):
    facial_emotion: Optional[str] = None
    facial_confidence: Optional[float] = None
    audio_emotion: Optional[str] = None
    audio_confidence: Optional[float] = None
    text_sentiment: Optional[str] = None
    text_confidence: Optional[float] = None
    predicted_state: Optional[str] = None
    fusion_confidence: Optional[float] = None
    timestamp: datetime

class EmotionAnalysisRequest(BaseModel):
    image_data: Optional[str] = None  # base64 encoded
    audio_data: Optional[str] = None  # base64 encoded
    text_data: Optional[str] = None
    classroom_id: str
    session_id: str

class EmotionSummaryResponse(BaseModel):
    classroom_id: str
    user_id: str
    total_detections: int
    emotion_distribution: dict
    dominant_emotion: str
    average_confidence: float
    session_duration: Optional[float] = None

@router.post("/analyze")
async def analyze_emotions(
    request: EmotionAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Analyze emotions from multimodal input
    Week 2: Facial emotion recognition implemented
    """
    
    facial_emotion = None
    facial_confidence = 0.0
    audio_emotion = "neutral"  # Will be updated if audio data provided
    audio_confidence = 0.0
    text_sentiment = "neutral"  # Will be updated if text data provided
    text_confidence = 0.0
    
    # Process facial emotion if image data provided
    if request.image_data and facial_service:
        try:
            # Decode base64 image
            image_data = base64.b64decode(request.image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotions
            results = facial_service.detector.detect_emotions_in_frame(frame)
            
            if results:
                # Use the first detected face
                result = results[0]
                facial_emotion = result['emotion']
                facial_confidence = result['confidence']
        
        except Exception as e:
            print(f"Facial emotion analysis error: {e}")
            facial_emotion = "error"
            facial_confidence = 0.0
    
    # Audio emotion analysis (Week 3 implemented)
    if request.audio_data and audio_service:
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(request.audio_data)
            
            # Convert bytes to numpy array (assuming float32 format)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Detect emotion
            audio_result = audio_service.detector.detect_emotion_from_audio(audio_array)
            
            if audio_result['emotion'] != 'error':
                audio_emotion = audio_result['emotion']
                audio_confidence = audio_result['confidence']
        
        except Exception as e:
            print(f"Audio emotion analysis error: {e}")
            audio_emotion = "error"
            audio_confidence = 0.0
    
    # Text sentiment analysis (Week 4 implemented)
    if request.text_data and text_service:
        try:
            # Analyze text sentiment
            text_result = text_service.analyzer.analyze_text(request.text_data)
            
            if text_result.get('learning_state') != 'error':
                # Use learning state as sentiment
                text_sentiment = text_result.get('smoothed_learning_state', text_result['learning_state'])
                text_confidence = text_result.get('smoothed_confidence', text_result['confidence'])
        
        except Exception as e:
            print(f"Text sentiment analysis error: {e}")
            text_sentiment = "error"
            text_confidence = 0.0
    
    # Multimodal fusion (enhanced with audio)
    predicted_state = "engaged"  # Default state
    fusion_confidence = 0.5
    
    # Enhanced emotion to learning state mapping
    emotion_scores = {'engaged': 0, 'bored': 0, 'confused': 0, 'frustrated': 0, 'curious': 0}
    total_weight = 0
    
    # Facial emotion contribution
    if facial_emotion and facial_confidence > 0.3:
        weight = facial_confidence * 0.6  # 60% weight for facial
        if facial_emotion in ['happy', 'surprise']:
            emotion_scores['engaged'] += weight
            emotion_scores['curious'] += weight * 0.5
        elif facial_emotion == 'sad':
            emotion_scores['bored'] += weight
        elif facial_emotion in ['fear', 'disgust']:
            emotion_scores['confused'] += weight
        elif facial_emotion == 'angry':
            emotion_scores['frustrated'] += weight
        total_weight += weight
    
    # Audio emotion contribution  
    if audio_emotion and audio_confidence > 0.3:
        weight = audio_confidence * 0.4  # 40% weight for audio
        if audio_emotion == 'happy':
            emotion_scores['engaged'] += weight
            emotion_scores['curious'] += weight * 0.3
        elif audio_emotion == 'sad':
            emotion_scores['bored'] += weight
        elif audio_emotion == 'fear':
            emotion_scores['confused'] += weight
        elif audio_emotion == 'angry':
            emotion_scores['frustrated'] += weight
        total_weight += weight
    
    # Determine final state
    if total_weight > 0:
        # Normalize scores
        for state in emotion_scores:
            emotion_scores[state] /= total_weight
        
        # Find dominant state
        predicted_state = max(emotion_scores, key=emotion_scores.get)
        fusion_confidence = emotion_scores[predicted_state]
        
        # Boost confidence if multiple modalities agree
        if facial_emotion and audio_emotion and facial_confidence > 0.3 and audio_confidence > 0.3:
            fusion_confidence = min(fusion_confidence * 1.2, 1.0)
    
    # Store emotion data in database
    emotion_data = EmotionData(
        user_id=current_user.id,
        classroom_id=request.classroom_id,
        session_id=request.session_id,
        facial_emotion=facial_emotion,
        facial_confidence=facial_confidence,
        audio_emotion=audio_emotion,
        audio_confidence=0.0,
        text_sentiment=text_sentiment,
        text_confidence=0.0,
        predicted_state=predicted_state,
        fusion_confidence=fusion_confidence
    )
    
    db.add(emotion_data)
    db.commit()
    
    return {
        "id": str(emotion_data.id),
        "facial_emotion": facial_emotion,
        "facial_confidence": facial_confidence,
        "audio_emotion": audio_emotion,
        "text_sentiment": text_sentiment,
        "predicted_state": predicted_state,
        "fusion_confidence": fusion_confidence,
        "timestamp": emotion_data.timestamp.isoformat()
    }

@router.get("/history/{classroom_id}")
async def get_emotion_history(
    classroom_id: str,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get emotion analysis history for a classroom"""
    
    # Query emotion data for the classroom and user
    emotion_records = db.query(EmotionData).filter(
        EmotionData.classroom_id == classroom_id,
        EmotionData.user_id == current_user.id
    ).order_by(EmotionData.timestamp.desc()).limit(limit).all()
    
    # Convert to response format
    history = []
    for record in emotion_records:
        history.append({
            "id": str(record.id),
            "timestamp": record.timestamp.isoformat(),
            "facial_emotion": record.facial_emotion,
            "facial_confidence": record.facial_confidence,
            "audio_emotion": record.audio_emotion,
            "audio_confidence": record.audio_confidence,
            "text_sentiment": record.text_sentiment,
            "text_confidence": record.text_confidence,
            "predicted_state": record.predicted_state,
            "fusion_confidence": record.fusion_confidence
        })
    
@router.get("/summary/{classroom_id}")
async def get_emotion_summary(
    classroom_id: str,
    session_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get emotion analysis summary for a classroom session"""
    
    # Build query
    query = db.query(EmotionData).filter(
        EmotionData.classroom_id == classroom_id,
        EmotionData.user_id == current_user.id
    )
    
    if session_id:
        query = query.filter(EmotionData.session_id == session_id)
    
    emotion_records = query.all()
    
    if not emotion_records:
        return {
            "classroom_id": classroom_id,
            "user_id": str(current_user.id),
            "total_detections": 0,
            "emotion_distribution": {},
            "dominant_emotion": None,
            "average_confidence": 0.0
        }
    
    # Calculate statistics
    total_detections = len(emotion_records)
    emotion_counts = {}
    confidence_sum = 0
    
    for record in emotion_records:
        # Count facial emotions
        if record.facial_emotion:
            emotion = record.facial_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_sum += record.facial_confidence or 0
    
    # Calculate percentages
    emotion_distribution = {
        emotion: (count / total_detections) * 100
        for emotion, count in emotion_counts.items()
    }
    
    # Find dominant emotion
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
    
    # Calculate average confidence
    average_confidence = confidence_sum / total_detections if total_detections > 0 else 0.0
    
    # Calculate session duration
    session_duration = None
    if emotion_records:
        start_time = min(record.timestamp for record in emotion_records)
        end_time = max(record.timestamp for record in emotion_records)
        session_duration = (end_time - start_time).total_seconds()
    
    return EmotionSummaryResponse(
        classroom_id=classroom_id,
        user_id=str(current_user.id),
        total_detections=total_detections,
        emotion_distribution=emotion_distribution,
        dominant_emotion=dominant_emotion,
        average_confidence=average_confidence,
        session_duration=session_duration
    )

@router.post("/start-session")
async def start_emotion_session(
    classroom_id: str,
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Start an emotion recognition session"""
    
    if facial_service:
        facial_service.start_session(session_id, str(current_user.id))
    
    return {
        "message": "Emotion recognition session started",
        "session_id": session_id,
        "classroom_id": classroom_id,
        "user_id": str(current_user.id)
    }

@router.post("/end-session")
async def end_emotion_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """End an emotion recognition session"""
    
    summary = None
    if facial_service:
        summary = facial_service.end_session(session_id)
    
    return {
        "message": "Emotion recognition session ended",
        "session_id": session_id,
        "summary": summary
    }

@router.get("/real-time/{session_id}")
async def get_realtime_emotions(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get real-time emotion data for a session (last 10 seconds)"""
    
    # Get recent emotion data (last 10 seconds)
    cutoff_time = datetime.utcnow() - timedelta(seconds=10)
    
    recent_emotions = db.query(EmotionData).filter(
        EmotionData.session_id == session_id,
        EmotionData.user_id == current_user.id,
        EmotionData.timestamp >= cutoff_time
    ).order_by(EmotionData.timestamp.desc()).all()
    
    # Format for real-time display
    emotions = []
    for record in recent_emotions:
        emotions.append({
            "timestamp": record.timestamp.isoformat(),
            "facial_emotion": record.facial_emotion,
            "confidence": record.facial_confidence,
            "predicted_state": record.predicted_state
        })
    
    # Get current state (most recent)
    current_state = emotions[0] if emotions else None
    
    return {
        "session_id": session_id,
        "current_state": current_state,
        "recent_emotions": emotions,
        "timestamp": datetime.utcnow().isoformat()
    }