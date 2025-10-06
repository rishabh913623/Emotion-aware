"""
Dashboard API for Emotion-Aware Virtual Classroom
Week 6: Real-time instructor dashboard with class mood and student analytics
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque

from backend.core.database import get_db, User, Classroom, EmotionData, ClassSession
from backend.core.config import get_settings, LEARNING_STATES
from ml_modules.multimodal_fusion.fusion_model import MultimodalEmotionFusion

router = APIRouter()
security = HTTPBearer()
settings = get_settings()

# Global dashboard state management
class DashboardManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.class_sessions: Dict[str, Dict] = {}
        self.emotion_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.fusion_model = MultimodalEmotionFusion()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def broadcast_to_instructors(self, message: Dict, class_id: str):
        """Broadcast real-time updates to instructor dashboards"""
        for session_id, websocket in self.active_connections.items():
            if session_id.startswith(f"instructor_{class_id}_"):
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    # Remove disconnected websocket
                    self.disconnect(session_id)
    
    def update_class_emotion(self, class_id: str, student_id: str, emotion_data: Dict):
        """Update real-time emotion data for a class"""
        timestamp = datetime.now().isoformat()
        
        # Store in emotion history
        self.emotion_history[f"{class_id}_{student_id}"].append({
            "timestamp": timestamp,
            **emotion_data
        })
        
        # Update class session data
        if class_id not in self.class_sessions:
            self.class_sessions[class_id] = {
                "students": {},
                "class_mood": {"engaged": 0, "confused": 0, "bored": 0, "frustrated": 0, "curious": 0, "neutral": 0},
                "last_updated": timestamp
            }
        
        # Update student data
        self.class_sessions[class_id]["students"][student_id] = {
            "current_emotion": emotion_data,
            "last_updated": timestamp
        }
        
        # Calculate class mood
        self._calculate_class_mood(class_id)
        
    def _calculate_class_mood(self, class_id: str):
        """Calculate overall class mood from individual student emotions"""
        if class_id not in self.class_sessions:
            return
            
        session = self.class_sessions[class_id]
        mood_counts = defaultdict(int)
        total_students = 0
        
        for student_data in session["students"].values():
            emotion = student_data["current_emotion"].get("learning_state", "neutral")
            mood_counts[emotion] += 1
            total_students += 1
        
        # Calculate percentages
        class_mood = {}
        for state in LEARNING_STATES.keys():
            class_mood[state] = (mood_counts[state] / max(total_students, 1)) * 100
        
        session["class_mood"] = class_mood
        session["total_students"] = total_students
        session["last_updated"] = datetime.now().isoformat()

# Global dashboard manager
dashboard_manager = DashboardManager()

@router.websocket("/ws/dashboard/{class_id}/{instructor_id}")
async def dashboard_websocket(websocket: WebSocket, class_id: str, instructor_id: str):
    """WebSocket connection for real-time dashboard updates"""
    session_id = f"instructor_{class_id}_{instructor_id}"
    await dashboard_manager.connect(websocket, session_id)
    
    try:
        # Send initial class state
        if class_id in dashboard_manager.class_sessions:
            initial_state = {
                "type": "class_state",
                "data": dashboard_manager.class_sessions[class_id]
            }
            await websocket.send_text(json.dumps(initial_state))
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        dashboard_manager.disconnect(session_id)

@router.post("/api/emotion-update")
async def receive_emotion_update(
    class_id: str,
    student_id: str,
    emotion_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Receive emotion updates from students and broadcast to instructors"""
    
    # Validate class and student exist
    class_session = db.query(ClassSession).filter(
        ClassSession.id == class_id,
        ClassSession.is_active == True
    ).first()
    
    if not class_session:
        raise HTTPException(status_code=404, detail="Class session not found")
    
    # Update dashboard state
    dashboard_manager.update_class_emotion(class_id, student_id, emotion_data)
    
    # Store in database
    emotion_record = EmotionData(
        user_id=student_id,
        classroom_id=class_id,
        facial_emotion=emotion_data.get("facial_emotion"),
        audio_emotion=emotion_data.get("audio_emotion"),
        text_sentiment=emotion_data.get("text_sentiment"),
        learning_state=emotion_data.get("learning_state"),
        confidence_score=emotion_data.get("confidence", 0.0),
        timestamp=datetime.now()
    )
    
    db.add(emotion_record)
    db.commit()
    
    # Broadcast to instructor dashboards
    broadcast_message = {
        "type": "emotion_update",
        "class_id": class_id,
        "student_id": student_id,
        "emotion_data": emotion_data,
        "class_mood": dashboard_manager.class_sessions[class_id]["class_mood"],
        "timestamp": datetime.now().isoformat()
    }
    
    await dashboard_manager.broadcast_to_instructors(broadcast_message, class_id)
    
    return {"status": "success", "message": "Emotion data updated"}

@router.get("/api/class/{class_id}/current-state")
async def get_current_class_state(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Get current class emotional state"""
    
    if class_id not in dashboard_manager.class_sessions:
        return {
            "class_id": class_id,
            "class_mood": {state: 0 for state in LEARNING_STATES.keys()},
            "students": {},
            "total_students": 0,
            "last_updated": None
        }
    
    return {
        "class_id": class_id,
        **dashboard_manager.class_sessions[class_id]
    }

@router.get("/api/class/{class_id}/student/{student_id}/history")
async def get_student_emotion_history(
    class_id: str,
    student_id: str,
    minutes: int = 30,
    db: Session = Depends(get_db)
):
    """Get emotion history for a specific student"""
    
    # Get from real-time data
    history_key = f"{class_id}_{student_id}"
    real_time_history = list(dashboard_manager.emotion_history[history_key])
    
    # Get from database for longer history
    since_time = datetime.now() - timedelta(minutes=minutes)
    db_history = db.query(EmotionData).filter(
        EmotionData.user_id == student_id,
        EmotionData.classroom_id == class_id,
        EmotionData.timestamp >= since_time
    ).order_by(EmotionData.timestamp.desc()).limit(100).all()
    
    # Combine and format
    history = []
    
    # Add database records
    for record in db_history:
        history.append({
            "timestamp": record.timestamp.isoformat(),
            "facial_emotion": record.facial_emotion,
            "audio_emotion": record.audio_emotion,
            "text_sentiment": record.text_sentiment,
            "learning_state": record.learning_state,
            "confidence": record.confidence_score
        })
    
    # Add real-time records (remove duplicates)
    db_timestamps = {record.timestamp.isoformat()[:19] for record in db_history}
    for rt_record in real_time_history:
        if rt_record["timestamp"][:19] not in db_timestamps:
            history.append(rt_record)
    
    # Sort by timestamp
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "student_id": student_id,
        "class_id": class_id,
        "history": history[:100],  # Limit to 100 records
        "duration_minutes": minutes
    }

@router.get("/api/class/{class_id}/alerts")
async def get_class_alerts(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Get current alerts for the class (confusion, boredom, etc.)"""
    
    if class_id not in dashboard_manager.class_sessions:
        return {"alerts": []}
    
    session = dashboard_manager.class_sessions[class_id]
    alerts = []
    
    # Check for high confusion levels
    if session["class_mood"].get("confused", 0) > 30:
        alerts.append({
            "type": "high_confusion",
            "severity": "warning",
            "message": f"{session['class_mood']['confused']:.1f}% of students appear confused",
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Consider clarifying the current topic or asking for questions"
        })
    
    # Check for high boredom levels
    if session["class_mood"].get("bored", 0) > 25:
        alerts.append({
            "type": "high_boredom",
            "severity": "warning",
            "message": f"{session['class_mood']['bored']:.1f}% of students appear bored",
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Consider adding interactive elements or changing the pace"
        })
    
    # Check for low engagement
    engagement_states = ["engaged", "curious"]
    total_engagement = sum(session["class_mood"].get(state, 0) for state in engagement_states)
    
    if total_engagement < 40 and session["total_students"] > 3:
        alerts.append({
            "type": "low_engagement",
            "severity": "info",
            "message": f"Only {total_engagement:.1f}% of students appear engaged",
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Consider using interactive teaching methods to increase engagement"
        })
    
    # Check individual student alerts
    for student_id, student_data in session["students"].items():
        emotion = student_data["current_emotion"]
        if emotion.get("learning_state") == "frustrated" and emotion.get("confidence", 0) > 0.7:
            alerts.append({
                "type": "student_frustrated",
                "severity": "urgent",
                "message": f"Student {student_id} appears frustrated",
                "timestamp": datetime.now().isoformat(),
                "suggestion": "Consider providing individual assistance",
                "student_id": student_id
            })
    
    return {"alerts": alerts}

@router.get("/api/class/{class_id}/summary")
async def get_class_summary(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Get summary statistics for the current class session"""
    
    # Get class session info
    class_session = db.query(ClassSession).filter(
        ClassSession.id == class_id
    ).first()
    
    if not class_session:
        raise HTTPException(status_code=404, detail="Class session not found")
    
    # Calculate session duration
    session_start = class_session.start_time
    session_duration = datetime.now() - session_start if session_start else timedelta(0)
    
    # Get current state
    current_state = dashboard_manager.class_sessions.get(class_id, {
        "class_mood": {state: 0 for state in LEARNING_STATES.keys()},
        "students": {},
        "total_students": 0
    })
    
    # Get emotion statistics from database
    emotion_stats = db.query(EmotionData).filter(
        EmotionData.classroom_id == class_id,
        EmotionData.timestamp >= session_start if session_start else datetime.now()
    ).all()
    
    # Calculate overall statistics
    state_counts = defaultdict(int)
    total_records = len(emotion_stats)
    
    for record in emotion_stats:
        if record.learning_state:
            state_counts[record.learning_state] += 1
    
    overall_distribution = {}
    for state in LEARNING_STATES.keys():
        overall_distribution[state] = (state_counts[state] / max(total_records, 1)) * 100
    
    return {
        "class_id": class_id,
        "session_duration_minutes": int(session_duration.total_seconds() / 60),
        "current_students": current_state["total_students"],
        "current_mood": current_state["class_mood"],
        "overall_mood_distribution": overall_distribution,
        "total_emotion_records": total_records,
        "session_start": session_start.isoformat() if session_start else None,
        "last_updated": current_state.get("last_updated")
    }

@router.post("/api/class/{class_id}/start")
async def start_class_session(
    class_id: str,
    instructor_id: str,
    db: Session = Depends(get_db)
):
    """Start a new class session"""
    
    # Check if instructor exists and has permission
    instructor = db.query(User).filter(User.id == instructor_id).first()
    if not instructor or instructor.role != "instructor":
        raise HTTPException(status_code=403, detail="Only instructors can start class sessions")
    
    # Create or update class session
    class_session = db.query(ClassSession).filter(
        ClassSession.id == class_id
    ).first()
    
    if not class_session:
        class_session = ClassSession(
            id=class_id,
            instructor_id=instructor_id,
            start_time=datetime.now(),
            is_active=True
        )
        db.add(class_session)
    else:
        class_session.start_time = datetime.now()
        class_session.is_active = True
        class_session.instructor_id = instructor_id
    
    db.commit()
    
    # Initialize dashboard session
    dashboard_manager.class_sessions[class_id] = {
        "students": {},
        "class_mood": {state: 0 for state in LEARNING_STATES.keys()},
        "total_students": 0,
        "last_updated": datetime.now().isoformat(),
        "instructor_id": instructor_id
    }
    
    return {
        "status": "success",
        "message": "Class session started",
        "class_id": class_id,
        "start_time": class_session.start_time.isoformat()
    }

@router.post("/api/class/{class_id}/end")
async def end_class_session(
    class_id: str,
    instructor_id: str,
    db: Session = Depends(get_db)
):
    """End the current class session"""
    
    class_session = db.query(ClassSession).filter(
        ClassSession.id == class_id,
        ClassSession.instructor_id == instructor_id
    ).first()
    
    if not class_session:
        raise HTTPException(status_code=404, detail="Class session not found")
    
    # Update session
    class_session.end_time = datetime.now()
    class_session.is_active = False
    db.commit()
    
    # Clean up dashboard state
    if class_id in dashboard_manager.class_sessions:
        final_summary = dashboard_manager.class_sessions[class_id].copy()
        del dashboard_manager.class_sessions[class_id]
    else:
        final_summary = {}
    
    return {
        "status": "success",
        "message": "Class session ended",
        "class_id": class_id,
        "end_time": class_session.end_time.isoformat(),
        "final_summary": final_summary
    }