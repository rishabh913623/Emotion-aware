#!/usr/bin/env python3
"""
Simplified backend runner for development with real user support
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

# Data models
class UserRegistration(BaseModel):
    username: str
    email: str
    role: str = "student"  # student, instructor

class JoinRoom(BaseModel):
    username: str
    room_id: Optional[str] = None

class Participant:
    def __init__(self, user_id: str, username: str, role: str, websocket: WebSocket):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.websocket = websocket
        self.video_enabled = True
        self.audio_enabled = True
        self.hand_raised = False
        self.joined_at = datetime.now()

class ClassroomRoom:
    def __init__(self, room_id: str, host_user_id: str, room_name: str):
        self.room_id = room_id
        self.host_user_id = host_user_id
        self.room_name = room_name
        self.participants: Dict[str, Participant] = {}
        self.created_at = datetime.now()
        self.is_active = True
        self.session_start_time = datetime.now()
        
    def add_participant(self, participant: Participant):
        self.participants[participant.user_id] = participant
        
    def remove_participant(self, user_id: str):
        if user_id in self.participants:
            del self.participants[user_id]
            
    def get_participant_list(self):
        return [
            {
                "user_id": p.user_id,
                "username": p.username,
                "role": p.role,
                "video_enabled": p.video_enabled,
                "audio_enabled": p.audio_enabled,
                "hand_raised": p.hand_raised,
                "joined_at": p.joined_at.isoformat()
            }
            for p in self.participants.values()
        ]
    
    async def broadcast_to_all(self, message: dict, exclude_user_id: str = None):
        for user_id, participant in self.participants.items():
            if user_id != exclude_user_id:
                try:
                    await participant.websocket.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to send to {user_id}: {e}")

# Global state
users_db: Dict[str, dict] = {}  # Simple in-memory user storage
rooms_db: Dict[str, ClassroomRoom] = {}  # Active rooms
active_connections: Dict[str, WebSocket] = {}  # User ID -> WebSocket
attendance_db: Dict[str, List[dict]] = {}  # Room ID -> List of attendance records

# Create FastAPI app
app = FastAPI(
    title="Emotion-Aware Virtual Classroom API",
    description="Real-time virtual classroom with emotion recognition",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Emotion-Aware Virtual Classroom API",
        "version": "1.0.0",
        "status": "healthy",
        "features": [
            "real_user_connections",
            "webrtc_video_conferencing",
            "multimodal_emotion_recognition",
            "real_time_analytics"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "environment": "development",
        "active_rooms": len(rooms_db),
        "total_users": len(users_db),
        "features": {
            "real_users": True,
            "webrtc_streaming": True,
            "emotion_recognition": True,
            "real_time_dashboard": True
        }
    }

# User Management Endpoints
@app.post("/api/auth/register")
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    user_id = str(uuid.uuid4())
    
    # Check if email already exists
    existing_user_id = None
    for uid, existing_user in users_db.items():
        if existing_user["email"] == user_data.email:
            existing_user_id = uid
            break
    
    if existing_user_id:
        # Return existing user instead of error
        existing_user = users_db[existing_user_id]
        return {
            "user_id": existing_user_id,
            "username": existing_user["username"],
            "role": existing_user["role"],
            "message": "User already exists, logged in successfully"
        }
    
    users_db[user_id] = {
        "user_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "role": user_data.role,
        "created_at": datetime.now().isoformat(),
        "is_online": False
    }
    
    return {
        "user_id": user_id,
        "username": user_data.username,
        "role": user_data.role,
        "message": "User registered successfully"
    }

@app.post("/api/auth/login")
async def login_user(login_data: dict):
    """Simple login (for demo purposes)"""
    username = login_data.get("username")
    
    # Find user by username
    for user_id, user in users_db.items():
        if user["username"] == username:
            users_db[user_id]["is_online"] = True
            return {
                "user_id": user_id,
                "username": user["username"],
                "role": user["role"],
                "token": f"demo_token_{user_id}",
                "message": "Login successful"
            }
    
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/api/users/online")
async def get_online_users():
    """Get list of online users"""
    online_users = [
        {
            "user_id": user_id,
            "username": user["username"],
            "role": user["role"]
        }
        for user_id, user in users_db.items()
        if user.get("is_online", False)
    ]
    return {"online_users": online_users, "count": len(online_users)}

# Room Management Endpoints
@app.post("/api/classroom/create-room")
async def create_room(room_data: dict):
    """Create a new virtual classroom"""
    host_user_id = room_data.get("host_user_id")
    room_name = room_data.get("room_name", "Virtual Classroom")
    
    if host_user_id not in users_db:
        raise HTTPException(status_code=404, detail="Host user not found")
    
    room_id = str(uuid.uuid4())
    room = ClassroomRoom(room_id, host_user_id, room_name)
    rooms_db[room_id] = room
    
    return {
        "room_id": room_id,
        "room_name": room_name,
        "host_user_id": host_user_id,
        "join_url": f"/classroom/{room_id}",
        "created_at": room.created_at.isoformat()
    }

@app.get("/api/classroom/rooms")
async def list_active_rooms():
    """List all active rooms"""
    rooms = []
    for room_id, room in rooms_db.items():
        if room.is_active:
            host_info = users_db.get(room.host_user_id, {})
            rooms.append({
                "room_id": room_id,
                "room_name": room.room_name,
                "host_username": host_info.get("username", "Unknown"),
                "participant_count": len(room.participants),
                "created_at": room.created_at.isoformat()
            })
    
    return {"rooms": rooms, "total_rooms": len(rooms)}

@app.get("/api/classroom/room/{room_id}")
async def get_room_details(room_id: str):
    """Get room details and participants"""
    room = find_room_by_id(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    host_info = users_db.get(room.host_user_id, {})
    
    return {
        "room_id": room.room_id,
        "room_name": room.room_name,
        "host_username": host_info.get("username", "Unknown"),
        "participants": room.get_participant_list(),
        "participant_count": len(room.participants),
        "created_at": room.created_at.isoformat(),
        "is_active": room.is_active
    }

@app.delete("/api/classroom/room/{room_id}")
async def close_room(room_id: str, user_data: dict):
    """Close a room (host only)"""
    room = find_room_by_id(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    user_id = user_data.get("user_id")
    
    if user_id != room.host_user_id:
        raise HTTPException(status_code=403, detail="Only host can close the room")
    
    # Notify all participants
    await room.broadcast_to_all({
        "type": "room_closed",
        "message": "Room has been closed by the host",
        "timestamp": datetime.now().isoformat()
    })
    
    room.is_active = False
    # Find the actual room ID in the database to delete it
    actual_room_id = None
    for rid, r in rooms_db.items():
        if r == room:
            actual_room_id = rid
            break
    
    if actual_room_id:
        del rooms_db[actual_room_id]
    
    return {"message": "Room closed successfully"}

# Attendance Management Endpoints
@app.get("/api/attendance/room/{room_id}")
async def get_room_attendance(room_id: str):
    """Get attendance records for a specific room"""
    # Find room by full or short ID
    room = find_room_by_id(room_id)
    if not room:
        # Check if we have attendance for this room even if it's closed
        actual_room_id = None
        if room_id in attendance_db:
            actual_room_id = room_id
        else:
            # Try to find by short ID
            for rid in attendance_db.keys():
                if rid.startswith(room_id):
                    actual_room_id = rid
                    break
        
        if not actual_room_id:
            raise HTTPException(status_code=404, detail="Room not found or no attendance records")
        
        attendance_records = attendance_db.get(actual_room_id, [])
    else:
        attendance_records = attendance_db.get(room.room_id, [])
    
    # Calculate statistics
    total_students = sum(1 for record in attendance_records if record["role"] == "student")
    total_instructors = sum(1 for record in attendance_records if record["role"] == "instructor")
    
    return {
        "room_id": room_id,
        "attendance_records": attendance_records,
        "statistics": {
            "total_attendees": len(attendance_records),
            "students_present": total_students,
            "instructors_present": total_instructors
        },
        "generated_at": datetime.now().isoformat()
    }

@app.get("/api/attendance/room/{room_id}/export")
async def export_attendance(room_id: str, format: str = "json"):
    """Export attendance records (JSON or CSV format)"""
    room = find_room_by_id(room_id)
    
    # Get attendance records
    actual_room_id = room.room_id if room else room_id
    if actual_room_id not in attendance_db and room:
        # Try to find by short ID
        for rid in attendance_db.keys():
            if rid.startswith(room_id):
                actual_room_id = rid
                break
    
    attendance_records = attendance_db.get(actual_room_id, [])
    
    if not attendance_records:
        raise HTTPException(status_code=404, detail="No attendance records found")
    
    if format == "csv":
        # Generate CSV format
        csv_content = "User ID,Username,Role,Joined At,Status\n"
        for record in attendance_records:
            csv_content += f"{record['user_id']},{record['username']},{record['role']},{record['joined_at']},{record['status']}\n"
        
        return {
            "format": "csv",
            "content": csv_content,
            "filename": f"attendance_{room_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    else:
        # JSON format (default)
        return {
            "format": "json",
            "room_id": room_id,
            "attendance_records": attendance_records,
            "exported_at": datetime.now().isoformat()
        }

@app.get("/api/attendance/student/{user_id}")
async def get_student_attendance_history(user_id: str):
    """Get attendance history for a specific student across all rooms"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_attendance = []
    for room_id, records in attendance_db.items():
        for record in records:
            if record["user_id"] == user_id:
                user_attendance.append({
                    "room_id": room_id,
                    "room_short_id": room_id[:8],
                    **record
                })
    
    return {
        "user_id": user_id,
        "username": users_db[user_id]["username"],
        "total_sessions_attended": len(user_attendance),
        "attendance_history": user_attendance
    }

@app.get("/api/attendance/summary")
async def get_attendance_summary():
    """Get overall attendance summary across all rooms"""
    total_records = sum(len(records) for records in attendance_db.values())
    rooms_with_attendance = len(attendance_db)
    
    room_summaries = []
    for room_id, records in attendance_db.items():
        room = rooms_db.get(room_id)
        room_name = room.room_name if room else f"Room {room_id[:8]}"
        
        room_summaries.append({
            "room_id": room_id,
            "room_short_id": room_id[:8],
            "room_name": room_name,
            "total_attendees": len(records),
            "students_count": sum(1 for r in records if r["role"] == "student"),
            "first_join": min(r["joined_at"] for r in records) if records else None,
            "last_join": max(r["joined_at"] for r in records) if records else None
        })
    
    return {
        "total_attendance_records": total_records,
        "rooms_with_attendance": rooms_with_attendance,
        "room_summaries": room_summaries,
        "generated_at": datetime.now().isoformat()
    }

def find_room_by_id(room_input: str) -> Optional[ClassroomRoom]:
    """Find room by full ID or short ID (first 8 characters)"""
    # First try exact match
    if room_input in rooms_db:
        return rooms_db[room_input]
    
    # If input is 8 characters, try to find by prefix
    if len(room_input) == 8:
        for room_id, room in rooms_db.items():
            if room_id.startswith(room_input):
                return room
    
    return None
@app.websocket("/ws/classroom/{room_id}")
async def websocket_classroom_endpoint(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for real-time classroom communication"""
    await websocket.accept()
    
    user_id = None
    room = None
    
    try:
        # Wait for user authentication message
        auth_data = await websocket.receive_text()
        auth_info = json.loads(auth_data)
        
        user_id = auth_info.get("user_id")
        username = auth_info.get("username")
        role = auth_info.get("role", "student")
        
        if not user_id or user_id not in users_db:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid user credentials"
            }))
            return
        
        # Get or create room - handle both full and short room IDs
        room = find_room_by_id(room_id)
        if not room:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Room not found"
            }))
            return
        
        # Create participant and add to room
        participant = Participant(user_id, username, role, websocket)
        room.add_participant(participant)
        active_connections[user_id] = websocket
        
        # Mark user as online
        users_db[user_id]["is_online"] = True
        
        # 🎯 AUTOMATIC ATTENDANCE TRACKING - Record attendance as soon as user joins
        attendance_recorded = record_attendance(room.room_id, user_id, username, role)
        
        # Send current room state to new participant
        await websocket.send_text(json.dumps({
            "type": "room_joined",
            "room_id": room_id,
            "participants": room.get_participant_list(),
            "is_host": user_id == room.host_user_id,
            "message": f"Welcome to {room.room_name}!",
            "attendance_recorded": attendance_recorded
        }))
        
        # Notify other participants
        await room.broadcast_to_all({
            "type": "participant_joined",
            "user_id": user_id,
            "username": username,
            "role": role,
            "participant_count": len(room.participants),
            "timestamp": datetime.now().isoformat()
        }, exclude_user_id=user_id)
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await handle_websocket_message(room, user_id, message)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
    finally:
        # Cleanup on disconnect
        if user_id and room:
            room.remove_participant(user_id)
            if user_id in active_connections:
                del active_connections[user_id]
            
            # Mark user as offline if no other connections
            if user_id in users_db:
                users_db[user_id]["is_online"] = False
            
            # Notify other participants
            await room.broadcast_to_all({
                "type": "participant_left",
                "user_id": user_id,
                "participant_count": len(room.participants),
                "timestamp": datetime.now().isoformat()
            })

async def handle_websocket_message(room: ClassroomRoom, user_id: str, message: dict):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "chat_message":
        # Broadcast chat message to all participants
        chat_data = {
            "type": "chat_message",
            "user_id": user_id,
            "username": users_db[user_id]["username"],
            "message": message.get("message", ""),
            "timestamp": datetime.now().isoformat()
        }
        await room.broadcast_to_all(chat_data)
        
    elif message_type == "video_toggle":
        # Update participant video status
        if user_id in room.participants:
            room.participants[user_id].video_enabled = message.get("enabled", True)
            await room.broadcast_to_all({
                "type": "participant_video_toggle",
                "user_id": user_id,
                "video_enabled": message.get("enabled", True),
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "audio_toggle":
        # Update participant audio status
        if user_id in room.participants:
            room.participants[user_id].audio_enabled = message.get("enabled", True)
            await room.broadcast_to_all({
                "type": "participant_audio_toggle",
                "user_id": user_id,
                "audio_enabled": message.get("enabled", True),
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "raise_hand":
        # Toggle hand raising
        if user_id in room.participants:
            room.participants[user_id].hand_raised = message.get("raised", False)
            await room.broadcast_to_all({
                "type": "hand_raised",
                "user_id": user_id,
                "username": users_db[user_id]["username"],
                "hand_raised": message.get("raised", False),
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "emotion_update":
        # 🎯 ONLY DETECT STUDENT EMOTIONS - Skip if user is instructor
        user_role = users_db[user_id].get("role", "student")
        
        if user_role == "instructor":
            # Instructors' emotions are not tracked
            return
        
        # Handle emotion recognition data (STUDENTS ONLY)
        emotion_data = {
            "type": "emotion_update",
            "user_id": user_id,
            "username": users_db[user_id]["username"],
            "role": user_role,
            "emotion": message.get("emotion", "neutral"),
            "confidence": message.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Update participant emotion data
        if user_id in room.participants:
            room.participants[user_id].emotion_data = {
                "primary_emotion": message.get("emotion", "neutral"),
                "confidence": message.get("confidence", 0.0),
                "updated_at": datetime.now().isoformat()
            }
        
        # Broadcast emotion update only to instructors/admins
        # Students don't see other students' emotions
        for participant_id, participant in room.participants.items():
            if users_db.get(participant_id, {}).get("role") == "instructor":
                try:
                    await participant.websocket.send_text(json.dumps(emotion_data))
                except Exception as e:
                    print(f"Failed to send emotion to instructor {participant_id}: {e}")
        
        # Store emotion data for analytics (simplified in-memory storage)
        if not hasattr(room, 'emotion_history'):
            room.emotion_history = []
        room.emotion_history.append(emotion_data)
    
    elif message_type == "screen_share_start":
        # Handle screen sharing start
        await room.broadcast_to_all({
            "type": "screen_share_started",
            "user_id": user_id,
            "username": users_db[user_id]["username"],
            "timestamp": datetime.now().isoformat()
        }, exclude_user_id=user_id)
    
    elif message_type == "screen_share_stop":
        # Handle screen sharing stop
        await room.broadcast_to_all({
            "type": "screen_share_stopped",
            "user_id": user_id,
            "username": users_db[user_id]["username"],
            "timestamp": datetime.now().isoformat()
        }, exclude_user_id=user_id)
    
    elif message_type == "webrtc_signal":
        # Forward WebRTC signaling data (for video/audio/screen sharing)
        target_user_id = message.get("target_user_id")
        if target_user_id in room.participants:
            await room.participants[target_user_id].websocket.send_text(json.dumps({
                "type": "webrtc_signal",
                "from_user_id": user_id,
                "signal_data": message.get("signal_data"),
                "signal_type": message.get("signal_type", "video"),  # video, audio, or screen
                "timestamp": datetime.now().isoformat()
            }))
    
    elif message_type == "request_emotion_summary" and room.is_host(user_id):
        # Send emotion analytics to host/admin
        if hasattr(room, 'emotion_history'):
            emotion_summary = analyze_emotions(room.emotion_history)
            await room.send_to_user(user_id, {
                "type": "emotion_analytics",
                "summary": emotion_summary,
                "timestamp": datetime.now().isoformat()
            })

def analyze_emotions(emotion_history):
    """Analyze emotion data for admin dashboard"""
    if not emotion_history:
        return {"message": "No emotion data available"}
    
    # Group emotions by user
    user_emotions = {}
    emotion_counts = {}

def record_attendance(room_id: str, user_id: str, username: str, role: str):
    """Automatically record attendance when a student joins the room"""
    if room_id not in attendance_db:
        attendance_db[room_id] = []
    
    # Check if user already marked present in this session
    already_recorded = any(
        record["user_id"] == user_id 
        for record in attendance_db[room_id]
    )
    
    if not already_recorded:
        attendance_record = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "joined_at": datetime.now().isoformat(),
            "status": "present",
            "timestamp": datetime.now().timestamp()
        }
        attendance_db[room_id].append(attendance_record)
        print(f"✅ Attendance recorded: {username} ({role}) joined room {room_id[:8]}")
        return True
    return False
    
    for entry in emotion_history[-50:]:  # Last 50 entries
        user_id = entry["user_id"]
        emotion = entry["emotion"]
        
        if user_id not in user_emotions:
            user_emotions[user_id] = []
        user_emotions[user_id].append(emotion)
        
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Calculate overall mood
    total_emotions = sum(emotion_counts.values())
    mood_percentages = {
        emotion: (count / total_emotions) * 100 
        for emotion, count in emotion_counts.items()
    }
    
    # Identify students needing attention
    attention_needed = []
    for user_id, emotions in user_emotions.items():
        recent_emotions = emotions[-5:]  # Last 5 emotions
        negative_count = sum(1 for e in recent_emotions if e in ['confused', 'frustrated', 'bored'])
        if negative_count >= 3:
            username = users_db.get(user_id, {}).get("username", "Unknown")
            attention_needed.append({
                "user_id": user_id,
                "username": username,
                "dominant_emotion": max(set(recent_emotions), key=recent_emotions.count)
            })
    
    return {
        "overall_mood": mood_percentages,
        "attention_needed": attention_needed,
        "total_participants": len(user_emotions),
        "engagement_score": mood_percentages.get('engaged', 0) + mood_percentages.get('curious', 0)
    }

@app.get("/api/classroom/analytics/{room_id}")
async def get_classroom_analytics(room_id: str, user_data: dict):
    """Get emotion analytics for room (admin/instructor only)"""
    room = find_room_by_id(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    user_id = user_data.get("user_id")
    
    # Check if user is host/instructor
    if user_id != room.host_user_id:
        # Allow if user is instructor role
        if user_id not in users_db or users_db[user_id].get("role") != "instructor":
            raise HTTPException(status_code=403, detail="Only instructors can view analytics")
    
    # Get emotion analytics
    analytics = analyze_emotions(getattr(room, 'emotion_history', []))
    
    # Add current participant emotions
    current_emotions = {}
    for participant_id, participant in room.participants.items():
        if hasattr(participant, 'emotion_data') and participant.emotion_data:
            current_emotions[participant_id] = {
                "username": users_db.get(participant_id, {}).get("username", "Unknown"),
                "current_emotion": participant.emotion_data.get("primary_emotion", "neutral"),
                "confidence": participant.emotion_data.get("confidence", 0.0),
                "last_updated": participant.emotion_data.get("updated_at")
            }
    
    return {
        "room_id": room.room_id,
        "analytics": analytics,
        "current_emotions": current_emotions,
        "participant_count": len(room.participants),
        "timestamp": datetime.now().isoformat()
    }

# Serve the classroom join page
@app.get("/classroom", response_class=HTMLResponse)
async def classroom_join_page():
    """Serve the classroom join interface"""
    try:
        with open("classroom_join.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Classroom join page not found</h1>"

@app.get("/classroom/{room_id}", response_class=HTMLResponse)
async def classroom_with_room(room_id: str):
    """Serve the classroom join interface with pre-filled room ID"""
    try:
        with open("classroom_join.html", "r") as f:
            content = f.read()
            # Pre-fill the room ID in the form
            content = content.replace(
                'id="roomId" placeholder="Leave empty to create new room"',
                f'id="roomId" placeholder="Leave empty to create new room" value="{room_id}"'
            )
            return content
    except FileNotFoundError:
        return "<h1>Classroom join page not found</h1>"

# Demo data endpoint (for quick testing)
@app.get("/api/demo/classroom")
async def demo_classroom():
    """Demo classroom endpoint with real user data"""
    # Create some demo users if none exist
    if not users_db:
        demo_users = [
            {"username": "Dr. Smith", "email": "teacher@demo.com", "role": "instructor"},
            {"username": "Alice", "email": "alice@demo.com", "role": "student"},
            {"username": "Bob", "email": "bob@demo.com", "role": "student"}
        ]
        
        for user_data in demo_users:
            user_id = str(uuid.uuid4())
            users_db[user_id] = {
                "user_id": user_id,
                "username": user_data["username"],
                "email": user_data["email"],
                "role": user_data["role"],
                "created_at": datetime.now().isoformat(),
                "is_online": False
            }
    
    return {
        "message": "Demo data loaded",
        "total_users": len(users_db),
        "active_rooms": len(rooms_db),
        "users": list(users_db.values()),
        "demo_features": [
            "Real user registration",
            "Live video conferencing",
            "WebRTC peer-to-peer connections",
            "Real-time chat",
            "Hand raising functionality",
            "Instructor controls"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Emotion-Aware Virtual Classroom Backend (Real Multi-User)")
    print("📊 Environment: Development")
    print("🔗 Server will be available at: http://localhost:8001")
    print("📚 API Documentation: http://localhost:8001/docs")
    print("👥 Features: Real user registration, WebRTC video, live chat")
    print("🎥 WebSocket: Real-time classroom communication")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )