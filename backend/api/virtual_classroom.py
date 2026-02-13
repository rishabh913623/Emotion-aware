"""
Virtual Classroom with WebRTC Video Conferencing
Zoom-like meeting functionality integrated with emotion recognition
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional
import json
import uuid
from datetime import datetime
import logging
from collections import defaultdict
from backend.database.models import User
from backend.api.auth import get_current_user

router = APIRouter(prefix="/api/classroom", tags=["Virtual Classroom"])
logger = logging.getLogger(__name__)

class WebRTCRoom:
    def __init__(self, room_id: str, host_id: str):
        self.room_id = room_id
        self.host_id = host_id
        self.participants: Dict[str, dict] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.created_at = datetime.now()
        self.is_recording = False
        self.chat_messages = []
        self.current_topic = None
        self.topics_history = []
        self.emotion_data = defaultdict(list)  # user_id -> list of emotions
        self.generated_materials = {}  # topic -> material
        self.generated_quizzes = {}  # topic -> quiz
        
    def add_participant(self, user_id: str, websocket: WebSocket, user_info: dict):
        """Add participant to the room"""
        self.participants[user_id] = {
            "user_id": user_id,
            "username": user_info.get("username", "Unknown"),
            "role": user_info.get("role", "student"),
            "joined_at": datetime.now().isoformat(),
            "video_enabled": True,
            "audio_enabled": True,
            "screen_sharing": False,
            "hand_raised": False,
            "emotion_data": None
        }
        self.connections[user_id] = websocket
        
    def remove_participant(self, user_id: str):
        """Remove participant from the room"""
        if user_id in self.participants:
            del self.participants[user_id]
        if user_id in self.connections:
            del self.connections[user_id]
    
    def get_participant_count(self) -> int:
        return len(self.participants)
    
    def is_host(self, user_id: str) -> bool:
        return user_id == self.host_id
    
    async def broadcast_to_all(self, message: dict, exclude_user: str = None):
        """Broadcast message to all participants"""
        for user_id, websocket in self.connections.items():
            if user_id != exclude_user:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send message to {user_id}: {e}")
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user"""
        if user_id in self.connections:
            try:
                await self.connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
    
    def add_emotion_data(self, user_id: str, emotion_data: dict):
        """Store emotion data for a student"""
        self.emotion_data[user_id].append({
            "emotion": emotion_data.get("primary_emotion"),
            "confidence": emotion_data.get("confidence"),
            "timestamp": datetime.now().isoformat(),
            "topic": self.current_topic
        })
    
    def get_emotion_summary(self) -> dict:
        """Calculate emotion summary for all students"""
        if not self.emotion_data:
            return {
                "engaged_percent": 0,
                "confused_percent": 0,
                "bored_percent": 0,
                "neutral_percent": 0,
                "total_samples": 0,
                "suggestion": "No emotion data available yet."
            }
        
        total_count = 0
        emotion_counts = {"engaged": 0, "confused": 0, "bored": 0, "neutral": 0}
        
        for user_id, emotions in self.emotion_data.items():
            # Only count students, not instructor
            if user_id != self.host_id:
                for emotion_entry in emotions:
                    emotion = emotion_entry.get("emotion", "neutral")
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
                    total_count += 1
        
        if total_count == 0:
            return {
                "engaged_percent": 0,
                "confused_percent": 0,
                "bored_percent": 0,
                "neutral_percent": 0,
                "total_samples": 0,
                "suggestion": "No emotion data from students yet."
            }
        
        engaged_pct = (emotion_counts["engaged"] / total_count) * 100
        confused_pct = (emotion_counts["confused"] / total_count) * 100
        bored_pct = (emotion_counts["bored"] / total_count) * 100
        neutral_pct = (emotion_counts["neutral"] / total_count) * 100
        
        # Generate suggestion based on dominant emotion
        suggestion = ""
        if confused_pct > 40:
            suggestion = "High confusion detected. Consider slowing down and reviewing key concepts."
        elif bored_pct > 40:
            suggestion = "Students appear bored. Try adding interactive activities or changing pace."
        elif engaged_pct > 60:
            suggestion = "Great! Students are highly engaged. Keep up the current approach."
        else:
            suggestion = "Mixed emotions. Monitor individual students and adjust as needed."
        
        return {
            "engaged_percent": round(engaged_pct, 1),
            "confused_percent": round(confused_pct, 1),
            "bored_percent": round(bored_pct, 1),
            "neutral_percent": round(neutral_pct, 1),
            "total_samples": total_count,
            "suggestion": suggestion
        }

# Room manager
class RoomManager:
    def __init__(self):
        self.active_rooms: Dict[str, WebRTCRoom] = {}
    
    def create_room(self, host_id: str, room_name: str = None) -> str:
        """Create a new room"""
        room_id = str(uuid.uuid4())
        self.active_rooms[room_id] = WebRTCRoom(room_id, host_id)
        return room_id
    
    def get_room(self, room_id: str) -> Optional[WebRTCRoom]:
        """Get room by ID"""
        return self.active_rooms.get(room_id)
    
    def delete_room(self, room_id: str):
        """Delete room"""
        if room_id in self.active_rooms:
            del self.active_rooms[room_id]
    
    def get_user_rooms(self, user_id: str) -> List[str]:
        """Get rooms where user is a participant"""
        rooms = []
        for room_id, room in self.active_rooms.items():
            if user_id in room.participants:
                rooms.append(room_id)
        return rooms

room_manager = RoomManager()

@router.post("/create-room")
async def create_room(current_user: User = Depends(get_current_user)):
    """Create a new virtual classroom room"""
    room_id = room_manager.create_room(str(current_user.id))
    
    return {
        "room_id": room_id,
        "host_id": str(current_user.id),
        "join_url": f"/classroom/join/{room_id}",
        "created_at": datetime.now().isoformat()
    }

@router.get("/rooms")
async def list_rooms(current_user: User = Depends(get_current_user)):
    """List available rooms"""
    rooms = []
    for room_id, room in room_manager.active_rooms.items():
        # Only return rooms where user is host or participant
        if room.is_host(str(current_user.id)) or str(current_user.id) in room.participants:
            rooms.append({
                "room_id": room_id,
                "host_id": room.host_id,
                "participant_count": room.get_participant_count(),
                "created_at": room.created_at.isoformat(),
                "is_host": room.is_host(str(current_user.id))
            })
    
    return {"rooms": rooms}

@router.get("/room/{room_id}")
async def get_room_info(room_id: str, current_user: User = Depends(get_current_user)):
    """Get room information"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {
        "room_id": room.room_id,
        "host_id": room.host_id,
        "participants": list(room.participants.values()),
        "created_at": room.created_at.isoformat(),
        "is_host": room.is_host(str(current_user.id)),
        "chat_messages": room.chat_messages[-50:]  # Last 50 messages
    }

@router.delete("/room/{room_id}")
async def delete_room(room_id: str, current_user: User = Depends(get_current_user)):
    """Delete room (host only)"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    if not room.is_host(str(current_user.id)):
        raise HTTPException(status_code=403, detail="Only host can delete room")
    
    # Notify all participants
    await room.broadcast_to_all({
        "type": "room_closed",
        "message": "Room has been closed by the host"
    })
    
    room_manager.delete_room(room_id)
    return {"message": "Room deleted successfully"}

@router.post("/room/{room_id}/set-topic")
async def set_topic(room_id: str, topic_data: dict):
    """Set the current topic for the room"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    topic_name = topic_data.get("topic")
    if not topic_name:
        raise HTTPException(status_code=400, detail="Topic name required")
    
    room.current_topic = topic_name
    room.topics_history.append({
        "topic": topic_name,
        "set_at": datetime.now().isoformat()
    })
    
    # Broadcast topic to all participants
    await room.broadcast_to_all({
        "type": "topic_update",
        "topic": topic_name
    })
    
    return {"message": "Topic set successfully", "topic": topic_name}

@router.post("/room/{room_id}/generate-material")
async def generate_material(room_id: str, material_request: dict):
    """Generate teaching material for a topic"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    topic = material_request.get("topic") or room.current_topic
    if not topic:
        raise HTTPException(status_code=400, detail="No topic specified")
    
    # Generate structured teaching material
    material = generate_teaching_material(topic)
    room.generated_materials[topic] = material
    
    return {
        "topic": topic,
        "material": material,
        "generated_at": datetime.now().isoformat()
    }

@router.post("/room/{room_id}/generate-quiz")
async def generate_quiz(room_id: str, quiz_request: dict):
    """Generate quiz for a topic"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    topic = quiz_request.get("topic") or room.current_topic
    if not topic:
        raise HTTPException(status_code=400, detail="No topic specified")
    
    # Generate quiz questions
    quiz = generate_quiz_questions(topic)
    room.generated_quizzes[topic] = quiz
    
    return {
        "topic": topic,
        "quiz": quiz,
        "generated_at": datetime.now().isoformat()
    }

@router.get("/room/{room_id}/emotion-summary")
async def get_emotion_summary(room_id: str):
    """Get emotion summary for the room"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    summary = room.get_emotion_summary()
    return summary

@router.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for room communication"""
    await websocket.accept()
    
    try:
        # Get user info from connection (simplified - in production use proper auth)
        user_data = await websocket.receive_text()
        user_info = json.loads(user_data)
        user_id = user_info.get("user_id")
        
        if not user_id:
            await websocket.close(code=4000, reason="User ID required")
            return
        
        # Get or create room
        room = room_manager.get_room(room_id)
        if not room:
            await websocket.close(code=4004, reason="Room not found")
            return
        
        # Add participant to room
        room.add_participant(user_id, websocket, user_info)
        
        # Send current room state to new participant
        await room.send_to_user(user_id, {
            "type": "room_state",
            "participants": list(room.participants.values()),
            "room_info": {
                "room_id": room.room_id,
                "host_id": room.host_id,
                "is_host": room.is_host(user_id),
                "current_topic": room.current_topic
            }
        })
        
        # Notify others about new participant
        await room.broadcast_to_all({
            "type": "participant_joined",
            "user_id": user_id,
            "username": user_info.get("username", "Unknown"),
            "role": user_info.get("role", "student"),
            "participant_count": room.get_participant_count()
        }, exclude_user=user_id)
        
        # Send current room state to new participant
        await room.send_to_user(user_id, {
            "type": "room_state",
            "participants": list(room.participants.values()),
            "room_info": {
                "room_id": room.room_id,
                "host_id": room.host_id,
                "is_host": room.is_host(user_id)
            }
        })
        
        # Handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_room_message(room, user_id, message)
            
    except WebSocketDisconnect:
        # Remove participant when they disconnect
        if 'room' in locals() and 'user_id' in locals():
            room.remove_participant(user_id)
            await room.broadcast_to_all({
                "type": "participant_left",
                "user_id": user_id,
                "participant_count": room.get_participant_count()
            })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=4000, reason="Internal error")

async def handle_room_message(room: WebRTCRoom, user_id: str, message: dict):
    """Handle different types of room messages"""
    message_type = message.get("type")
    
    if message_type == "webrtc_offer":
        # CRITICAL: Forward WebRTC offer to target peer
        target_user = message.get("target_user")
        if target_user and target_user in room.connections:
            await room.send_to_user(target_user, {
                "type": "webrtc_offer",
                "from_user": user_id,
                "offer": message.get("offer")
            })
    
    elif message_type == "webrtc_answer":
        # CRITICAL: Forward WebRTC answer to target peer
        target_user = message.get("target_user")
        if target_user and target_user in room.connections:
            await room.send_to_user(target_user, {
                "type": "webrtc_answer",
                "from_user": user_id,
                "answer": message.get("answer")
            })
    
    elif message_type == "webrtc_ice_candidate":
        # CRITICAL: Forward ICE candidate to target peer
        target_user = message.get("target_user")
        if target_user and target_user in room.connections:
            await room.send_to_user(target_user, {
                "type": "webrtc_ice_candidate",
                "from_user": user_id,
                "candidate": message.get("candidate")
            })
    
    elif message_type == "chat_message":
        # Broadcast chat message to all participants
        chat_message = {
            "type": "chat_message",
            "user_id": user_id,
            "username": room.participants[user_id]["username"],
            "message": message.get("message", ""),
            "timestamp": datetime.now().isoformat()
        }
        room.chat_messages.append(chat_message)
        await room.broadcast_to_all(chat_message)
    
    elif message_type == "media_state_change":
        # Update participant media state (video/audio on/off)
        if user_id in room.participants:
            participant = room.participants[user_id]
            participant["video_enabled"] = message.get("video_enabled", participant["video_enabled"])
            participant["audio_enabled"] = message.get("audio_enabled", participant["audio_enabled"])
            participant["screen_sharing"] = message.get("screen_sharing", participant["screen_sharing"])
            
            # Broadcast state change
            await room.broadcast_to_all({
                "type": "participant_media_change",
                "user_id": user_id,
                "video_enabled": participant["video_enabled"],
                "audio_enabled": participant["audio_enabled"],
                "screen_sharing": participant["screen_sharing"]
            }, exclude_user=user_id)
    
    elif message_type == "raise_hand":
        # Toggle hand raised state
        if user_id in room.participants:
            participant = room.participants[user_id]
            participant["hand_raised"] = not participant["hand_raised"]
            
            await room.broadcast_to_all({
                "type": "hand_raised",
                "user_id": user_id,
                "username": participant["username"],
                "hand_raised": participant["hand_raised"]
            })
    
    elif message_type == "emotion_update":
        # Update participant emotion data (students only)
        if user_id in room.participants and not room.is_host(user_id):
            emotion_data = message.get("emotion_data")
            room.add_emotion_data(user_id, emotion_data)
            
            # Send emotion update to host for real-time monitoring
            await room.send_to_user(room.host_id, {
                "type": "student_emotion_update",
                "user_id": user_id,
                "emotion_data": emotion_data,
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "request_emotion_summary":
        # Instructor requests emotion summary
        if room.is_host(user_id):
            summary = room.get_emotion_summary()
            await room.send_to_user(user_id, {
                "type": "emotion_summary",
                "summary": summary
            })
    
    elif message_type == "host_control":
        # Handle host control actions (mute participant, etc.)
        if room.is_host(user_id):
            action = message.get("action")
            target_user = message.get("target_user")
            
            if action and target_user and target_user in room.participants:
                await room.send_to_user(target_user, {
                    "type": "host_control",
                    "action": action,
                    "from_host": True
                })
    
    else:
        # Forward unknown messages as-is (for future extensibility)
        await room.broadcast_to_all(message, exclude_user=user_id)

def generate_teaching_material(topic: str) -> dict:
    """Generate structured teaching material for a topic"""
    # In production, this would use an LLM or curriculum database
    # For now, generate structured placeholder content
    
    material = {
        "title": topic,
        "sections": [
            {
                "heading": "Introduction",
                "content": f"This section introduces the fundamental concepts of {topic}. "
                          f"Understanding {topic} is essential for building a strong foundation in this subject area."
            },
            {
                "heading": "Key Concepts",
                "content": f"The main ideas in {topic} include:\n"
                          f"• Concept 1: Foundational principles and definitions\n"
                          f"• Concept 2: Practical applications and real-world examples\n"
                          f"• Concept 3: Advanced considerations and edge cases\n"
                          f"• Concept 4: Best practices and common patterns"
            },
            {
                "heading": "Practical Examples",
                "content": f"Let's explore {topic} through concrete examples:\n\n"
                          f"Example 1: A basic implementation demonstrating core principles.\n"
                          f"Example 2: An intermediate scenario showing practical application.\n"
                          f"Example 3: An advanced use case with complex interactions."
            },
            {
                "heading": "Common Pitfalls",
                "content": f"When learning {topic}, students often encounter these challenges:\n"
                          f"• Misunderstanding the scope and limitations\n"
                          f"• Incorrect application in edge cases\n"
                          f"• Performance considerations\n"
                          f"• Integration with other concepts"
            },
            {
                "heading": "Summary and Next Steps",
                "content": f"Key takeaways from {topic}:\n"
                          f"1. Master the fundamental concepts before moving to advanced topics\n"
                          f"2. Practice with real-world examples to reinforce learning\n"
                          f"3. Review common pitfalls to avoid mistakes\n"
                          f"4. Apply knowledge through hands-on projects"
            }
        ],
        "resources": [
            {"type": "reading", "title": f"Comprehensive guide to {topic}"},
            {"type": "video", "title": f"Video tutorial series: {topic}"},
            {"type": "practice", "title": f"Practice exercises and challenges for {topic}"},
            {"type": "reference", "title": f"Quick reference sheet for {topic}"}
        ]
    }
    
    return material

def generate_quiz_questions(topic: str) -> dict:
    """Generate quiz questions for a topic"""
    # In production, this would use an LLM or question bank
    # For now, generate structured MCQ format
    
    quiz = {
        "topic": topic,
        "questions": [
            {
                "id": 1,
                "question": f"What is the primary concept behind {topic}?",
                "options": [
                    {"id": "a", "text": "The foundational principle that defines the core of the subject"},
                    {"id": "b", "text": "A secondary concept that builds upon other topics"},
                    {"id": "c", "text": "An advanced application used in specialized scenarios"},
                    {"id": "d", "text": "An unrelated concept from a different domain"}
                ],
                "correct_answer": "a",
                "explanation": f"The fundamental principle is the core of {topic} and must be understood before progressing to advanced concepts."
            },
            {
                "id": 2,
                "question": f"Which of the following is a practical application of {topic}?",
                "options": [
                    {"id": "a", "text": "Solving real-world problems in industry settings"},
                    {"id": "b", "text": "Building scalable systems and architectures"},
                    {"id": "c", "text": "Optimizing performance in critical applications"},
                    {"id": "d", "text": "All of the above"}
                ],
                "correct_answer": "d",
                "explanation": f"{topic} has widespread applications across multiple domains including industry, systems design, and performance optimization."
            },
            {
                "id": 3,
                "question": f"What is a common misconception about {topic}?",
                "options": [
                    {"id": "a", "text": "That it can be applied universally without considering context"},
                    {"id": "b", "text": "That it requires no foundational knowledge to understand"},
                    {"id": "c", "text": "That it has no limitations or edge cases"},
                    {"id": "d", "text": "That it is independent of other concepts in the field"}
                ],
                "correct_answer": "a",
                "explanation": f"While {topic} is powerful, it must be applied thoughtfully with consideration for specific contexts and requirements."
            },
            {
                "id": 4,
                "question": f"When implementing {topic}, what is the most important consideration?",
                "options": [
                    {"id": "a", "text": "Understanding the requirements and constraints"},
                    {"id": "b", "text": "Choosing the simplest possible approach"},
                    {"id": "c", "text": "Following best practices and established patterns"},
                    {"id": "d", "text": "Both A and C"}
                ],
                "correct_answer": "d",
                "explanation": "Successful implementation requires understanding requirements while following established best practices."
            },
            {
                "id": 5,
                "question": f"How does {topic} relate to other concepts in the field?",
                "options": [
                    {"id": "a", "text": "It builds upon foundational concepts and enables advanced techniques"},
                    {"id": "b", "text": "It operates completely independently"},
                    {"id": "c", "text": "It replaces all previous approaches"},
                    {"id": "d", "text": "It is incompatible with other methodologies"}
                ],
                "correct_answer": "a",
                "explanation": f"{topic} is part of a broader ecosystem of concepts, building on fundamentals while enabling more advanced work."
            }
        ],
        "passing_score": 70,
        "time_limit_minutes": 15,
        "instructions": f"This quiz tests your understanding of {topic}. Read each question carefully and select the best answer."
    }
    
    return quiz

@router.get("/join/{room_id}")
async def join_room_page(room_id: str):
    """Serve the enhanced virtual classroom interface with WebRTC and instructor features"""
    import os
    
    # Read the HTML template from file
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "classroom_interface.html")
    
    try:
        with open(template_path, "r") as f:
            html_template = f.read()
        
        # Replace room_id placeholder
        html_content = html_template.replace("{{ROOM_ID}}", room_id)
        
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Classroom interface template not found")