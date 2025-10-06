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
        
        # Notify others about new participant
        await room.broadcast_to_all({
            "type": "participant_joined",
            "user_id": user_id,
            "username": user_info.get("username", "Unknown"),
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
        # Forward WebRTC offer to target peer
        target_user = message.get("target_user")
        if target_user:
            await room.send_to_user(target_user, {
                "type": "webrtc_offer",
                "from_user": user_id,
                "offer": message.get("offer"),
                "media_constraints": message.get("media_constraints", {})
            })
    
    elif message_type == "webrtc_answer":
        # Forward WebRTC answer to target peer
        target_user = message.get("target_user")
        if target_user:
            await room.send_to_user(target_user, {
                "type": "webrtc_answer",
                "from_user": user_id,
                "answer": message.get("answer")
            })
    
    elif message_type == "webrtc_ice_candidate":
        # Forward ICE candidate to target peer
        target_user = message.get("target_user")
        if target_user:
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
        # Update participant emotion data for instructor dashboard
        if user_id in room.participants:
            room.participants[user_id]["emotion_data"] = message.get("emotion_data")
            
            # Send emotion update to host for dashboard
            await room.send_to_user(room.host_id, {
                "type": "student_emotion_update",
                "user_id": user_id,
                "emotion_data": message.get("emotion_data"),
                "timestamp": datetime.now().isoformat()
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

@router.get("/join/{room_id}")
async def join_room_page(room_id: str):
    """Serve the virtual classroom interface"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Virtual Classroom - {room_id}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5;
            }}
            .classroom-container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .video-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                padding: 20px;
                background: #2c3e50;
            }}
            .video-container {{ 
                position: relative;
                background: #34495e;
                border-radius: 8px;
                overflow: hidden;
                aspect-ratio: 16/9;
            }}
            .video-container video {{ 
                width: 100%; 
                height: 100%; 
                object-fit: cover;
            }}
            .participant-info {{
                position: absolute;
                bottom: 5px;
                left: 5px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
            }}
            .controls {{ 
                padding: 15px 20px;
                background: #ecf0f1;
                display: flex;
                gap: 10px;
                align-items: center;
                justify-content: center;
            }}
            .control-btn {{ 
                padding: 10px 15px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s;
            }}
            .control-btn.active {{ background: #3498db; color: white; }}
            .control-btn.inactive {{ background: #e74c3c; color: white; }}
            .control-btn.neutral {{ background: #95a5a6; color: white; }}
            .chat-container {{
                position: fixed;
                right: 20px;
                top: 20px;
                width: 300px;
                height: 400px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                display: flex;
                flex-direction: column;
            }}
            .chat-messages {{
                flex: 1;
                overflow-y: auto;
                padding: 10px;
                border-bottom: 1px solid #eee;
            }}
            .chat-input {{
                display: flex;
                padding: 10px;
            }}
            .chat-input input {{
                flex: 1;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .chat-input button {{
                margin-left: 5px;
                padding: 8px 12px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            .emotion-indicator {{
                position: absolute;
                top: 5px;
                right: 5px;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                border: 2px solid white;
            }}
            .emotion-engaged {{ background: #2ecc71; }}
            .emotion-confused {{ background: #f39c12; }}
            .emotion-bored {{ background: #e74c3c; }}
            .emotion-neutral {{ background: #95a5a6; }}
        </style>
    </head>
    <body>
        <div class="classroom-container">
            <div class="video-grid" id="videoGrid">
                <!-- Videos will be added dynamically -->
            </div>
            
            <div class="controls">
                <button class="control-btn neutral" id="videoBtn" onclick="toggleVideo()">📹 Video</button>
                <button class="control-btn neutral" id="audioBtn" onclick="toggleAudio()">🎤 Audio</button>
                <button class="control-btn neutral" id="shareBtn" onclick="shareScreen()">📺 Share</button>
                <button class="control-btn neutral" id="handBtn" onclick="raiseHand()">✋ Hand</button>
                <button class="control-btn neutral" onclick="leaveRoom()">📞 Leave</button>
            </div>
        </div>
        
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <!-- Chat messages will appear here -->
            </div>
            <div class="chat-input">
                <input type="text" id="chatInput" placeholder="Type a message...">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            // Virtual Classroom WebRTC Implementation
            const roomId = "{room_id}";
            let localStream = null;
            let peerConnections = {{}};
            let websocket = null;
            let localVideo = null;
            let isVideoEnabled = true;
            let isAudioEnabled = true;
            let isScreenSharing = false;
            let handRaised = false;
            
            const configuration = {{
                iceServers: [
                    {{ urls: 'stun:stun.l.google.com:19302' }},
                    {{ urls: 'stun:stun1.l.google.com:19302' }}
                ]
            }};

            async function initializeClassroom() {{
                try {{
                    // Get user media
                    localStream = await navigator.mediaDevices.getUserMedia({{
                        video: true,
                        audio: true
                    }});
                    
                    // Create local video element
                    createVideoElement('local', localStream, 'You', true);
                    
                    // Connect to WebSocket
                    connectWebSocket();
                    
                }} catch (error) {{
                    console.error('Error initializing classroom:', error);
                    alert('Error accessing camera/microphone. Please check permissions.');
                }}
            }}
            
            function connectWebSocket() {{
                const wsUrl = `ws://localhost:8000/api/classroom/ws/{room_id}`;
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function() {{
                    console.log('Connected to classroom');
                    // Send user info
                    websocket.send(JSON.stringify({{
                        user_id: 'user_' + Math.random().toString(36).substr(2, 9),
                        username: prompt('Enter your name:') || 'Anonymous',
                        role: 'student'
                    }}));
                }};
                
                websocket.onmessage = async function(event) {{
                    const message = JSON.parse(event.data);
                    await handleWebSocketMessage(message);
                }};
                
                websocket.onclose = function() {{
                    console.log('Disconnected from classroom');
                }};
            }}
            
            async function handleWebSocketMessage(message) {{
                switch (message.type) {{
                    case 'participant_joined':
                        console.log('Participant joined:', message.username);
                        addChatMessage('System', `${{message.username}} joined the room`);
                        break;
                        
                    case 'participant_left':
                        console.log('Participant left:', message.user_id);
                        removeVideoElement(message.user_id);
                        break;
                        
                    case 'webrtc_offer':
                        await handleOffer(message);
                        break;
                        
                    case 'webrtc_answer':
                        await handleAnswer(message);
                        break;
                        
                    case 'webrtc_ice_candidate':
                        await handleIceCandidate(message);
                        break;
                        
                    case 'chat_message':
                        addChatMessage(message.username, message.message);
                        break;
                        
                    case 'participant_media_change':
                        updateParticipantMedia(message);
                        break;
                        
                    case 'room_state':
                        console.log('Room state received:', message.participants);
                        break;
                }}
            }}
            
            async function createPeerConnection(userId) {{
                const pc = new RTCPeerConnection(configuration);
                peerConnections[userId] = pc;
                
                // Add local stream to connection
                if (localStream) {{
                    localStream.getTracks().forEach(track => {{
                        pc.addTrack(track, localStream);
                    }});
                }}
                
                // Handle remote stream
                pc.ontrack = function(event) {{
                    createVideoElement(userId, event.streams[0], userId);
                }};
                
                // Handle ICE candidates
                pc.onicecandidate = function(event) {{
                    if (event.candidate) {{
                        websocket.send(JSON.stringify({{
                            type: 'webrtc_ice_candidate',
                            target_user: userId,
                            candidate: event.candidate
                        }}));
                    }}
                }};
                
                return pc;
            }}
            
            function createVideoElement(userId, stream, username, isLocal = false) {{
                const videoGrid = document.getElementById('videoGrid');
                
                // Remove existing video if it exists
                removeVideoElement(userId);
                
                const container = document.createElement('div');
                container.className = 'video-container';
                container.id = `video-${{userId}}`;
                
                const video = document.createElement('video');
                video.srcObject = stream;
                video.autoplay = true;
                video.playsInline = true;
                if (isLocal) {{
                    video.muted = true; // Mute local video to prevent feedback
                    localVideo = video;
                }}
                
                const info = document.createElement('div');
                info.className = 'participant-info';
                info.textContent = username;
                
                const emotionIndicator = document.createElement('div');
                emotionIndicator.className = 'emotion-indicator emotion-neutral';
                
                container.appendChild(video);
                container.appendChild(info);
                container.appendChild(emotionIndicator);
                videoGrid.appendChild(container);
            }}
            
            function removeVideoElement(userId) {{
                const element = document.getElementById(`video-${{userId}}`);
                if (element) {{
                    element.remove();
                }}
                
                // Close peer connection
                if (peerConnections[userId]) {{
                    peerConnections[userId].close();
                    delete peerConnections[userId];
                }}
            }}
            
            function toggleVideo() {{
                if (localStream) {{
                    const videoTrack = localStream.getVideoTracks()[0];
                    if (videoTrack) {{
                        videoTrack.enabled = !videoTrack.enabled;
                        isVideoEnabled = videoTrack.enabled;
                        
                        const btn = document.getElementById('videoBtn');
                        btn.className = `control-btn ${{isVideoEnabled ? 'active' : 'inactive'}}`;
                        btn.textContent = `📹 ${{isVideoEnabled ? 'Video' : 'Video Off'}}`;
                        
                        // Notify others
                        websocket.send(JSON.stringify({{
                            type: 'media_state_change',
                            video_enabled: isVideoEnabled
                        }}));
                    }}
                }}
            }}
            
            function toggleAudio() {{
                if (localStream) {{
                    const audioTrack = localStream.getAudioTracks()[0];
                    if (audioTrack) {{
                        audioTrack.enabled = !audioTrack.enabled;
                        isAudioEnabled = audioTrack.enabled;
                        
                        const btn = document.getElementById('audioBtn');
                        btn.className = `control-btn ${{isAudioEnabled ? 'active' : 'inactive'}}`;
                        btn.textContent = `🎤 ${{isAudioEnabled ? 'Audio' : 'Muted'}}`;
                        
                        // Notify others
                        websocket.send(JSON.stringify({{
                            type: 'media_state_change',
                            audio_enabled: isAudioEnabled
                        }}));
                    }}
                }}
            }}
            
            function raiseHand() {{
                handRaised = !handRaised;
                const btn = document.getElementById('handBtn');
                btn.className = `control-btn ${{handRaised ? 'active' : 'neutral'}}`;
                btn.textContent = `✋ ${{handRaised ? 'Hand Up' : 'Hand'}}`;
                
                websocket.send(JSON.stringify({{
                    type: 'raise_hand'
                }}));
            }}
            
            function sendMessage() {{
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                if (message && websocket) {{
                    websocket.send(JSON.stringify({{
                        type: 'chat_message',
                        message: message
                    }}));
                    input.value = '';
                }}
            }}
            
            function addChatMessage(username, message) {{
                const chatMessages = document.getElementById('chatMessages');
                const messageElement = document.createElement('div');
                messageElement.innerHTML = `<strong>${{username}}:</strong> ${{message}}`;
                messageElement.style.marginBottom = '5px';
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }}
            
            function leaveRoom() {{
                if (confirm('Are you sure you want to leave the room?')) {{
                    if (localStream) {{
                        localStream.getTracks().forEach(track => track.stop());
                    }}
                    if (websocket) {{
                        websocket.close();
                    }}
                    window.close();
                }}
            }}
            
            // Handle chat input on Enter key
            document.getElementById('chatInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    sendMessage();
                }}
            }});
            
            // Initialize classroom on page load
            initializeClassroom();
            
            // Simulate emotion detection (integration with existing emotion modules)
            setInterval(() => {{
                // This would integrate with the actual emotion detection
                const emotions = ['engaged', 'confused', 'bored', 'neutral'];
                const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
                
                // Update local emotion indicator
                const indicator = document.querySelector('#video-local .emotion-indicator');
                if (indicator) {{
                    indicator.className = `emotion-indicator emotion-${{randomEmotion}}`;
                }}
                
                // Send emotion update
                if (websocket) {{
                    websocket.send(JSON.stringify({{
                        type: 'emotion_update',
                        emotion_data: {{
                            primary_emotion: randomEmotion,
                            confidence: Math.random() * 0.4 + 0.6,
                            learning_state: randomEmotion
                        }}
                    }}));
                }}
            }}, 5000); // Update every 5 seconds
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)