"""
WebRTC Data Pipeline for Video/Audio/Chat Collection
Week 1: Real-time media capture with consent management
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import cv2
import numpy as np
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time data pipeline"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.classroom_participants: Dict[str, List[str]] = {}
        self.user_permissions: Dict[str, Dict[str, bool]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
        # Send connection confirmation
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
    
    def disconnect(self, client_id: str):
        """Remove client from active connections"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # Remove from classroom participants
        for classroom_id, participants in self.classroom_participants.items():
            if client_id in participants:
                participants.remove(client_id)
                
        logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_classroom(self, message: dict, classroom_id: str, exclude_client: str = None):
        """Broadcast message to all participants in a classroom"""
        if classroom_id in self.classroom_participants:
            participants = self.classroom_participants[classroom_id]
            for client_id in participants:
                if client_id != exclude_client:
                    await self.send_personal_message(message, client_id)
    
    async def join_classroom(self, client_id: str, classroom_id: str):
        """Add client to classroom"""
        if classroom_id not in self.classroom_participants:
            self.classroom_participants[classroom_id] = []
        
        if client_id not in self.classroom_participants[classroom_id]:
            self.classroom_participants[classroom_id].append(client_id)
            
        # Notify other participants
        await self.broadcast_to_classroom({
            "type": "participant_joined",
            "client_id": client_id,
            "classroom_id": classroom_id,
            "timestamp": datetime.utcnow().isoformat()
        }, classroom_id, exclude_client=client_id)
    
    async def leave_classroom(self, client_id: str, classroom_id: str):
        """Remove client from classroom"""
        if (classroom_id in self.classroom_participants and 
            client_id in self.classroom_participants[classroom_id]):
            self.classroom_participants[classroom_id].remove(client_id)
            
        # Notify other participants
        await self.broadcast_to_classroom({
            "type": "participant_left",
            "client_id": client_id,
            "classroom_id": classroom_id,
            "timestamp": datetime.utcnow().isoformat()
        }, classroom_id)

class MediaProcessor:
    """Processes incoming media data from WebRTC streams"""
    
    @staticmethod
    def process_video_frame(frame_data: str, client_id: str) -> Optional[dict]:
        """
        Process base64-encoded video frame
        Returns processed frame info for emotion recognition
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Basic frame validation
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                return None
            
            # Prepare for emotion recognition (Week 2 implementation)
            return {
                "client_id": client_id,
                "frame_shape": frame.shape,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "ready_for_processing"
            }
            
        except Exception as e:
            logger.error(f"Error processing video frame for {client_id}: {e}")
            return None
    
    @staticmethod
    def process_audio_chunk(audio_data: bytes, client_id: str) -> Optional[dict]:
        """
        Process audio chunk for emotion analysis
        Returns processed audio info
        """
        try:
            # Basic audio validation
            if len(audio_data) < 1024:  # Minimum chunk size
                return None
            
            # Prepare for audio emotion analysis (Week 3 implementation)
            return {
                "client_id": client_id,
                "audio_length": len(audio_data),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "ready_for_processing"
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for {client_id}: {e}")
            return None
    
    @staticmethod
    def process_text_message(text: str, client_id: str) -> Optional[dict]:
        """
        Process chat message for sentiment analysis
        Returns processed text info
        """
        try:
            # Basic text validation
            if not text or len(text.strip()) == 0:
                return None
            
            # Clean text
            cleaned_text = text.strip()[:500]  # Limit message length
            
            # Prepare for sentiment analysis (Week 4 implementation)
            return {
                "client_id": client_id,
                "text": cleaned_text,
                "text_length": len(cleaned_text),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "ready_for_processing"
            }
            
        except Exception as e:
            logger.error(f"Error processing text message for {client_id}: {e}")
            return None

class ConsentManager:
    """Manages student consent for ethical compliance"""
    
    def __init__(self):
        self.consent_cache: Dict[str, Dict[str, bool]] = {}
    
    def check_consent(self, user_id: str, data_type: str) -> bool:
        """
        Check if user has given consent for specific data type
        data_type: 'facial', 'audio', 'text', 'data_sharing'
        """
        if user_id not in self.consent_cache:
            return False
        
        return self.consent_cache[user_id].get(data_type, False)
    
    def update_consent(self, user_id: str, consent_data: dict):
        """Update user consent preferences"""
        if user_id not in self.consent_cache:
            self.consent_cache[user_id] = {}
        
        self.consent_cache[user_id].update(consent_data)
    
    def get_consent_status(self, user_id: str) -> dict:
        """Get complete consent status for user"""
        return self.consent_cache.get(user_id, {
            'facial': False,
            'audio': False,
            'text': False,
            'data_sharing': False
        })

class DataPipeline:
    """Main data pipeline coordinator"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.media_processor = MediaProcessor()
        self.consent_manager = ConsentManager()
        self.processing_queue = asyncio.Queue()
        self.stats = {
            'frames_processed': 0,
            'audio_chunks_processed': 0,
            'messages_processed': 0,
            'consent_violations': 0
        }
    
    async def process_incoming_data(self, data: dict, client_id: str):
        """Main entry point for processing incoming data"""
        data_type = data.get('type')
        user_id = data.get('user_id', client_id)
        
        # Check consent before processing
        if not self.consent_manager.check_consent(user_id, data_type.replace('_frame', '').replace('_chunk', '').replace('_message', '')):
            self.stats['consent_violations'] += 1
            await self.connection_manager.send_personal_message({
                "type": "consent_violation",
                "message": f"No consent given for {data_type} processing",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
            return
        
        # Process based on data type
        if data_type == "video_frame":
            result = self.media_processor.process_video_frame(
                data.get('frame_data'), client_id
            )
            if result:
                self.stats['frames_processed'] += 1
                await self.processing_queue.put(('video', result))
                
        elif data_type == "audio_chunk":
            audio_data = base64.b64decode(data.get('audio_data', ''))
            result = self.media_processor.process_audio_chunk(audio_data, client_id)
            if result:
                self.stats['audio_chunks_processed'] += 1
                await self.processing_queue.put(('audio', result))
                
        elif data_type == "chat_message":
            result = self.media_processor.process_text_message(
                data.get('message'), client_id
            )
            if result:
                self.stats['messages_processed'] += 1
                await self.processing_queue.put(('text', result))
    
    async def get_pipeline_stats(self) -> dict:
        """Get current pipeline statistics"""
        return {
            **self.stats,
            'active_connections': len(self.connection_manager.active_connections),
            'active_classrooms': len(self.connection_manager.classroom_participants),
            'queue_size': self.processing_queue.qsize(),
            'timestamp': datetime.utcnow().isoformat()
        }

# Global pipeline instance
data_pipeline = DataPipeline()