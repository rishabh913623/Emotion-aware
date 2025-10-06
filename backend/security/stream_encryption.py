"""
Stream Encryption and Secure Communication Module
Week 8: WebRTC stream encryption, secure data transmission, and privacy protection
"""

import asyncio
import json
import ssl
import logging
from typing import Dict, Any, Optional, Set
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets
import base64
from datetime import datetime, timedelta
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
crypto_logger = logging.getLogger("crypto")

class StreamEncryptionService:
    """Service for encrypting WebRTC streams and secure communication"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_keys: Dict[str, bytes] = {}
        self.master_key = self._generate_master_key()
        self.fernet = Fernet(self.master_key)
        
    def _generate_master_key(self) -> bytes:
        """Generate or load master encryption key"""
        key_file = "stream_master_key.key"
        try:
            with open(key_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            crypto_logger.info("Generated new master encryption key")
            return key
    
    def generate_session_key(self, session_id: str) -> str:
        """Generate unique encryption key for session"""
        session_key = secrets.token_bytes(32)  # 256-bit key
        self.session_keys[session_id] = session_key
        
        # Return base64 encoded key for transmission
        encoded_key = base64.b64encode(session_key).decode()
        crypto_logger.info(f"Generated session key for: {session_id}")
        return encoded_key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt general data using master key"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt general data using master key"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_stream_data(self, session_id: str, data: bytes) -> bytes:
        """Encrypt stream data using session key"""
        if session_id not in self.session_keys:
            raise ValueError(f"No session key found for {session_id}")
        
        session_key = self.session_keys[session_id]
        
        # Generate random IV for each encryption
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(session_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Encrypt data
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + encrypted data
        return iv + encrypted_data
    
    def decrypt_stream_data(self, session_id: str, encrypted_data: bytes) -> bytes:
        """Decrypt stream data using session key"""
        if session_id not in self.session_keys:
            raise ValueError(f"No session key found for {session_id}")
        
        session_key = self.session_keys[session_id]
        
        # Extract IV and encrypted data
        iv = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        
        # Decrypt
        cipher = Cipher(algorithms.AES(session_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def encrypt_emotion_data(self, emotion_data: Dict[str, Any]) -> str:
        """Encrypt emotion data for secure storage/transmission"""
        json_data = json.dumps(emotion_data)
        encrypted = self.fernet.encrypt(json_data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_emotion_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt emotion data"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode())
    
    def create_secure_session(self, session_id: str, participants: Set[str]) -> Dict[str, Any]:
        """Create secure session with encryption keys"""
        session_key = self.generate_session_key(session_id)
        
        session_info = {
            "session_id": session_id,
            "participants": list(participants),
            "created_at": datetime.now().isoformat(),
            "encryption_enabled": True,
            "key_rotation_interval": 3600,  # 1 hour
            "last_key_rotation": datetime.now().isoformat()
        }
        
        self.active_sessions[session_id] = session_info
        
        return {
            "session_info": session_info,
            "session_key": session_key,
            "encryption_algorithm": "AES-256-CBC"
        }
    
    def rotate_session_key(self, session_id: str) -> str:
        """Rotate encryption key for session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Generate new key
        new_key = self.generate_session_key(session_id)
        
        # Update session info
        self.active_sessions[session_id]["last_key_rotation"] = datetime.now().isoformat()
        
        crypto_logger.info(f"Rotated encryption key for session: {session_id}")
        return new_key
    
    def end_secure_session(self, session_id: str):
        """Clean up session keys and data"""
        if session_id in self.session_keys:
            del self.session_keys[session_id]
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        crypto_logger.info(f"Ended secure session: {session_id}")

class AccessControlService:
    """Service for managing access control and permissions"""
    
    def __init__(self):
        self.session_permissions: Dict[str, Dict[str, Set[str]]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
    def grant_session_access(self, session_id: str, user_id: str, permissions: Set[str]):
        """Grant user access to session with specific permissions"""
        if session_id not in self.session_permissions:
            self.session_permissions[session_id] = {}
        
        self.session_permissions[session_id][user_id] = permissions
        logging.info(f"Granted access to {user_id} for session {session_id}: {permissions}")
    
    def check_permission(self, session_id: str, user_id: str, required_permission: str) -> bool:
        """Check if user has required permission for session"""
        if session_id not in self.session_permissions:
            return False
        
        if user_id not in self.session_permissions[session_id]:
            return False
        
        return required_permission in self.session_permissions[session_id][user_id]
    
    def revoke_session_access(self, session_id: str, user_id: str):
        """Revoke user access from session"""
        if session_id in self.session_permissions:
            if user_id in self.session_permissions[session_id]:
                del self.session_permissions[session_id][user_id]
                logging.info(f"Revoked access for {user_id} from session {session_id}")
    
    def apply_rate_limit(self, user_id: str, action: str, limit: int, window_seconds: int) -> bool:
        """Apply rate limiting for user actions"""
        now = datetime.now()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {}
        
        if action not in self.rate_limits[user_id]:
            self.rate_limits[user_id][action] = {
                "count": 0,
                "window_start": now,
                "limit": limit,
                "window_seconds": window_seconds
            }
        
        action_data = self.rate_limits[user_id][action]
        
        # Check if we're in a new window
        if (now - action_data["window_start"]).seconds >= action_data["window_seconds"]:
            action_data["count"] = 0
            action_data["window_start"] = now
        
        # Check rate limit
        if action_data["count"] >= action_data["limit"]:
            return False  # Rate limit exceeded
        
        action_data["count"] += 1
        return True
    
    def get_session_participants(self, session_id: str) -> Dict[str, Set[str]]:
        """Get all participants and their permissions for a session"""
        return self.session_permissions.get(session_id, {})

class SecureWebSocketHandler:
    """Secure WebSocket handler with encryption and access control"""
    
    def __init__(self, encryption_service: StreamEncryptionService, access_service: AccessControlService):
        self.encryption = encryption_service
        self.access = access_service
        self.active_connections: Dict[str, Any] = {}
    
    async def handle_connection(self, websocket, session_id: str, user_id: str):
        """Handle secure WebSocket connection"""
        
        # Check if user has access to session
        if not self.access.check_permission(session_id, user_id, "join_session"):
            await websocket.close(code=4003, reason="Access denied")
            return
        
        # Store connection
        connection_id = f"{session_id}_{user_id}"
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "session_id": session_id,
            "user_id": user_id,
            "connected_at": datetime.now()
        }
        
        try:
            # Send encryption key to client
            session_key = self.encryption.generate_session_key(session_id)
            await websocket.send(json.dumps({
                "type": "encryption_key",
                "key": session_key,
                "algorithm": "AES-256-CBC"
            }))
            
            async for message in websocket:
                await self.handle_message(connection_id, message)
                
        except Exception as e:
            logging.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle encrypted WebSocket message"""
        
        connection = self.active_connections[connection_id]
        session_id = connection["session_id"]
        user_id = connection["user_id"]
        
        try:
            # Parse message
            data = json.loads(message)
            
            # Check rate limits
            if not self.access.apply_rate_limit(user_id, "message", 100, 60):  # 100 messages per minute
                await connection["websocket"].send(json.dumps({
                    "type": "error",
                    "message": "Rate limit exceeded"
                }))
                return
            
            # Handle different message types
            if data["type"] == "emotion_data":
                await self.handle_emotion_data(connection, data)
            elif data["type"] == "stream_data":
                await self.handle_stream_data(connection, data)
            elif data["type"] == "chat_message":
                await self.handle_chat_message(connection, data)
                
        except Exception as e:
            logging.error(f"Message handling error for {connection_id}: {e}")
    
    async def handle_emotion_data(self, connection: Dict[str, Any], data: Dict[str, Any]):
        """Handle encrypted emotion data"""
        
        user_id = connection["user_id"]
        session_id = connection["session_id"]
        
        # Check permissions
        if not self.access.check_permission(session_id, user_id, "send_emotion_data"):
            await connection["websocket"].send(json.dumps({
                "type": "error",
                "message": "No permission to send emotion data"
            }))
            return
        
        # Encrypt emotion data
        encrypted_data = self.encryption.encrypt_emotion_data(data["emotion"])
        
        # Broadcast to authorized participants
        await self.broadcast_to_session(session_id, {
            "type": "emotion_update",
            "user_id": user_id,
            "encrypted_data": encrypted_data,
            "timestamp": datetime.now().isoformat()
        }, "receive_emotion_data")
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any], required_permission: str):
        """Broadcast message to all authorized session participants"""
        
        message_json = json.dumps(message)
        
        for connection_id, connection in self.active_connections.items():
            if connection["session_id"] == session_id:
                user_id = connection["user_id"]
                
                # Check if user has permission to receive this message type
                if self.access.check_permission(session_id, user_id, required_permission):
                    try:
                        await connection["websocket"].send(message_json)
                    except Exception as e:
                        logging.error(f"Failed to send message to {connection_id}: {e}")

# Global service instances
stream_encryption = StreamEncryptionService()
access_control = AccessControlService()
secure_websocket = SecureWebSocketHandler(stream_encryption, access_control)

# Permission constants
class Permissions:
    JOIN_SESSION = "join_session"
    SEND_EMOTION_DATA = "send_emotion_data"
    RECEIVE_EMOTION_DATA = "receive_emotion_data"
    SEND_AUDIO = "send_audio"
    RECEIVE_AUDIO = "receive_audio"
    SEND_VIDEO = "send_video"
    RECEIVE_VIDEO = "receive_video"
    MODERATE_SESSION = "moderate_session"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"

# Role-based permission sets
STUDENT_PERMISSIONS = {
    Permissions.JOIN_SESSION,
    Permissions.SEND_EMOTION_DATA,
    Permissions.RECEIVE_EMOTION_DATA,
    Permissions.SEND_AUDIO,
    Permissions.RECEIVE_AUDIO,
    Permissions.SEND_VIDEO,
    Permissions.RECEIVE_VIDEO
}

INSTRUCTOR_PERMISSIONS = STUDENT_PERMISSIONS | {
    Permissions.MODERATE_SESSION,
    Permissions.VIEW_ANALYTICS,
    Permissions.EXPORT_DATA
}

ADMIN_PERMISSIONS = INSTRUCTOR_PERMISSIONS  # Full access