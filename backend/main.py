"""
Emotion-Aware Virtual Classroom - Main FastAPI Application
Week 1: Setup & Data Pipeline
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
import uvicorn
import os
from contextlib import asynccontextmanager

from backend.core.config import get_settings
from backend.core.database import init_db
from backend.api import emotion, classroom, auth, consent, dashboard, reports, security, virtual_classroom
from backend.websocket.connection_manager import ConnectionManager

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    settings = get_settings()
    await init_db()
    print("🚀 Emotion-Aware Virtual Classroom Backend Started")
    print(f"📊 Environment: {settings.environment}")
    print(f"🔗 Database URL: {settings.database_url[:30]}...")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down backend...")

# Create FastAPI app
app = FastAPI(
    title="Emotion-Aware Virtual Classroom API",
    description="Multimodal emotion recognition system for virtual learning environments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
connection_manager = ConnectionManager()

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(consent.router, prefix="/api/v1/consent", tags=["Consent & Privacy"])
app.include_router(emotion.router, prefix="/api/v1/emotion", tags=["Emotion Recognition"])
app.include_router(classroom.router, prefix="/api/v1/classroom", tags=["Virtual Classroom"])
app.include_router(virtual_classroom.router, prefix="/api/classroom", tags=["Virtual Classroom WebRTC"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports & Analytics"])
app.include_router(security.router, prefix="/api/security", tags=["Security & Authentication"])

# Static files for frontend
if os.path.exists("frontend/build"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Emotion-Aware Virtual Classroom API",
        "version": "1.0.0",
        "status": "healthy",
        "features": [
            "multimodal_emotion_recognition",
            "virtual_classroom",
            "real_time_analytics",
            "privacy_compliance"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    settings = get_settings()
    return {
        "status": "healthy",
        "environment": settings.environment,
        "features": {
            "facial_emotion": True,
            "audio_emotion": True,
            "text_sentiment": True,
            "multimodal_fusion": True,
            "webrtc_streaming": True,
            "real_time_dashboard": True
        }
    }

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket, client_id: str):
    """WebSocket endpoint for real-time emotion data and video streaming"""
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process emotion data, video frames, etc.
            if data.get("type") == "emotion_frame":
                # Process emotion recognition
                await process_emotion_frame(data, client_id)
            elif data.get("type") == "audio_chunk":
                # Process audio emotion
                await process_audio_emotion(data, client_id)
            elif data.get("type") == "chat_message":
                # Process text sentiment
                await process_text_sentiment(data, client_id)
                
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
    finally:
        connection_manager.disconnect(client_id)

async def process_emotion_frame(data, client_id: str):
    """Process facial emotion recognition"""
    # This will be implemented in Week 2
    pass

async def process_audio_emotion(data, client_id: str):
    """Process audio emotion recognition"""
    # This will be implemented in Week 3
    pass

async def process_text_sentiment(data, client_id: str):
    """Process text sentiment analysis"""
    # This will be implemented in Week 4
    pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )