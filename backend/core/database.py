"""
Database configuration and models
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import asyncpg
import asyncio

from backend.core.config import get_settings

settings = get_settings()

# SQLAlchemy setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    """User model for students and instructors"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)  # student, instructor, admin
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    profile_image = Column(String(500))
    preferences = Column(JSON, default={})

class Classroom(Base):
    """Virtual classroom sessions"""
    __tablename__ = "classrooms"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    instructor_id = Column(UUID(as_uuid=True), nullable=False)
    description = Column(Text)
    scheduled_time = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, default=60)
    status = Column(String(20), default='scheduled')  # scheduled, active, ended, cancelled
    meeting_room_id = Column(String(100), unique=True)
    max_participants = Column(Integer, default=100)
    is_recorded = Column(Boolean, default=False)
    recording_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON, default={})

class ConsentRecord(Base):
    """Student consent for emotion monitoring"""
    __tablename__ = "consent_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    classroom_id = Column(UUID(as_uuid=True), nullable=False)
    consent_given = Column(Boolean, nullable=False)
    facial_consent = Column(Boolean, default=False)
    audio_consent = Column(Boolean, default=False)
    text_consent = Column(Boolean, default=False)
    data_sharing_consent = Column(Boolean, default=False)
    consent_date = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(Text)

class EmotionData(Base):
    """Emotion recognition results"""
    __tablename__ = "emotion_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    classroom_id = Column(UUID(as_uuid=True), nullable=False)
    session_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Individual modality results
    facial_emotion = Column(String(20))
    facial_confidence = Column(Float)
    audio_emotion = Column(String(20))
    audio_confidence = Column(Float)
    text_sentiment = Column(String(20))
    text_confidence = Column(Float)
    
    # Fusion results
    learning_state = Column(String(20))  # learning state (new field name)
    predicted_state = Column(String(20))  # learning state (alias for compatibility)
    fusion_confidence = Column(Float)
    confidence_score = Column(Float)  # alias for fusion_confidence
    
    # Raw data (optional, for debugging)
    raw_data = Column(JSON)

class ClassSession(Base):
    """Active class session management"""
    __tablename__ = "class_sessions"
    
    id = Column(String(100), primary_key=True)  # class_id from dashboard
    instructor_id = Column(String(100), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    is_active = Column(Boolean, default=True)
    participant_count = Column(Integer, default=0)
    session_data = Column(JSON, default={})

class ClassroomSession(Base):
    """Individual classroom session data"""
    __tablename__ = "classroom_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    classroom_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    left_at = Column(DateTime)
    session_duration = Column(Integer)  # seconds
    engagement_score = Column(Float)
    emotion_summary = Column(JSON)  # aggregated emotion data
    alerts_triggered = Column(JSON, default=[])

class AlertLog(Base):
    """System alerts and notifications"""
    __tablename__ = "alert_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    classroom_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))  # null for class-wide alerts
    alert_type = Column(String(50), nullable=False)  # confusion, boredom, technical
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    message = Column(Text, nullable=False)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    alert_metadata = Column(JSON, default={})

# Database dependency
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Database utilities
async def test_db_connection():
    """Test database connection"""
    try:
        # Test PostgreSQL connection
        conn = await asyncpg.connect(settings.database_url)
        await conn.close()
        return True
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False