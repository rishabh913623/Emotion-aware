"""
Virtual Classroom API endpoints
Implementation will be completed in Week 6 + Virtual Classroom Integration
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel
import uuid

from backend.core.database import get_db, Classroom, ClassroomSession, User
from api.auth import get_current_active_user

router = APIRouter()

# Pydantic models
class ClassroomCreate(BaseModel):
    name: str
    description: Optional[str] = None
    scheduled_time: datetime
    duration_minutes: int = 60
    max_participants: int = 100
    is_recorded: bool = False

class ClassroomResponse(BaseModel):
    id: str
    name: str
    instructor_id: str
    description: Optional[str]
    scheduled_time: datetime
    duration_minutes: int
    status: str
    meeting_room_id: str
    max_participants: int
    is_recorded: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

@router.post("/create", response_model=ClassroomResponse)
async def create_classroom(
    classroom_data: ClassroomCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create new virtual classroom"""
    
    if current_user.role not in ["instructor", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Only instructors and admins can create classrooms"
        )
    
    # Generate unique meeting room ID
    meeting_room_id = str(uuid.uuid4())[:8].upper()
    
    classroom = Classroom(
        name=classroom_data.name,
        instructor_id=current_user.id,
        description=classroom_data.description,
        scheduled_time=classroom_data.scheduled_time,
        duration_minutes=classroom_data.duration_minutes,
        meeting_room_id=meeting_room_id,
        max_participants=classroom_data.max_participants,
        is_recorded=classroom_data.is_recorded
    )
    
    db.add(classroom)
    db.commit()
    db.refresh(classroom)
    
    return ClassroomResponse.from_orm(classroom)

@router.get("/list")
async def list_classrooms(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List classrooms for current user"""
    
    if current_user.role == "instructor":
        classrooms = db.query(Classroom).filter(
            Classroom.instructor_id == current_user.id
        ).all()
    else:
        # Students see all active classrooms (in production, this would be filtered by enrollment)
        classrooms = db.query(Classroom).filter(
            Classroom.status.in_(["scheduled", "active"])
        ).all()
    
    return [ClassroomResponse.from_orm(classroom) for classroom in classrooms]

@router.get("/{classroom_id}")
async def get_classroom(
    classroom_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get classroom details"""
    
    classroom = db.query(Classroom).filter(
        Classroom.id == classroom_id
    ).first()
    
    if not classroom:
        raise HTTPException(
            status_code=404,
            detail="Classroom not found"
        )
    
    return ClassroomResponse.from_orm(classroom)

@router.post("/{classroom_id}/join")
async def join_classroom(
    classroom_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Join virtual classroom session"""
    
    classroom = db.query(Classroom).filter(
        Classroom.id == classroom_id
    ).first()
    
    if not classroom:
        raise HTTPException(
            status_code=404,
            detail="Classroom not found"
        )
    
    # Check if classroom is active or starting soon
    now = datetime.utcnow()
    start_time = classroom.scheduled_time
    end_time = start_time + timedelta(minutes=classroom.duration_minutes)
    
    if now < start_time - timedelta(minutes=15):  # 15 minutes before start
        raise HTTPException(
            status_code=400,
            detail="Classroom is not yet available for joining"
        )
    
    if now > end_time:
        raise HTTPException(
            status_code=400,
            detail="Classroom session has ended"
        )
    
    # Create session record
    session = ClassroomSession(
        classroom_id=classroom.id,
        user_id=current_user.id,
        joined_at=now
    )
    
    db.add(session)
    db.commit()
    
    return {
        "message": "Successfully joined classroom",
        "classroom_id": classroom_id,
        "meeting_room_id": classroom.meeting_room_id,
        "webrtc_config": {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"}
            ]
        }
    }

@router.post("/{classroom_id}/leave")
async def leave_classroom(
    classroom_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Leave virtual classroom session"""
    
    # Find active session
    session = db.query(ClassroomSession).filter(
        ClassroomSession.classroom_id == classroom_id,
        ClassroomSession.user_id == current_user.id,
        ClassroomSession.left_at.is_(None)
    ).first()
    
    if session:
        session.left_at = datetime.utcnow()
        session.session_duration = int((session.left_at - session.joined_at).total_seconds())
        db.commit()
    
    return {"message": "Successfully left classroom"}