"""
Consent Management API endpoints
Week 1: Student opt-in and consent system for ethical compliance
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import uuid

from backend.core.database import get_db, ConsentRecord, User, Classroom
from api.auth import get_current_active_user

router = APIRouter()

# Pydantic models
class ConsentCreate(BaseModel):
    classroom_id: str
    facial_consent: bool = False
    audio_consent: bool = False
    text_consent: bool = False
    data_sharing_consent: bool = False

class ConsentResponse(BaseModel):
    id: str
    user_id: str
    classroom_id: str
    consent_given: bool
    facial_consent: bool
    audio_consent: bool
    text_consent: bool
    data_sharing_consent: bool
    consent_date: datetime
    
    class Config:
        from_attributes = True

class ConsentUpdate(BaseModel):
    facial_consent: Optional[bool] = None
    audio_consent: Optional[bool] = None
    text_consent: Optional[bool] = None
    data_sharing_consent: Optional[bool] = None

class PrivacyPolicy(BaseModel):
    version: str = "1.0"
    effective_date: str = "2024-01-01"
    content: dict

# Privacy policy content
PRIVACY_POLICY_CONTENT = {
    "introduction": {
        "title": "Emotion-Aware Virtual Classroom Privacy Policy",
        "description": "This policy explains how we collect, use, and protect your emotional and biometric data during virtual classroom sessions."
    },
    "data_collection": {
        "facial_data": {
            "description": "We analyze facial expressions to detect emotions in real-time",
            "purpose": "To assess student engagement and emotional states during learning",
            "retention": "Data is processed in real-time and not stored permanently unless explicitly consented",
            "opt_out": "You can disable facial emotion recognition at any time"
        },
        "audio_data": {
            "description": "We analyze voice tone and speech patterns to detect emotional states",
            "purpose": "To understand student engagement through voice analysis",
            "retention": "Audio is processed in real-time and not recorded unless session recording is enabled",
            "opt_out": "You can disable audio emotion analysis independently of your microphone"
        },
        "text_data": {
            "description": "We analyze chat messages for sentiment and emotional context",
            "purpose": "To provide instructors with insights into student communication patterns",
            "retention": "Chat messages may be stored for the duration of the course unless deleted",
            "opt_out": "You can disable sentiment analysis while still participating in chat"
        }
    },
    "data_usage": {
        "real_time_analysis": "Data is used to provide immediate feedback to instructors about class engagement",
        "learning_analytics": "Aggregated data helps improve teaching methods and course design",
        "research": "Anonymous data may be used for educational research with explicit consent",
        "no_identification": "Emotion data is never used to identify individuals outside the learning context"
    },
    "data_protection": {
        "encryption": "All data transmission is encrypted using industry-standard protocols",
        "access_control": "Only authorized instructors and administrators can access aggregated data",
        "anonymization": "Personal identifiers are removed from research datasets",
        "compliance": "We comply with GDPR, FERPA, and other applicable privacy regulations"
    },
    "your_rights": {
        "consent": "You have the right to give or withdraw consent at any time",
        "access": "You can request access to your personal data",
        "deletion": "You can request deletion of your data (right to be forgotten)",
        "portability": "You can request a copy of your data in a machine-readable format",
        "correction": "You can request correction of inaccurate data"
    },
    "contact": {
        "data_protection_officer": "privacy@emotion-classroom.edu",
        "support": "support@emotion-classroom.edu"
    }
}

def get_client_info(request: Request) -> dict:
    """Extract client information for consent logging"""
    return {
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent", "")
    }

@router.get("/privacy-policy")
async def get_privacy_policy():
    """Get current privacy policy"""
    return PrivacyPolicy(
        version="1.0",
        effective_date="2024-01-01",
        content=PRIVACY_POLICY_CONTENT
    )

@router.post("/consent", response_model=ConsentResponse)
async def create_consent_record(
    consent_data: ConsentCreate,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create or update consent record for a classroom"""
    
    # Verify classroom exists
    classroom = db.query(Classroom).filter(
        Classroom.id == consent_data.classroom_id
    ).first()
    
    if not classroom:
        raise HTTPException(
            status_code=404,
            detail="Classroom not found"
        )
    
    # Check if consent record already exists
    existing_consent = db.query(ConsentRecord).filter(
        ConsentRecord.user_id == current_user.id,
        ConsentRecord.classroom_id == consent_data.classroom_id
    ).first()
    
    client_info = get_client_info(request)
    
    if existing_consent:
        # Update existing consent
        existing_consent.facial_consent = consent_data.facial_consent
        existing_consent.audio_consent = consent_data.audio_consent
        existing_consent.text_consent = consent_data.text_consent
        existing_consent.data_sharing_consent = consent_data.data_sharing_consent
        existing_consent.consent_given = any([
            consent_data.facial_consent,
            consent_data.audio_consent,
            consent_data.text_consent
        ])
        existing_consent.consent_date = datetime.utcnow()
        existing_consent.ip_address = client_info["ip_address"]
        existing_consent.user_agent = client_info["user_agent"]
        
        db.commit()
        db.refresh(existing_consent)
        return ConsentResponse.from_orm(existing_consent)
    
    else:
        # Create new consent record
        consent_given = any([
            consent_data.facial_consent,
            consent_data.audio_consent,
            consent_data.text_consent
        ])
        
        consent_record = ConsentRecord(
            user_id=current_user.id,
            classroom_id=consent_data.classroom_id,
            consent_given=consent_given,
            facial_consent=consent_data.facial_consent,
            audio_consent=consent_data.audio_consent,
            text_consent=consent_data.text_consent,
            data_sharing_consent=consent_data.data_sharing_consent,
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"]
        )
        
        db.add(consent_record)
        db.commit()
        db.refresh(consent_record)
        
        return ConsentResponse.from_orm(consent_record)

@router.get("/consent/{classroom_id}", response_model=ConsentResponse)
async def get_consent_record(
    classroom_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get consent record for a classroom"""
    
    consent_record = db.query(ConsentRecord).filter(
        ConsentRecord.user_id == current_user.id,
        ConsentRecord.classroom_id == classroom_id
    ).first()
    
    if not consent_record:
        raise HTTPException(
            status_code=404,
            detail="Consent record not found"
        )
    
    return ConsentResponse.from_orm(consent_record)

@router.put("/consent/{classroom_id}", response_model=ConsentResponse)
async def update_consent_record(
    classroom_id: str,
    consent_update: ConsentUpdate,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update consent record for a classroom"""
    
    consent_record = db.query(ConsentRecord).filter(
        ConsentRecord.user_id == current_user.id,
        ConsentRecord.classroom_id == classroom_id
    ).first()
    
    if not consent_record:
        raise HTTPException(
            status_code=404,
            detail="Consent record not found"
        )
    
    # Update fields if provided
    client_info = get_client_info(request)
    
    if consent_update.facial_consent is not None:
        consent_record.facial_consent = consent_update.facial_consent
    if consent_update.audio_consent is not None:
        consent_record.audio_consent = consent_update.audio_consent
    if consent_update.text_consent is not None:
        consent_record.text_consent = consent_update.text_consent
    if consent_update.data_sharing_consent is not None:
        consent_record.data_sharing_consent = consent_update.data_sharing_consent
    
    # Update overall consent status
    consent_record.consent_given = any([
        consent_record.facial_consent,
        consent_record.audio_consent,
        consent_record.text_consent
    ])
    
    consent_record.consent_date = datetime.utcnow()
    consent_record.ip_address = client_info["ip_address"]
    consent_record.user_agent = client_info["user_agent"]
    
    db.commit()
    db.refresh(consent_record)
    
    return ConsentResponse.from_orm(consent_record)

@router.delete("/consent/{classroom_id}")
async def withdraw_consent(
    classroom_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Withdraw all consent for a classroom"""
    
    consent_record = db.query(ConsentRecord).filter(
        ConsentRecord.user_id == current_user.id,
        ConsentRecord.classroom_id == classroom_id
    ).first()
    
    if not consent_record:
        raise HTTPException(
            status_code=404,
            detail="Consent record not found"
        )
    
    # Set all consent to False
    consent_record.consent_given = False
    consent_record.facial_consent = False
    consent_record.audio_consent = False
    consent_record.text_consent = False
    consent_record.data_sharing_consent = False
    consent_record.consent_date = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Consent withdrawn successfully"}

@router.get("/consent")
async def list_consent_records(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all consent records for current user"""
    
    consent_records = db.query(ConsentRecord).filter(
        ConsentRecord.user_id == current_user.id
    ).all()
    
    return [ConsentResponse.from_orm(record) for record in consent_records]

@router.get("/data-export")
async def request_data_export(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Request export of all user data (GDPR compliance)"""
    
    # Get all user data
    consent_records = db.query(ConsentRecord).filter(
        ConsentRecord.user_id == current_user.id
    ).all()
    
    user_data = {
        "user_info": {
            "id": str(current_user.id),
            "email": current_user.email,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "role": current_user.role,
            "created_at": current_user.created_at.isoformat(),
            "preferences": current_user.preferences
        },
        "consent_records": [
            {
                "classroom_id": str(record.classroom_id),
                "consent_given": record.consent_given,
                "facial_consent": record.facial_consent,
                "audio_consent": record.audio_consent,
                "text_consent": record.text_consent,
                "data_sharing_consent": record.data_sharing_consent,
                "consent_date": record.consent_date.isoformat()
            }
            for record in consent_records
        ],
        "export_date": datetime.utcnow().isoformat(),
        "export_format": "JSON"
    }
    
    return user_data

@router.post("/data-deletion")
async def request_data_deletion(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Request deletion of all user data (Right to be forgotten)"""
    
    # Note: In production, this should trigger a background job
    # to safely delete data while preserving system integrity
    
    return {
        "message": "Data deletion request submitted",
        "user_id": str(current_user.id),
        "status": "pending",
        "estimated_completion": "within 30 days",
        "note": "You will receive email confirmation when deletion is complete"
    }