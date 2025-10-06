"""
Privacy Compliance and Data Protection Module
Week 8: GDPR compliance, data anonymization, consent management, and privacy controls
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from enum import Enum
import logging
from cryptography.fernet import Fernet
import re

from backend.core.database import get_db, User, ConsentRecord, EmotionData
from backend.security.stream_encryption import stream_encryption

# Setup logging
privacy_logger = logging.getLogger("privacy")

class ConsentType(Enum):
    FACIAL_RECOGNITION = "facial_recognition"
    AUDIO_ANALYSIS = "audio_analysis"  
    TEXT_ANALYSIS = "text_analysis"
    DATA_SHARING = "data_sharing"
    ANALYTICS = "analytics"
    RECORDING = "recording"

class DataRetentionPeriod(Enum):
    SESSION_ONLY = 0  # Delete after session
    ONE_WEEK = 7
    ONE_MONTH = 30
    THREE_MONTHS = 90
    ONE_YEAR = 365
    INDEFINITE = -1

class PrivacyComplianceService:
    """Service for GDPR compliance and privacy protection"""
    
    def __init__(self):
        self.anonymization_salt = self._get_anonymization_salt()
        self.data_retention_policies = self._initialize_retention_policies()
        
    def _get_anonymization_salt(self) -> str:
        """Get or create salt for anonymization"""
        salt_file = "anonymization_salt.key"
        try:
            with open(salt_file, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            salt = uuid.uuid4().hex
            with open(salt_file, "w") as f:
                f.write(salt)
            return salt
    
    def _initialize_retention_policies(self) -> Dict[str, DataRetentionPeriod]:
        """Initialize default data retention policies"""
        return {
            "emotion_data": DataRetentionPeriod.THREE_MONTHS,
            "session_recordings": DataRetentionPeriod.ONE_MONTH,
            "analytics_reports": DataRetentionPeriod.ONE_YEAR,
            "consent_records": DataRetentionPeriod.INDEFINITE,
            "audit_logs": DataRetentionPeriod.ONE_YEAR
        }
    
    def record_consent(
        self, 
        user_id: str, 
        classroom_id: str, 
        consent_types: List[ConsentType], 
        ip_address: str,
        user_agent: str,
        db: Session
    ) -> ConsentRecord:
        """Record user consent for data processing"""
        
        consent_record = ConsentRecord(
            user_id=user_id,
            classroom_id=classroom_id,
            consent_given=True,
            facial_consent=ConsentType.FACIAL_RECOGNITION in consent_types,
            audio_consent=ConsentType.AUDIO_ANALYSIS in consent_types,
            text_consent=ConsentType.TEXT_ANALYSIS in consent_types,
            data_sharing_consent=ConsentType.DATA_SHARING in consent_types,
            consent_date=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.add(consent_record)
        db.commit()
        
        privacy_logger.info(f"Recorded consent for user {user_id} in classroom {classroom_id}")
        return consent_record
    
    def withdraw_consent(
        self, 
        user_id: str, 
        classroom_id: str, 
        consent_types: List[ConsentType],
        db: Session
    ) -> bool:
        """Withdraw user consent and handle data deletion"""
        
        # Update consent record
        consent_record = db.query(ConsentRecord).filter(
            and_(
                ConsentRecord.user_id == user_id,
                ConsentRecord.classroom_id == classroom_id
            )
        ).first()
        
        if consent_record:
            if ConsentType.FACIAL_RECOGNITION in consent_types:
                consent_record.facial_consent = False
            if ConsentType.AUDIO_ANALYSIS in consent_types:
                consent_record.audio_consent = False
            if ConsentType.TEXT_ANALYSIS in consent_types:
                consent_record.text_consent = False
            if ConsentType.DATA_SHARING in consent_types:
                consent_record.data_sharing_consent = False
            
            db.commit()
            
            # Handle data deletion based on withdrawn consent
            self._handle_consent_withdrawal(user_id, classroom_id, consent_types, db)
            
            privacy_logger.info(f"Consent withdrawn for user {user_id} in classroom {classroom_id}")
            return True
        
        return False
    
    def _handle_consent_withdrawal(
        self, 
        user_id: str, 
        classroom_id: str, 
        consent_types: List[ConsentType],
        db: Session
    ):
        """Handle data deletion when consent is withdrawn"""
        
        # Delete or anonymize emotion data based on withdrawn consent
        emotion_records = db.query(EmotionData).filter(
            and_(
                EmotionData.user_id == user_id,
                EmotionData.classroom_id == classroom_id
            )
        ).all()
        
        for record in emotion_records:
            if ConsentType.FACIAL_RECOGNITION in consent_types:
                record.facial_emotion = None
                record.facial_confidence = None
            if ConsentType.AUDIO_ANALYSIS in consent_types:
                record.audio_emotion = None
                record.audio_confidence = None
            if ConsentType.TEXT_ANALYSIS in consent_types:
                record.text_sentiment = None
                record.text_confidence = None
        
        db.commit()
    
    def anonymize_user_data(self, user_id: str) -> str:
        """Create anonymized identifier for user"""
        combined = f"{user_id}_{self.anonymization_salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def anonymize_emotion_record(self, emotion_record: EmotionData) -> Dict[str, Any]:
        """Anonymize emotion data record"""
        
        anonymized_id = self.anonymize_user_data(emotion_record.user_id)
        
        return {
            "anonymized_user_id": anonymized_id,
            "classroom_id": emotion_record.classroom_id,
            "timestamp": emotion_record.timestamp.isoformat(),
            "learning_state": emotion_record.predicted_state,
            "confidence": emotion_record.fusion_confidence,
            "session_id": getattr(emotion_record, 'session_id', None)
        }
    
    def check_data_retention(self, db: Session):
        """Check and enforce data retention policies"""
        
        for data_type, retention_period in self.data_retention_policies.items():
            if retention_period == DataRetentionPeriod.INDEFINITE:
                continue
            
            cutoff_date = datetime.utcnow() - timedelta(days=retention_period.value)
            
            if data_type == "emotion_data":
                old_records = db.query(EmotionData).filter(
                    EmotionData.timestamp < cutoff_date
                ).all()
                
                for record in old_records:
                    db.delete(record)
                
                if old_records:
                    privacy_logger.info(f"Deleted {len(old_records)} old emotion records")
        
        db.commit()
    
    def export_user_data(self, user_id: str, db: Session) -> Dict[str, Any]:
        """Export all user data (GDPR right to data portability)"""
        
        # Get user information
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {}
        
        # Get consent records
        consent_records = db.query(ConsentRecord).filter(
            ConsentRecord.user_id == user_id
        ).all()
        
        # Get emotion data
        emotion_records = db.query(EmotionData).filter(
            EmotionData.user_id == user_id
        ).all()
        
        return {
            "user_profile": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            },
            "consent_history": [
                {
                    "classroom_id": record.classroom_id,
                    "consent_date": record.consent_date.isoformat(),
                    "facial_consent": record.facial_consent,
                    "audio_consent": record.audio_consent,
                    "text_consent": record.text_consent,
                    "data_sharing_consent": record.data_sharing_consent
                }
                for record in consent_records
            ],
            "emotion_data": [
                {
                    "classroom_id": record.classroom_id,
                    "timestamp": record.timestamp.isoformat(),
                    "facial_emotion": record.facial_emotion,
                    "audio_emotion": record.audio_emotion,
                    "text_sentiment": record.text_sentiment,
                    "learning_state": record.predicted_state,
                    "confidence": record.fusion_confidence
                }
                for record in emotion_records
            ],
            "export_date": datetime.utcnow().isoformat(),
            "data_retention_info": {
                data_type: period.value for data_type, period in self.data_retention_policies.items()
            }
        }
    
    def delete_user_data(self, user_id: str, db: Session) -> bool:
        """Delete all user data (GDPR right to erasure)"""
        
        try:
            # Delete emotion data
            emotion_records = db.query(EmotionData).filter(
                EmotionData.user_id == user_id
            ).delete()
            
            # Delete consent records (keep anonymized version for audit)
            consent_records = db.query(ConsentRecord).filter(
                ConsentRecord.user_id == user_id
            ).all()
            
            for record in consent_records:
                # Keep anonymized record for compliance
                record.user_id = self.anonymize_user_data(user_id)
                record.ip_address = "anonymized"
                record.user_agent = "anonymized"
            
            # Anonymize or delete user profile
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.email = f"deleted_user_{self.anonymize_user_data(user_id)}@deleted.local"
                user.username = f"deleted_user_{self.anonymize_user_data(user_id)}"
                user.full_name = "Deleted User"
                user.is_active = False
            
            db.commit()
            
            privacy_logger.info(f"Deleted/anonymized data for user {user_id}")
            return True
            
        except Exception as e:
            privacy_logger.error(f"Error deleting user data: {e}")
            db.rollback()
            return False
    
    def generate_privacy_report(self, classroom_id: Optional[str] = None, db: Session = None) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        
        report = {
            "report_date": datetime.utcnow().isoformat(),
            "classroom_id": classroom_id
        }
        
        if classroom_id:
            # Classroom-specific report
            consent_records = db.query(ConsentRecord).filter(
                ConsentRecord.classroom_id == classroom_id
            ).all()
            
            emotion_records = db.query(EmotionData).filter(
                EmotionData.classroom_id == classroom_id
            ).all()
            
        else:
            # Global report
            consent_records = db.query(ConsentRecord).all()
            emotion_records = db.query(EmotionData).all()
        
        # Calculate consent statistics
        total_consents = len(consent_records)
        facial_consents = sum(1 for r in consent_records if r.facial_consent)
        audio_consents = sum(1 for r in consent_records if r.audio_consent)
        text_consents = sum(1 for r in consent_records if r.text_consent)
        sharing_consents = sum(1 for r in consent_records if r.data_sharing_consent)
        
        # Calculate data statistics
        total_emotion_records = len(emotion_records)
        unique_users = len(set(r.user_id for r in emotion_records))
        
        report.update({
            "consent_statistics": {
                "total_consent_records": total_consents,
                "facial_consent_rate": (facial_consents / max(total_consents, 1)) * 100,
                "audio_consent_rate": (audio_consents / max(total_consents, 1)) * 100,
                "text_consent_rate": (text_consents / max(total_consents, 1)) * 100,
                "data_sharing_consent_rate": (sharing_consents / max(total_consents, 1)) * 100
            },
            "data_statistics": {
                "total_emotion_records": total_emotion_records,
                "unique_users": unique_users,
                "average_records_per_user": total_emotion_records / max(unique_users, 1)
            },
            "retention_policies": {
                data_type: {
                    "retention_days": period.value,
                    "description": period.name
                }
                for data_type, period in self.data_retention_policies.items()
            }
        })
        
        return report
    
    def validate_consent_for_processing(
        self, 
        user_id: str, 
        classroom_id: str, 
        processing_type: ConsentType, 
        db: Session
    ) -> bool:
        """Validate that user has given consent for specific data processing"""
        
        consent_record = db.query(ConsentRecord).filter(
            and_(
                ConsentRecord.user_id == user_id,
                ConsentRecord.classroom_id == classroom_id,
                ConsentRecord.consent_given == True
            )
        ).first()
        
        if not consent_record:
            return False
        
        consent_mapping = {
            ConsentType.FACIAL_RECOGNITION: consent_record.facial_consent,
            ConsentType.AUDIO_ANALYSIS: consent_record.audio_consent,
            ConsentType.TEXT_ANALYSIS: consent_record.text_consent,
            ConsentType.DATA_SHARING: consent_record.data_sharing_consent
        }
        
        return consent_mapping.get(processing_type, False)

class DataMaskingService:
    """Service for data masking and de-identification"""
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address for privacy"""
        if '@' not in email:
            return "****"
        
        local, domain = email.split('@', 1)
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1] if len(local) > 2 else '***'
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def mask_ip_address(ip: str) -> str:
        """Mask IP address for privacy"""
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        return "xxx.xxx.xxx.xxx"
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: str) -> str:
        """Hash sensitive data with salt"""
        combined = f"{data}_{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()

# Global privacy service instance
privacy_service = PrivacyComplianceService()
data_masking = DataMaskingService()

# Privacy middleware
class PrivacyMiddleware:
    """Middleware to ensure privacy compliance in API responses"""
    
    @staticmethod
    def sanitize_user_data(user_data: Dict[str, Any], requesting_user_id: str, target_user_id: str) -> Dict[str, Any]:
        """Sanitize user data based on privacy settings and requesting user"""
        
        # If requesting own data, return full data
        if requesting_user_id == target_user_id:
            return user_data
        
        # For other users, return sanitized data
        sanitized = user_data.copy()
        
        if 'email' in sanitized:
            sanitized['email'] = data_masking.mask_email(sanitized['email'])
        
        # Remove sensitive fields
        sensitive_fields = ['ip_address', 'user_agent', 'hashed_password']
        for field in sensitive_fields:
            sanitized.pop(field, None)
        
        return sanitized
    
    @staticmethod
    def filter_emotion_data_by_consent(
        emotion_data: List[Dict[str, Any]], 
        consent_check_func
    ) -> List[Dict[str, Any]]:
        """Filter emotion data based on user consent"""
        
        filtered_data = []
        
        for record in emotion_data:
            user_id = record.get('user_id')
            classroom_id = record.get('classroom_id')
            
            # Check consent for each modality
            filtered_record = record.copy()
            
            if not consent_check_func(user_id, classroom_id, ConsentType.FACIAL_RECOGNITION):
                filtered_record['facial_emotion'] = None
                filtered_record['facial_confidence'] = None
            
            if not consent_check_func(user_id, classroom_id, ConsentType.AUDIO_ANALYSIS):
                filtered_record['audio_emotion'] = None
                filtered_record['audio_confidence'] = None
            
            if not consent_check_func(user_id, classroom_id, ConsentType.TEXT_ANALYSIS):
                filtered_record['text_sentiment'] = None
                filtered_record['text_confidence'] = None
            
            filtered_data.append(filtered_record)
        
        return filtered_data