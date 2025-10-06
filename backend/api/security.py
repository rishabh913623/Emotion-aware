"""
Enhanced Security and Authentication Module
Week 8: JWT authentication, encryption, access control, and privacy compliance
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import secrets
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging
from pydantic import BaseModel, EmailStr
import ipaddress
import re
from functools import wraps

from backend.core.database import get_db, User, ConsentRecord, ClassSession
from backend.core.config import get_settings

router = APIRouter()
settings = get_settings()
security = HTTPBearer()

# Logging setup
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger("security")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

class SecurityService:
    """Comprehensive security service for authentication and encryption"""
    
    def __init__(self):
        self.failed_login_attempts = {}  # IP-based tracking
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for data encryption"""
        key_file = "encryption_key.key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        try:
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt
        except Exception as e:
            security_logger.error(f"JWT encoding error: {e}")
            raise HTTPException(status_code=500, detail="Token generation failed")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def is_ip_locked(self, ip_address: str) -> bool:
        """Check if IP is locked due to failed login attempts"""
        if ip_address in self.failed_login_attempts:
            attempts_info = self.failed_login_attempts[ip_address]
            if attempts_info["count"] >= self.max_failed_attempts:
                if datetime.now() - attempts_info["last_attempt"] < timedelta(seconds=self.lockout_duration):
                    return True
                else:
                    # Reset after lockout period
                    del self.failed_login_attempts[ip_address]
        return False
    
    def record_failed_login(self, ip_address: str):
        """Record a failed login attempt"""
        if ip_address not in self.failed_login_attempts:
            self.failed_login_attempts[ip_address] = {"count": 0, "last_attempt": datetime.now()}
        
        self.failed_login_attempts[ip_address]["count"] += 1
        self.failed_login_attempts[ip_address]["last_attempt"] = datetime.now()
        
        security_logger.warning(f"Failed login attempt from IP: {ip_address}")
        
        # This implements rate limiting for failed logins
        return self.is_ip_locked(ip_address)
    
    def reset_failed_login(self, ip_address: str):
        """Reset failed login attempts for successful login"""
        if ip_address in self.failed_login_attempts:
            del self.failed_login_attempts[ip_address]
    
    def validate_password_strength(self, password: str) -> tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize input to prevent injection attacks"""
        # Remove potential SQL injection patterns
        dangerous_patterns = [
            r"'", r'"', r";", r"--", r"/*", r"*/", r"xp_", r"sp_",
            r"<script", r"</script>", r"javascript:", r"vbscript:",
            r"onload", r"onerror", r"onclick"
        ]
        
        sanitized = input_str
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

# Initialize security service
security_service = SecurityService()

# Pydantic models
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    password: str
    role: str = "student"  # student, instructor, admin

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user_info: dict

# Authentication dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    
    try:
        token = credentials.credentials
        payload = security_service.verify_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated"
            )
        
        return user
        
    except Exception as e:
        security_logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Role-based access control decorators
def require_role(required_role: str):
    """Decorator to require specific user role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user or current_user.role != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. {required_role} role required."
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_roles(required_roles: List[str]):
    """Decorator to require any of the specified roles"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user or current_user.role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. One of {required_roles} roles required."
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# IP validation
def validate_ip_access(request: Request):
    """Validate IP access (can be extended with whitelist/blacklist)"""
    client_ip = request.client.host
    
    # Check if IP is locked
    if security_service.is_ip_locked(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later."
        )
    
    return client_ip

# Authentication endpoints
@router.post("/register", response_model=TokenResponse)
async def register_user(
    user_data: UserRegister,
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    
    client_ip = validate_ip_access(request)
    
    try:
        # Sanitize inputs
        email = security_service.sanitize_input(user_data.email)
        username = security_service.sanitize_input(user_data.username)
        full_name = security_service.sanitize_input(user_data.full_name)
        
        # Validate password strength
        is_strong, errors = security_service.validate_password_strength(user_data.password)
        if not is_strong:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Password does not meet requirements", "errors": errors}
            )
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == email) | (User.username == username)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email or username already exists"
            )
        
        # Create new user
        hashed_password = security_service.hash_password(user_data.password)
        
        new_user = User(
            email=email,
            username=username,
            full_name=full_name,
            role=user_data.role,
            hashed_password=hashed_password,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security_service.create_access_token(
            data={"sub": str(new_user.id), "role": new_user.role},
            expires_delta=access_token_expires
        )
        
        security_logger.info(f"New user registered: {email} from IP: {client_ip}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "id": str(new_user.id),
                "email": new_user.email,
                "username": new_user.username,
                "full_name": new_user.full_name,
                "role": new_user.role
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        security_logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token"""
    
    client_ip = validate_ip_access(request)
    
    try:
        # Sanitize email input
        email = security_service.sanitize_input(login_data.email)
        
        # Get user from database
        user = db.query(User).filter(User.email == email).first()
        
        if not user or not security_service.verify_password(login_data.password, user.hashed_password):
            security_service.record_failed_login(client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated"
            )
        
        # Reset failed login attempts
        security_service.reset_failed_login(client_ip)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        if login_data.remember_me:
            token_expires = timedelta(days=7)  # Extended for remember me
        
        access_token = security_service.create_access_token(
            data={"sub": str(user.id), "role": user.role},
            expires_delta=token_expires
        )
        
        security_logger.info(f"User login: {email} from IP: {client_ip}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(token_expires.total_seconds()),
            user_info={
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        security_logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    
    try:
        # Verify current password
        if not security_service.verify_password(password_data.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        is_strong, errors = security_service.validate_password_strength(password_data.new_password)
        if not is_strong:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "New password does not meet requirements", "errors": errors}
            )
        
        # Update password
        current_user.hashed_password = security_service.hash_password(password_data.new_password)
        db.commit()
        
        security_logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        security_logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.get("/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }

@router.post("/logout")
async def logout_user(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Logout user (invalidate token on client side)"""
    
    client_ip = request.client.host
    security_logger.info(f"User logout: {current_user.email} from IP: {client_ip}")
    
    return {"message": "Logged out successfully"}

# Admin endpoints (require admin role)
@router.get("/admin/users")
async def get_all_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    users = db.query(User).all()
    
    return {
        "users": [
            {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            for user in users
        ]
    }

@router.put("/admin/users/{user_id}/activate")
async def activate_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Activate/deactivate user (admin only)"""
    
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = not user.is_active
    db.commit()
    
    security_logger.info(f"User {user.email} {'activated' if user.is_active else 'deactivated'} by admin {current_user.email}")
    
    return {
        "message": f"User {'activated' if user.is_active else 'deactivated'} successfully",
        "user_id": str(user.id),
        "is_active": user.is_active
    }