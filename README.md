# Emotion-Aware Virtual Classroom System

A comprehensive multimodal emotion recognition system for virtual learning environments with integrated video conferencing capabilities similar to Zoom.

## 🎯 Project Overview

This system combines advanced emotion recognition technologies with virtual classroom functionality to create an intelligent learning environment that monitors student engagement and provides real-time insights to instructors.

### Key Features

- **Multimodal Emotion Recognition**: Facial expressions, audio emotions, and text sentiment analysis
- **Virtual Classroom**: WebRTC-based video conferencing with Zoom-like functionality  
- **Real-time Analytics**: Live dashboard for instructors with emotion monitoring
- **Privacy Compliance**: GDPR-compliant consent system and data encryption
- **Scalable Deployment**: Docker/Kubernetes ready for 100+ concurrent students
- **Advanced Reports**: PDF/Excel export with comprehensive analytics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  Virtual        │   Instructor    │    Reports      │   │
│  │  Classroom      │   Dashboard     │    & Analytics  │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                         WebSocket/HTTP
                               │
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                         │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │   WebRTC API    │  Emotion API    │   Reports API   │   │
│  │   Auth & Security│  Dashboard API  │   Privacy API   │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                          ML Processing
                               │
┌─────────────────────────────────────────────────────────────┐
│                ML Modules (PyTorch)                         │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │   Facial CNN    │   Audio MFCC    │ Text Sentiment  │   │
│  │   (FER-2013)    │   (RAVDESS)     │ (HuggingFace)   │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
│                    Multimodal Fusion                        │
└─────────────────────────────────────────────────────────────┘
                               │
                        Data Storage
                               │
┌─────────────────────────────────────────────────────────────┐
│           Database & Storage (PostgreSQL + Redis)          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- Docker & Docker Compose (for deployment)
- Kubernetes (optional, for production)

### Development Setup

1. **Clone and Install**
```bash
git clone <repository-url>
cd emotion-aware-classroom
pip install -r requirements_complete.txt
cd frontend && npm install
```

2. **Start Backend**
```bash
python backend/main.py
```

3. **Start Frontend**
```bash
cd frontend && npm start
```

4. **Access Applications**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Production Deployment

1. **Docker Deployment**
```bash
docker-compose up -d
```

2. **Kubernetes Deployment**
```bash
./deployment/deploy.sh
```

## 📚 System Components

### Week 1-2: Core Setup & Facial Recognition
- ✅ Framework setup (PyTorch, FastAPI, React)
- ✅ WebRTC pipeline for video/audio capture
- ✅ Consent system for privacy compliance
- ✅ CNN model for facial emotion recognition (FER-2013 dataset)
- ✅ Real-time face detection with OpenCV

### Week 3-4: Audio & Text Analysis
- ✅ Audio emotion classifier using RAVDESS/IEMOCAP datasets
- ✅ MFCC feature extraction with Librosa
- ✅ HuggingFace transformer integration for text sentiment
- ✅ Live chat sentiment analysis API

### Week 5-6: Fusion & Dashboard
- ✅ Multimodal fusion system combining face/audio/text
- ✅ Learning state mapping (engaged, confused, bored, frustrated, curious, neutral)
- ✅ Real-time instructor dashboard with emotion monitoring
- ✅ WebSocket integration for live updates

### Week 7-8: Reports & Security
- ✅ Advanced visualization with charts and time-series analysis
- ✅ PDF/Excel report generation with comprehensive analytics
- ✅ JWT authentication and role-based access control
- ✅ Stream encryption and privacy compliance (GDPR)

### Week 9-10: Deployment & Testing
- ✅ Docker containerization with production configuration
- ✅ Kubernetes orchestration with auto-scaling
- ✅ Comprehensive stress testing for 100+ concurrent students
- ✅ Monitoring with Prometheus and Grafana

### Virtual Classroom Integration
- ✅ WebRTC-based video conferencing (Zoom-like functionality)
- ✅ Real-time chat and screen sharing
- ✅ Hand raising and participant management
- ✅ Integrated emotion monitoring during video calls
- ✅ Host controls for managing participants

## 🎥 Virtual Classroom Features

### For Students
- **Video Conferencing**: Join classroom with camera and microphone
- **Screen Sharing**: Share screen for presentations
- **Chat**: Real-time text chat with other participants  
- **Hand Raising**: Virtual hand raising for questions
- **Emotion Detection**: Automatic emotion recognition during class
- **Privacy Controls**: Enable/disable video and audio as needed

### For Instructors  
- **Host Controls**: Manage participant permissions and settings
- **Real-time Monitoring**: Live emotion analytics of all students
- **Recording**: Record classroom sessions (when consented)
- **Alerts**: Get notified of student confusion or disengagement
- **Reports**: Post-class analytics and engagement reports

### Technical Implementation
- **WebRTC**: Peer-to-peer video/audio streaming
- **WebSocket**: Real-time signaling and messaging
- **Emotion Integration**: Live emotion data overlaid on video feeds
- **Scalable Architecture**: Supports 100+ concurrent participants

## 📊 Emotion Recognition Pipeline

### Input Modalities
1. **Facial Expressions**: Live video stream analysis
2. **Audio Emotions**: Microphone input processing  
3. **Text Sentiment**: Chat message analysis

### Processing Pipeline
1. **Real-time Capture**: WebRTC streams capture video/audio
2. **Feature Extraction**: CNN features, MFCC coefficients, text embeddings
3. **Model Inference**: Trained models predict emotions per modality
4. **Multimodal Fusion**: Combine predictions using weighted averaging
5. **Learning State Mapping**: Map to educational states (engaged, confused, etc.)
6. **Dashboard Update**: Real-time visualization for instructors

### Supported Emotions
- **Engagement**: Active participation and focus
- **Confusion**: Difficulty understanding material
- **Boredom**: Lack of interest or attention
- **Frustration**: Struggling with concepts
- **Curiosity**: Interest in learning more
- **Neutral**: Baseline emotional state

## 🔧 API Documentation

### Virtual Classroom Endpoints

#### Room Management
- `POST /api/classroom/create-room` - Create new classroom
- `GET /api/classroom/rooms` - List available rooms
- `GET /api/classroom/room/{room_id}` - Get room details
- `DELETE /api/classroom/room/{room_id}` - Delete room (host only)

#### Real-time Communication
- `WebSocket /api/classroom/ws/{room_id}` - Join classroom WebSocket
- `GET /api/classroom/join/{room_id}` - Classroom web interface

#### Message Types (WebSocket)
- `webrtc_offer/answer/ice_candidate` - WebRTC signaling
- `chat_message` - Text chat
- `media_state_change` - Video/audio toggle
- `emotion_update` - Live emotion data
- `raise_hand` - Hand raising
- `host_control` - Instructor controls

### Emotion Recognition Endpoints
- `POST /api/v1/emotion/analyze` - Analyze emotion from image/audio/text
- `WebSocket /api/v1/emotion/stream` - Real-time emotion streaming
- `GET /api/v1/emotion/history` - Historical emotion data

### Dashboard & Reports
- `WebSocket /api/dashboard/ws/dashboard/{class_id}` - Real-time dashboard
- `GET /api/reports/api/class/{class_id}/analytics` - Analytics overview
- `GET /api/reports/api/class/{class_id}/export/pdf` - PDF report
- `GET /api/reports/api/class/{class_id}/export/excel` - Excel export

## 🛡️ Security & Privacy

### Authentication
- JWT-based authentication with refresh tokens
- Role-based access control (instructor/student/admin)
- Password strength validation and rate limiting

### Privacy Compliance
- GDPR-compliant consent management
- Data anonymization and retention policies
- Encrypted data transmission (TLS 1.3)
- Stream encryption using AES-256-CBC

### Data Protection
- No permanent storage of video/audio streams
- Emotion data aggregation and anonymization  
- User consent required for all data collection
- Right to data deletion and export

## 📈 Performance & Scalability

### System Capacity
- **Concurrent Students**: 100+ per classroom
- **Response Time**: <2 seconds average for emotion analysis
- **Uptime**: 99.9% availability target
- **Throughput**: 1000+ emotion updates per minute

### Scaling Configuration
- **Horizontal Pod Autoscaler**: CPU/memory based scaling
- **Load Balancing**: Nginx with multiple backend replicas
- **Database**: PostgreSQL with connection pooling
- **Cache**: Redis for session management and real-time data

### Monitoring
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Real-time dashboards and visualization
- **Health Checks**: Automated service monitoring
- **Log Aggregation**: Centralized logging for debugging

## 🧪 Testing

### Automated Testing
```bash
# Week 1-8 Individual Tests
python test_week1_setup.py
python test_week2_facial.py
python test_week3_audio.py
python test_week4_text.py
python test_week5_fusion.py
python test_week6_dashboard.py
python test_week7_reports.py
python test_week8_security.py

# Comprehensive Stress Testing
python test_week10_stress.py
```

### Manual Testing
- Virtual classroom functionality with multiple participants
- Emotion recognition accuracy across different conditions
- Dashboard real-time updates and alerts
- Report generation and export features

## 📁 Project Structure

```
emotion-aware-classroom/
├── backend/                    # FastAPI backend
│   ├── api/                   # API routes and endpoints
│   ├── core/                  # Configuration and database
│   ├── security/              # Authentication and encryption
│   └── main.py               # Application entry point
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/           # Main application pages
│   │   ├── services/        # API service layers
│   │   └── store/           # Redux state management
├── ml_modules/               # Machine learning components
│   ├── facial_emotion/      # CNN for facial recognition
│   ├── audio_emotion/       # Audio emotion classifier
│   ├── text_sentiment/      # Text sentiment analysis
│   └── fusion/              # Multimodal fusion system
├── deployment/              # Deployment configuration
│   ├── docker/             # Docker configurations
│   ├── kubernetes/         # Kubernetes manifests
│   └── deploy.sh          # Deployment scripts
└── docs/                   # Documentation and diagrams
```

## 🤝 Contributing

1. Follow the established code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure privacy compliance for any data handling
5. Test scalability impact for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Datasets**: FER-2013, RAVDESS, IEMOCAP for training emotion models
- **Frameworks**: FastAPI, React, PyTorch for core functionality
- **WebRTC**: For real-time video conferencing capabilities
- **Security**: JWT and encryption libraries for data protection

---

**🎉 The Emotion-Aware Virtual Classroom is now complete with full virtual conferencing capabilities!**

For support and questions, please refer to the API documentation at `/docs` endpoint.