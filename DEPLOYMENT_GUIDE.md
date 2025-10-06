# Deployment Guide - Emotion-Aware Virtual Classroom

## 🚀 Complete System Deployment

This guide provides step-by-step instructions for deploying the complete emotion-aware virtual classroom system with all features including the virtual classroom functionality.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended for production
- **Memory**: 8GB+ RAM for handling 100+ concurrent students  
- **Storage**: 20GB+ available space
- **Network**: High-bandwidth connection for video streaming

### Software Dependencies
- **Docker** & **Docker Compose** 20.10+
- **Kubernetes** 1.24+ (for production scaling)
- **Python** 3.9+
- **Node.js** 16+

## 🔧 Quick Development Setup

### 1. Environment Setup
```bash
# Clone and navigate to project
cd "/Users/rishabh/Desktop/Emotion aware"

# Install Python dependencies
pip install -r requirements_complete.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Start Development Servers
```bash
# Start backend (Terminal 1)
python backend/main.py

# Start frontend (Terminal 2) 
cd frontend && npm start
```

### 3. Access Applications
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Virtual Classroom**: http://localhost:8000/api/classroom/join/[room-id]

## 🐳 Docker Deployment

### 1. Build and Run with Docker Compose
```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 2. Services Included
- **Backend**: FastAPI server with emotion recognition
- **Frontend**: React application with nginx
- **Database**: PostgreSQL with emotion data storage
- **Cache**: Redis for session management
- **Worker**: Celery for background processing
- **Monitoring**: Prometheus + Grafana stack

### 3. Access Production Deployment
- **Application**: http://localhost:8080
- **API**: http://localhost:8080/api
- **Monitoring**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)

## ☸️ Kubernetes Production Deployment

### 1. Deploy to Kubernetes
```bash
# Deploy using automated script
./deployment/deploy.sh

# Or deploy manually
kubectl apply -f deployment/kubernetes/
```

### 2. Verify Deployment
```bash
# Check deployment status
./deployment/deploy.sh status

# View pods and services
kubectl get pods -n emotion-classroom
kubectl get svc -n emotion-classroom
```

### 3. Access Production Application
```bash
# Get external IP
kubectl get svc nginx-service -n emotion-classroom

# Or use port forwarding for testing
kubectl port-forward svc/nginx-service 8080:80 -n emotion-classroom
```

## 🎯 Feature Verification

### Core Functionality Tests
```bash
# Test emotion recognition pipeline
python test_integration_ml.py

# Test dashboard functionality  
python test_week6_dashboard.py

# Test virtual classroom
# (Open browser to classroom URL and test video/audio)

# Run comprehensive stress test
python test_week10_stress.py
```

### Manual Feature Testing

#### 1. Virtual Classroom Features
- ✅ **Video Conferencing**: Join room with camera/mic
- ✅ **Screen Sharing**: Share screen during sessions
- ✅ **Real-time Chat**: Text messaging between participants
- ✅ **Hand Raising**: Virtual hand raising mechanism
- ✅ **Host Controls**: Instructor management features
- ✅ **Emotion Integration**: Live emotion overlays

#### 2. Emotion Recognition Pipeline  
- ✅ **Facial Emotions**: Real-time face detection and analysis
- ✅ **Audio Emotions**: Microphone input processing
- ✅ **Text Sentiment**: Chat message analysis
- ✅ **Multimodal Fusion**: Combined emotion predictions
- ✅ **Learning States**: Educational context mapping

#### 3. Dashboard & Analytics
- ✅ **Real-time Monitoring**: Live student emotion tracking
- ✅ **Alerts System**: Automatic confusion/boredom detection
- ✅ **Historical Reports**: Time-series emotion analysis
- ✅ **Export Features**: PDF/Excel report generation

## 🔒 Security Configuration

### 1. Authentication Setup
```bash
# Set JWT secret key
export JWT_SECRET_KEY="your-super-secret-jwt-key-here"

# Configure password requirements
export MIN_PASSWORD_LENGTH=8
export REQUIRE_PASSWORD_COMPLEXITY=true
```

### 2. Database Security
```bash
# Set database credentials
export POSTGRES_USER="emotion_user" 
export POSTGRES_PASSWORD="secure_password_here"
export POSTGRES_DB="emotion_classroom"
```

### 3. SSL/TLS Configuration
```bash
# Enable HTTPS in production
export ENABLE_SSL=true
export SSL_CERT_PATH="/path/to/cert.pem"
export SSL_KEY_PATH="/path/to/key.pem"
```

## 📊 Monitoring & Observability

### 1. Prometheus Metrics
- **System Metrics**: CPU, memory, network usage
- **Application Metrics**: Request rates, response times
- **Business Metrics**: Active sessions, emotion predictions

### 2. Grafana Dashboards
- **System Overview**: Infrastructure health monitoring
- **Application Performance**: API response times and errors  
- **Virtual Classroom Analytics**: Session metrics and usage

### 3. Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Frontend health  
curl http://localhost:3000

# Database connectivity
curl http://localhost:8000/api/health/db
```

## 🔧 Scaling Configuration

### Horizontal Pod Autoscaler (HPA)
```yaml
# Automatically scale based on CPU/memory
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Testing
```bash
# Test system with 100 concurrent users
python test_week10_stress.py

# Expected performance:
# - 100+ concurrent students supported
# - <2 second response time
# - 95%+ success rate
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Backend Connection Errors
```bash
# Check if backend is running
curl http://localhost:8000/health

# Check logs
docker-compose logs backend
kubectl logs -l app=backend -n emotion-classroom
```

#### 2. WebRTC Connection Issues
```bash
# Check STUN/TURN server configuration
export WEBRTC_STUN_SERVERS="stun:stun.l.google.com:19302"
export WEBRTC_TURN_SERVERS="turn:your-turn-server:3478"

# Test browser WebRTC support
# Open browser console and check for WebRTC errors
```

#### 3. Emotion Model Loading Errors
```bash
# Verify model files exist
ls ml_modules/*/models/

# Check model compatibility
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

#### 4. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready

# Test connection
docker-compose exec backend python -c "from backend.database.database import test_connection; test_connection()"
```

## 📈 Performance Optimization

### 1. Backend Optimization
- **Increase worker processes**: Adjust gunicorn workers in Dockerfile
- **Database connection pooling**: Configure PostgreSQL pool size
- **Caching**: Enable Redis caching for frequent queries

### 2. Frontend Optimization
- **Code splitting**: Implement React lazy loading
- **Asset optimization**: Compress images and bundle files
- **CDN**: Use CDN for static asset delivery

### 3. Network Optimization
- **WebRTC optimization**: Configure proper STUN/TURN servers
- **Bandwidth management**: Implement adaptive video quality
- **Connection pooling**: Optimize WebSocket connections

## 🔄 Backup & Recovery

### 1. Database Backup
```bash
# Create database backup
kubectl exec postgres-0 -n emotion-classroom -- pg_dump -U emotion_user emotion_classroom > backup.sql

# Restore from backup
kubectl exec -i postgres-0 -n emotion-classroom -- psql -U emotion_user emotion_classroom < backup.sql
```

### 2. Configuration Backup
```bash
# Backup Kubernetes configs
kubectl get all -n emotion-classroom -o yaml > k8s-backup.yaml

# Backup Docker configs  
cp docker-compose.yml docker-compose.backup.yml
```

## 🎓 Production Checklist

### Before Going Live
- [ ] SSL certificates configured
- [ ] Database credentials secured
- [ ] JWT secret keys set
- [ ] Monitoring dashboards configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security scanning performed
- [ ] Documentation updated

### Post-Deployment  
- [ ] Health checks verified
- [ ] Performance monitoring active
- [ ] User access tested
- [ ] Virtual classroom functionality verified
- [ ] Emotion recognition accuracy validated
- [ ] Scaling policies configured

## 📞 Support

For deployment issues or questions:

1. **Check logs**: Always start with application and system logs
2. **Verify configuration**: Ensure all environment variables are set
3. **Test components**: Isolate and test individual system components  
4. **Monitor metrics**: Use Grafana dashboards for system insights
5. **Review documentation**: Refer to API docs at `/docs` endpoint

---

**🎉 The Emotion-Aware Virtual Classroom is ready for production deployment!**

This system now includes complete virtual classroom functionality with WebRTC video conferencing, real-time emotion monitoring, and comprehensive analytics - providing a complete solution for modern online education.