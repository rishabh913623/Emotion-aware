# Virtual Classroom - Important Notes

## Current Implementation

This is a **WebSocket-based classroom simulation** that demonstrates the interface and features, but **does not implement full WebRTC peer-to-peer connections** for real audio/video streaming.

### ✅ What Works:
- Real-time participant presence tracking
- Chat messaging between all participants
- Automatic attendance recording
- Student emotion detection (simulated)
- Video/audio toggle UI controls
- Screen sharing toggle (visual indicator only)
- Role-based permissions (Instructor vs Student)

### ⚠️ Current Limitations:
1. **Audio/Video**: Not transmitted between participants (no WebRTC implementation)
   - Camera/mic access is requested but streams are only local
   - Remote participants see avatars instead of video feeds
   
2. **Screen Sharing**: Shows notification but doesn't transmit actual screen content
   - Would require WebRTC implementation for real screen sharing

3. **Emotion Detection**: Currently simulated with random emotions
   - Real implementation would need ML models and face detection

## To Add Real Audio/Video:

Would require implementing:
1. STUN/TURN servers for NAT traversal
2. WebRTC peer connections (RTCPeerConnection)
3. SDP offer/answer exchange via WebSocket
4. ICE candidate gathering and exchange
5. Media stream handling for each peer

## Current Use Case:

This classroom is perfect for:
- **Text-based collaboration** with real-time chat
- **Attendance tracking** for online sessions
- **Presence awareness** (who's in the room)
- **Emotion monitoring** (when ML models are integrated)
- **Demo/prototype** of classroom features

## Recommended Setup:

For production use with real audio/video, consider integrating:
- **Agora.io** - Enterprise WebRTC service
- **Jitsi Meet** - Open-source video conferencing
- **Daily.co** - Video API platform
- **Twilio Video** - Programmable video API

Or implement custom WebRTC with services like:
- **Coturn** (TURN server)
- **Mediasoup** (SFU for multi-party calls)
