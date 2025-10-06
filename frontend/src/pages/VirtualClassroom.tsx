import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Card,
  CardContent,
  IconButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Chip,
  Tooltip,
  Paper,
} from '@mui/material';
import {
  Videocam,
  VideocamOff,
  Mic,
  MicOff,
  ScreenShare,
  StopScreenShare,
  PanTool,
  Chat,
  CallEnd,
  Group,
  Settings,
  FullscreenExit,
  Fullscreen,
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';

interface Participant {
  user_id: string;
  username: string;
  role: string;
  video_enabled: boolean;
  audio_enabled: boolean;
  screen_sharing: boolean;
  hand_raised: boolean;
  emotion_data?: {
    primary_emotion: string;
    confidence: number;
    learning_state: string;
  };
}

interface ChatMessage {
  type: string;
  user_id: string;
  username: string;
  message: string;
  timestamp: string;
}

const VirtualClassroom: React.FC = () => {
  const { roomId } = useParams<{ roomId: string }>();
  const navigate = useNavigate();
  
  // State
  const [participants, setParticipants] = useState<Participant[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [handRaised, setHandRaised] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessage, setChatMessage] = useState('');
  const [isHost, setIsHost] = useState(false);
  const [currentUser, setCurrentUser] = useState<string>('');
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Refs
  const websocketRef = useRef<WebSocket | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionsRef = useRef<{ [key: string]: RTCPeerConnection }>({});
  const chatEndRef = useRef<HTMLDivElement>(null);

  // WebRTC Configuration
  const rtcConfiguration = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' }
    ]
  };

  // Initialize classroom
  useEffect(() => {
    initializeClassroom();
    return () => {
      cleanup();
    };
  }, [roomId]);

  // Auto-scroll chat
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages]);

  const initializeClassroom = async () => {
    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true
      });
      
      localStreamRef.current = stream;
      
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }
      
      // Connect to WebSocket
      connectWebSocket();
      
    } catch (error) {
      console.error('Error initializing classroom:', error);
      alert('Failed to access camera/microphone. Please check permissions.');
    }
  };

  const connectWebSocket = () => {
    const wsUrl = `ws://localhost:8000/api/classroom/ws/${roomId}`;
    const ws = new WebSocket(wsUrl);
    websocketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      
      // Send user info
      const userId = `user_${Math.random().toString(36).substr(2, 9)}`;
      const username = localStorage.getItem('username') || prompt('Enter your name:') || 'Anonymous';
      
      setCurrentUser(userId);
      
      ws.send(JSON.stringify({
        user_id: userId,
        username: username,
        role: 'student'
      }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const handleWebSocketMessage = async (message: any) => {
    switch (message.type) {
      case 'room_state':
        setParticipants(message.participants);
        setIsHost(message.room_info.is_host);
        break;

      case 'participant_joined':
        // Create peer connection for new participant
        if (message.user_id !== currentUser) {
          createPeerConnection(message.user_id, true);
        }
        break;

      case 'participant_left':
        removePeerConnection(message.user_id);
        setParticipants(prev => prev.filter(p => p.user_id !== message.user_id));
        break;

      case 'webrtc_offer':
        await handleOffer(message);
        break;

      case 'webrtc_answer':
        await handleAnswer(message);
        break;

      case 'webrtc_ice_candidate':
        await handleIceCandidate(message);
        break;

      case 'chat_message':
        setChatMessages(prev => [...prev, message]);
        break;

      case 'participant_media_change':
        updateParticipantMedia(message);
        break;

      case 'hand_raised':
        updateParticipantHandRaised(message);
        break;

      case 'room_closed':
        alert(message.message);
        navigate('/dashboard');
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const createPeerConnection = async (userId: string, createOffer = false) => {
    const pc = new RTCPeerConnection(rtcConfiguration);
    peerConnectionsRef.current[userId] = pc;

    // Add local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => {
        pc.addTrack(track, localStreamRef.current!);
      });
    }

    // Handle remote stream
    pc.ontrack = (event) => {
      const remoteVideo = document.getElementById(`video-${userId}`) as HTMLVideoElement;
      if (remoteVideo) {
        remoteVideo.srcObject = event.streams[0];
      }
    };

    // Handle ICE candidates
    pc.onicecandidate = (event) => {
      if (event.candidate && websocketRef.current) {
        websocketRef.current.send(JSON.stringify({
          type: 'webrtc_ice_candidate',
          target_user: userId,
          candidate: event.candidate
        }));
      }
    };

    // Create offer if initiator
    if (createOffer) {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        
        if (websocketRef.current) {
          websocketRef.current.send(JSON.stringify({
            type: 'webrtc_offer',
            target_user: userId,
            offer: offer
          }));
        }
      } catch (error) {
        console.error('Error creating offer:', error);
      }
    }
  };

  const handleOffer = async (message: any) => {
    const pc = peerConnectionsRef.current[message.from_user];
    if (!pc) {
      await createPeerConnection(message.from_user, false);
    }

    try {
      const pc = peerConnectionsRef.current[message.from_user];
      await pc.setRemoteDescription(message.offer);
      
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      
      if (websocketRef.current) {
        websocketRef.current.send(JSON.stringify({
          type: 'webrtc_answer',
          target_user: message.from_user,
          answer: answer
        }));
      }
    } catch (error) {
      console.error('Error handling offer:', error);
    }
  };

  const handleAnswer = async (message: any) => {
    const pc = peerConnectionsRef.current[message.from_user];
    if (pc) {
      try {
        await pc.setRemoteDescription(message.answer);
      } catch (error) {
        console.error('Error handling answer:', error);
      }
    }
  };

  const handleIceCandidate = async (message: any) => {
    const pc = peerConnectionsRef.current[message.from_user];
    if (pc) {
      try {
        await pc.addIceCandidate(message.candidate);
      } catch (error) {
        console.error('Error handling ICE candidate:', error);
      }
    }
  };

  const updateParticipantMedia = (message: any) => {
    setParticipants(prev => prev.map(p => 
      p.user_id === message.user_id 
        ? { 
            ...p, 
            video_enabled: message.video_enabled,
            audio_enabled: message.audio_enabled,
            screen_sharing: message.screen_sharing
          }
        : p
    ));
  };

  const updateParticipantHandRaised = (message: any) => {
    setParticipants(prev => prev.map(p => 
      p.user_id === message.user_id 
        ? { ...p, hand_raised: message.hand_raised }
        : p
    ));
  };

  const removePeerConnection = (userId: string) => {
    const pc = peerConnectionsRef.current[userId];
    if (pc) {
      pc.close();
      delete peerConnectionsRef.current[userId];
    }
  };

  const toggleVideo = () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoEnabled(videoTrack.enabled);
        
        if (websocketRef.current) {
          websocketRef.current.send(JSON.stringify({
            type: 'media_state_change',
            video_enabled: videoTrack.enabled,
            audio_enabled: isAudioEnabled,
            screen_sharing: isScreenSharing
          }));
        }
      }
    }
  };

  const toggleAudio = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsAudioEnabled(audioTrack.enabled);
        
        if (websocketRef.current) {
          websocketRef.current.send(JSON.stringify({
            type: 'media_state_change',
            video_enabled: isVideoEnabled,
            audio_enabled: audioTrack.enabled,
            screen_sharing: isScreenSharing
          }));
        }
      }
    }
  };

  const toggleScreenShare = async () => {
    if (!isScreenSharing) {
      try {
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true
        });
        
        // Replace video track in all peer connections
        const videoTrack = screenStream.getVideoTracks()[0];
        Object.values(peerConnectionsRef.current).forEach(pc => {
          const sender = pc.getSenders().find(s => 
            s.track && s.track.kind === 'video'
          );
          if (sender) {
            sender.replaceTrack(videoTrack);
          }
        });
        
        // Update local video
        if (localVideoRef.current) {
          localVideoRef.current.srcObject = screenStream;
        }
        
        setIsScreenSharing(true);
        
        // Handle screen share end
        videoTrack.onended = () => {
          stopScreenShare();
        };
        
      } catch (error) {
        console.error('Error starting screen share:', error);
      }
    } else {
      stopScreenShare();
    }
  };

  const stopScreenShare = async () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      
      // Replace with camera stream in all peer connections
      Object.values(peerConnectionsRef.current).forEach(pc => {
        const sender = pc.getSenders().find(s => 
          s.track && s.track.kind === 'video'
        );
        if (sender && videoTrack) {
          sender.replaceTrack(videoTrack);
        }
      });
      
      // Update local video
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = localStreamRef.current;
      }
    }
    
    setIsScreenSharing(false);
  };

  const toggleHandRaise = () => {
    const newHandRaised = !handRaised;
    setHandRaised(newHandRaised);
    
    if (websocketRef.current) {
      websocketRef.current.send(JSON.stringify({
        type: 'raise_hand'
      }));
    }
  };

  const sendChatMessage = () => {
    if (chatMessage.trim() && websocketRef.current) {
      websocketRef.current.send(JSON.stringify({
        type: 'chat_message',
        message: chatMessage.trim()
      }));
      setChatMessage('');
    }
  };

  const leaveRoom = () => {
    if (window.confirm('Are you sure you want to leave the classroom?')) {
      cleanup();
      navigate('/dashboard');
    }
  };

  const cleanup = () => {
    // Stop local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    // Close peer connections
    Object.values(peerConnectionsRef.current).forEach(pc => pc.close());
    
    // Close WebSocket
    if (websocketRef.current) {
      websocketRef.current.close();
    }
  };

  const getEmotionColor = (emotion: string) => {
    switch (emotion) {
      case 'engaged': return '#4caf50';
      case 'confused': return '#ff9800';
      case 'bored': return '#f44336';
      case 'frustrated': return '#e91e63';
      default: return '#9e9e9e';
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#1a1a1a' }}>
      {/* Header */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 2, 
        bgcolor: '#2c2c2c',
        color: 'white'
      }}>
        <Typography variant="h6">
          Virtual Classroom - {roomId}
          <Chip 
            label={isConnected ? 'Connected' : 'Disconnected'} 
            color={isConnected ? 'success' : 'error'}
            size="small"
            sx={{ ml: 2 }}
          />
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip 
            icon={<Group />}
            label={`${participants.length} participants`}
            variant="outlined"
            sx={{ color: 'white' }}
          />
          
          <IconButton color="inherit" onClick={() => setChatOpen(true)}>
            <Chat />
          </IconButton>
          
          <IconButton color="inherit" onClick={toggleFullscreen}>
            {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
          </IconButton>
        </Box>
      </Box>

      {/* Video Grid */}
      <Box sx={{ flex: 1, p: 2 }}>
        <Grid container spacing={2} sx={{ height: '100%' }}>
          {/* Local Video */}
          <Grid item xs={12} sm={6} md={4} lg={3}>
            <Card sx={{ height: '100%', bgcolor: '#333', position: 'relative' }}>
              <video
                ref={localVideoRef}
                autoPlay
                muted
                playsInline
                style={{ 
                  width: '100%', 
                  height: '100%', 
                  objectFit: 'cover',
                  borderRadius: '8px'
                }}
              />
              <Box sx={{ 
                position: 'absolute', 
                bottom: 8, 
                left: 8, 
                bgcolor: 'rgba(0,0,0,0.7)', 
                color: 'white', 
                px: 1, 
                py: 0.5, 
                borderRadius: 1,
                fontSize: '0.8rem'
              }}>
                You {!isVideoEnabled && '(Video Off)'}
              </Box>
              {!isVideoEnabled && (
                <Box sx={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  right: 0, 
                  bottom: 0, 
                  bgcolor: '#444', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center' 
                }}>
                  <VideocamOff sx={{ fontSize: 48, color: '#666' }} />
                </Box>
              )}
            </Card>
          </Grid>

          {/* Remote Videos */}
          {participants.filter(p => p.user_id !== currentUser).map((participant) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={participant.user_id}>
              <Card sx={{ height: '100%', bgcolor: '#333', position: 'relative' }}>
                <video
                  id={`video-${participant.user_id}`}
                  autoPlay
                  playsInline
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover',
                    borderRadius: '8px'
                  }}
                />
                <Box sx={{ 
                  position: 'absolute', 
                  bottom: 8, 
                  left: 8, 
                  bgcolor: 'rgba(0,0,0,0.7)', 
                  color: 'white', 
                  px: 1, 
                  py: 0.5, 
                  borderRadius: 1,
                  fontSize: '0.8rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1
                }}>
                  {participant.username}
                  {!participant.video_enabled && <VideocamOff fontSize="small" />}
                  {!participant.audio_enabled && <MicOff fontSize="small" />}
                  {participant.hand_raised && <PanTool fontSize="small" color="warning" />}
                </Box>
                
                {/* Emotion indicator */}
                {participant.emotion_data && (
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 8, 
                    right: 8, 
                    width: 16, 
                    height: 16, 
                    borderRadius: '50%', 
                    bgcolor: getEmotionColor(participant.emotion_data.learning_state),
                    border: '2px solid white'
                  }} />
                )}
                
                {!participant.video_enabled && (
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0, 
                    bottom: 0, 
                    bgcolor: '#444', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center' 
                  }}>
                    <VideocamOff sx={{ fontSize: 48, color: '#666' }} />
                  </Box>
                )}
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Controls */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        gap: 2, 
        p: 2, 
        bgcolor: '#2c2c2c' 
      }}>
        <Tooltip title={isVideoEnabled ? 'Turn off video' : 'Turn on video'}>
          <IconButton 
            onClick={toggleVideo}
            color={isVideoEnabled ? 'primary' : 'error'}
            sx={{ 
              bgcolor: isVideoEnabled ? 'primary.main' : 'error.main',
              color: 'white',
              '&:hover': { 
                bgcolor: isVideoEnabled ? 'primary.dark' : 'error.dark' 
              }
            }}
          >
            {isVideoEnabled ? <Videocam /> : <VideocamOff />}
          </IconButton>
        </Tooltip>

        <Tooltip title={isAudioEnabled ? 'Mute' : 'Unmute'}>
          <IconButton 
            onClick={toggleAudio}
            color={isAudioEnabled ? 'primary' : 'error'}
            sx={{ 
              bgcolor: isAudioEnabled ? 'primary.main' : 'error.main',
              color: 'white',
              '&:hover': { 
                bgcolor: isAudioEnabled ? 'primary.dark' : 'error.dark' 
              }
            }}
          >
            {isAudioEnabled ? <Mic /> : <MicOff />}
          </IconButton>
        </Tooltip>

        <Tooltip title={isScreenSharing ? 'Stop sharing' : 'Share screen'}>
          <IconButton 
            onClick={toggleScreenShare}
            color={isScreenSharing ? 'secondary' : 'default'}
            sx={{ 
              bgcolor: isScreenSharing ? 'secondary.main' : 'grey.600',
              color: 'white',
              '&:hover': { 
                bgcolor: isScreenSharing ? 'secondary.dark' : 'grey.700' 
              }
            }}
          >
            {isScreenSharing ? <StopScreenShare /> : <ScreenShare />}
          </IconButton>
        </Tooltip>

        <Tooltip title={handRaised ? 'Lower hand' : 'Raise hand'}>
          <IconButton 
            onClick={toggleHandRaise}
            color={handRaised ? 'warning' : 'default'}
            sx={{ 
              bgcolor: handRaised ? 'warning.main' : 'grey.600',
              color: 'white',
              '&:hover': { 
                bgcolor: handRaised ? 'warning.dark' : 'grey.700' 
              }
            }}
          >
            <PanTool />
          </IconButton>
        </Tooltip>

        <Tooltip title="Leave classroom">
          <IconButton 
            onClick={leaveRoom}
            sx={{ 
              bgcolor: 'error.main',
              color: 'white',
              '&:hover': { bgcolor: 'error.dark' }
            }}
          >
            <CallEnd />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Chat Dialog */}
      <Dialog 
        open={chatOpen} 
        onClose={() => setChatOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Chat Messages</DialogTitle>
        <DialogContent>
          <Box sx={{ height: 300, overflow: 'auto', mb: 2 }}>
            {chatMessages.map((msg, index) => (
              <Box key={index} sx={{ mb: 1, p: 1, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                <Typography variant="subtitle2" color="primary">
                  {msg.username}
                </Typography>
                <Typography variant="body2">{msg.message}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </Typography>
              </Box>
            ))}
            <div ref={chatEndRef} />
          </Box>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your message..."
            value={chatMessage}
            onChange={(e) => setChatMessage(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                sendChatMessage();
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setChatOpen(false)}>Close</Button>
          <Button onClick={sendChatMessage} variant="contained">Send</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VirtualClassroom;