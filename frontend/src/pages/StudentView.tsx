import React, { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Avatar,
  Chip,
} from '@mui/material';
import {
  VideoCall,
  Mic,
  MicOff,
  Videocam,
  VideocamOff,
  ScreenShare,
  StopScreenShare,
  Chat,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';

const StudentView: React.FC = () => {
  const dispatch = useAppDispatch();
  const { user } = useAppSelector((state) => state.auth);
  const classroom = useAppSelector((state) => state.classroom);

  const [isVideoEnabled, setIsVideoEnabled] = useState(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('neutral');

  // Mock emotion detection for demo
  useEffect(() => {
    const emotions = ['engaged', 'curious', 'neutral', 'confused', 'bored'];
    const interval = setInterval(() => {
      const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
      setCurrentEmotion(randomEmotion);
    }, 10000); // Change emotion every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const toggleVideo = () => {
    setIsVideoEnabled(!isVideoEnabled);
  };

  const toggleAudio = () => {
    setIsAudioEnabled(!isAudioEnabled);
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Student Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Video/Audio Controls */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3, textAlign: 'center', minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Virtual Classroom
            </Typography>
            
            {/* Video placeholder */}
            <Box
              sx={{
                width: '100%',
                height: 300,
                bgcolor: 'grey.900',
                borderRadius: 2,
                mb: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
              }}
            >
              <Typography variant="h6">
                {isVideoEnabled ? 'Video Active' : 'Video Off'}
              </Typography>
            </Box>

            {/* Controls */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
              <Button
                variant={isAudioEnabled ? 'contained' : 'outlined'}
                onClick={toggleAudio}
                startIcon={isAudioEnabled ? <Mic /> : <MicOff />}
                color={isAudioEnabled ? 'primary' : 'error'}
              >
                {isAudioEnabled ? 'Mute' : 'Unmute'}
              </Button>
              
              <Button
                variant={isVideoEnabled ? 'contained' : 'outlined'}
                onClick={toggleVideo}
                startIcon={isVideoEnabled ? <Videocam /> : <VideocamOff />}
                color={isVideoEnabled ? 'primary' : 'error'}
              >
                {isVideoEnabled ? 'Stop Video' : 'Start Video'}
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<ScreenShare />}
              >
                Share Screen
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Student Info & Emotion Status */}
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ mr: 2 }}>
                  {user?.name?.charAt(0) || 'S'}
                </Avatar>
                <Box>
                  <Typography variant="h6">{user?.name || 'Student'}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {user?.email}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Emotion
              </Typography>
              <Box sx={{ textAlign: 'center' }}>
                <Chip
                  label={currentEmotion}
                  color={currentEmotion === 'engaged' || currentEmotion === 'curious' ? 'success' : 
                         currentEmotion === 'confused' ? 'warning' : 'default'}
                  sx={{ fontSize: '1rem', p: 2, textTransform: 'capitalize' }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Automatically detected via multimodal AI
                </Typography>
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Privacy Settings
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Your emotional state is being analyzed to improve learning experience.
              </Typography>
              <Button variant="outlined" size="small" fullWidth>
                Manage Privacy
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Chat Area */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 200 }}>
            <Typography variant="h6" gutterBottom>
              Class Chat
            </Typography>
            <Box
              sx={{
                height: 120,
                overflow: 'auto',
                bgcolor: 'grey.50',
                p: 1,
                borderRadius: 1,
                mb: 1,
              }}
            >
              <Typography variant="body2" color="text.secondary">
                Chat messages will appear here...
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Box sx={{ flexGrow: 1 }}>
                <input
                  placeholder="Type a message..."
                  style={{
                    width: '100%',
                    padding: '8px',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                  }}
                />
              </Box>
              <Button variant="contained" startIcon={<Chat />}>
                Send
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default StudentView;