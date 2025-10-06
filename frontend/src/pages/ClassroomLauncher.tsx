import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Alert,
} from '@mui/material';
import {
  Add,
  VideoCall,
  Group,
  Delete,
  Launch,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface Room {
  room_id: string;
  host_id: string;
  participant_count: number;
  created_at: string;
  is_host: boolean;
}

const ClassroomLauncher: React.FC = () => {
  const navigate = useNavigate();
  const [rooms, setRooms] = useState<Room[]>([]);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [joinDialogOpen, setJoinDialogOpen] = useState(false);
  const [roomName, setRoomName] = useState('');
  const [joinRoomId, setJoinRoomId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadRooms();
  }, []);

  const loadRooms = async () => {
    try {
      const response = await fetch('/api/classroom/rooms', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setRooms(data.rooms);
      } else {
        setError('Failed to load rooms');
      }
    } catch (error) {
      setError('Network error loading rooms');
    }
  };

  const createRoom = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/classroom/create-room', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          room_name: roomName || 'New Classroom'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setCreateDialogOpen(false);
        setRoomName('');
        
        // Join the created room
        navigate(`/classroom/${data.room_id}`);
      } else {
        setError('Failed to create room');
      }
    } catch (error) {
      setError('Network error creating room');
    } finally {
      setLoading(false);
    }
  };

  const joinRoom = (roomId: string) => {
    navigate(`/classroom/${roomId}`);
  };

  const joinRoomById = () => {
    if (joinRoomId.trim()) {
      navigate(`/classroom/${joinRoomId.trim()}`);
      setJoinDialogOpen(false);
      setJoinRoomId('');
    }
  };

  const deleteRoom = async (roomId: string) => {
    if (window.confirm('Are you sure you want to delete this room?')) {
      try {
        const response = await fetch(`/api/classroom/room/${roomId}`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (response.ok) {
          loadRooms();
        } else {
          setError('Failed to delete room');
        }
      } catch (error) {
        setError('Network error deleting room');
      }
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Virtual Classrooms
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setCreateDialogOpen(true)}
          >
            Create Room
          </Button>
          <Button
            variant="outlined"
            startIcon={<VideoCall />}
            onClick={() => setJoinDialogOpen(true)}
          >
            Join Room
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Quick Access */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Quick Start
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Start teaching or learning right away
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<VideoCall />}
                  onClick={() => setCreateDialogOpen(true)}
                >
                  Start Instant Meeting
                </Button>
                <Button
                  variant="text"
                  fullWidth
                  startIcon={<Launch />}
                  onClick={() => setJoinDialogOpen(true)}
                >
                  Join with Room ID
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Room List */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Your Rooms
                <Chip 
                  label={`${rooms.length} active`}
                  size="small"
                  sx={{ ml: 2 }}
                />
              </Typography>
              
              {rooms.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                  <VideoCall sx={{ fontSize: 48, mb: 2 }} />
                  <Typography variant="body1">
                    No active rooms. Create one to get started!
                  </Typography>
                </Box>
              ) : (
                <List>
                  {rooms.map((room) => (
                    <ListItem key={room.room_id} divider>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle1">
                              Room {room.room_id.slice(0, 8)}...
                            </Typography>
                            {room.is_host && (
                              <Chip label="Host" size="small" color="primary" />
                            )}
                          </Box>
                        }
                        secondary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Group fontSize="small" />
                              <Typography variant="body2">
                                {room.participant_count} participants
                              </Typography>
                            </Box>
                            <Typography variant="body2" color="text.secondary">
                              Created {new Date(room.created_at).toLocaleString()}
                            </Typography>
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Button
                            size="small"
                            variant="contained"
                            onClick={() => joinRoom(room.room_id)}
                          >
                            Join
                          </Button>
                          {room.is_host && (
                            <IconButton
                              size="small"
                              onClick={() => deleteRoom(room.room_id)}
                              color="error"
                            >
                              <Delete />
                            </IconButton>
                          )}
                        </Box>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Create Room Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)}>
        <DialogTitle>Create New Classroom</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Room Name (Optional)"
            type="text"
            fullWidth
            variant="outlined"
            value={roomName}
            onChange={(e) => setRoomName(e.target.value)}
            placeholder="e.g., CS101 Lecture, Math Study Group"
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            You'll be able to invite participants after creating the room.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={createRoom} 
            variant="contained"
            disabled={loading}
          >
            {loading ? 'Creating...' : 'Create Room'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Join Room Dialog */}
      <Dialog open={joinDialogOpen} onClose={() => setJoinDialogOpen(false)}>
        <DialogTitle>Join Classroom</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Room ID"
            type="text"
            fullWidth
            variant="outlined"
            value={joinRoomId}
            onChange={(e) => setJoinRoomId(e.target.value)}
            placeholder="Enter the room ID provided by your instructor"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setJoinDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={joinRoomById} 
            variant="contained"
            disabled={!joinRoomId.trim()}
          >
            Join Room
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ClassroomLauncher;