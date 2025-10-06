import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Alert,
  Chip,
  LinearProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Settings,
  VideoCall,
  People,
  Analytics,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import {
  setCurrentClass,
  updateClassState,
  updateStudentEmotion,
  updateAlerts,
  setConnectionStatus,
  clearDashboard,
} from '../store/slices/dashboardSlice';
import ClassMoodChart from '../components/ClassMoodChart';
import StudentEmotionGrid from '../components/StudentEmotionGrid';
import AlertPanel from '../components/AlertPanel';
import RealTimeChart from '../components/RealTimeChart';
import { dashboardService } from '../services/dashboardService';

const InstructorDashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const {
    currentClassId,
    classMood,
    students,
    totalStudents,
    alerts,
    lastUpdated,
    isConnected,
    loading,
  } = useAppSelector((state) => state.dashboard);
  const { user } = useAppSelector((state) => state.auth);

  const [isSessionActive, setIsSessionActive] = useState(false);
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [newClassId, setNewClassId] = useState('');

  useEffect(() => {
    // Initialize dashboard service
    dashboardService.setDispatch(dispatch);
    
    return () => {
      // Cleanup on unmount
      dashboardService.disconnect();
      dispatch(clearDashboard());
    };
  }, [dispatch]);

  const handleStartSession = async () => {
    if (!newClassId.trim()) return;
    
    try {
      await dashboardService.startClassSession(newClassId, user?.id || '');
      dispatch(setCurrentClass(newClassId));
      await dashboardService.connect(newClassId, user?.id || '');
      setIsSessionActive(true);
      setStartDialogOpen(false);
      setNewClassId('');
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  const handleEndSession = async () => {
    if (!currentClassId || !user?.id) return;
    
    try {
      await dashboardService.endClassSession(currentClassId, user.id);
      dashboardService.disconnect();
      dispatch(clearDashboard());
      setIsSessionActive(false);
    } catch (error) {
      console.error('Failed to end session:', error);
    }
  };

  const handleRefreshData = async () => {
    if (!currentClassId) return;
    
    try {
      const [classState, alertsData] = await Promise.all([
        dashboardService.getCurrentClassState(currentClassId),
        dashboardService.getClassAlerts(currentClassId),
      ]);
      
      dispatch(updateClassState(classState));
      dispatch(updateAlerts(alertsData.alerts));
    } catch (error) {
      console.error('Failed to refresh data:', error);
    }
  };

  const getEngagementLevel = () => {
    const engaged = classMood.engaged + classMood.curious;
    const disengaged = classMood.bored + classMood.frustrated;
    
    if (engaged > 60) return { level: 'High', color: 'success' as const };
    if (engaged > 30) return { level: 'Medium', color: 'warning' as const };
    return { level: 'Low', color: 'error' as const };
  };

  const engagementLevel = getEngagementLevel();

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Instructor Dashboard
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          {!isSessionActive ? (
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={() => setStartDialogOpen(true)}
              size="large"
            >
              Start Class Session
            </Button>
          ) : (
            <>
              <Button
                variant="outlined"
                startIcon={<VideoCall />}
                onClick={() => {/* Navigate to virtual classroom */}}
              >
                Join Virtual Classroom
              </Button>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={handleRefreshData}
              >
                Refresh
              </Button>
              <Button
                variant="contained"
                color="error"
                startIcon={<Stop />}
                onClick={handleEndSession}
              >
                End Session
              </Button>
            </>
          )}
        </Box>
      </Box>

      {/* Connection Status */}
      {isSessionActive && (
        <Box sx={{ mb: 2 }}>
          <Alert 
            severity={isConnected ? 'success' : 'warning'}
            action={
              <Chip
                label={isConnected ? 'Connected' : 'Disconnected'}
                color={isConnected ? 'success' : 'warning'}
                size="small"
              />
            }
          >
            Class: {currentClassId} | Students: {totalStudents} | 
            Last Updated: {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : 'Never'}
          </Alert>
        </Box>
      )}

      {isSessionActive ? (
        <>
          {/* Main Statistics Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Students
                  </Typography>
                  <Typography variant="h4" component="div">
                    {totalStudents}
                  </Typography>
                  <Typography variant="body2">
                    Active in session
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Engagement Level
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="h4" component="div">
                      {engagementLevel.level}
                    </Typography>
                    <Chip 
                      label={`${(classMood.engaged + classMood.curious).toFixed(1)}%`}
                      color={engagementLevel.color}
                      size="small"
                    />
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={classMood.engaged + classMood.curious}
                    color={engagementLevel.color}
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Confusion Level
                  </Typography>
                  <Typography variant="h4" component="div" color={classMood.confused > 30 ? 'error.main' : 'text.primary'}>
                    {classMood.confused.toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    {classMood.confused > 30 ? 'Needs attention' : 'Normal range'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Active Alerts
                  </Typography>
                  <Typography variant="h4" component="div" color={alerts.length > 0 ? 'warning.main' : 'text.primary'}>
                    {alerts.length}
                  </Typography>
                  <Typography variant="body2">
                    {alerts.length > 0 ? 'Require attention' : 'All clear'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Main Content Grid */}
          <Grid container spacing={3}>
            {/* Class Mood Chart */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, height: 400 }}>
                <Typography variant="h6" gutterBottom>
                  Class Emotional State
                </Typography>
                <ClassMoodChart classMood={classMood} />
              </Paper>
            </Grid>

            {/* Real-time Emotion Timeline */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, height: 400 }}>
                <Typography variant="h6" gutterBottom>
                  Engagement Timeline
                </Typography>
                <RealTimeChart classId={currentClassId} />
              </Paper>
            </Grid>

            {/* Alert Panel */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: 500 }}>
                <Typography variant="h6" gutterBottom>
                  Real-time Alerts
                </Typography>
                <AlertPanel alerts={alerts} />
              </Paper>
            </Grid>

            {/* Student Grid */}
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2, height: 500 }}>
                <Typography variant="h6" gutterBottom>
                  Student Emotion States ({Object.keys(students).length})
                </Typography>
                <StudentEmotionGrid students={students} />
              </Paper>
            </Grid>
          </Grid>
        </>
      ) : (
        /* Welcome Screen */
        <Box sx={{ textAlign: 'center', mt: 8 }}>
          <Analytics sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h4" gutterBottom>
            Welcome to your Emotion-Aware Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
            Start a class session to begin monitoring student engagement and emotions in real-time.
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrow />}
            onClick={() => setStartDialogOpen(true)}
          >
            Start Your First Class
          </Button>
        </Box>
      )}

      {/* Start Session Dialog */}
      <Dialog open={startDialogOpen} onClose={() => setStartDialogOpen(false)}>
        <DialogTitle>Start New Class Session</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Class ID"
            placeholder="e.g., CS101-Fall2024"
            fullWidth
            variant="outlined"
            value={newClassId}
            onChange={(e) => setNewClassId(e.target.value)}
            sx={{ mt: 2 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Enter a unique identifier for this class session
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleStartSession} variant="contained" disabled={!newClassId.trim()}>
            Start Session
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default InstructorDashboard;