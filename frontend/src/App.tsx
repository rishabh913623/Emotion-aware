import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Container, AppBar, Toolbar, Typography, Box } from '@mui/material';
import InstructorDashboard from './pages/InstructorDashboard';
import StudentView from './pages/StudentView';
import Login from './pages/Login';
import VirtualClassroom from './pages/VirtualClassroom';
import ReportsPage from './pages/ReportsPage';
import { useAppSelector } from './store/hooks';

const App: React.FC = () => {
  const { isAuthenticated, user } = useAppSelector((state) => state.auth);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Emotion-Aware Virtual Classroom
          </Typography>
          {isAuthenticated && user && (
            <Typography variant="body2">
              Welcome, {user.name} ({user.role})
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          <Route 
            path="/login" 
            element={!isAuthenticated ? <Login /> : <Navigate to="/" />} 
          />
          <Route 
            path="/dashboard" 
            element={
              isAuthenticated && user?.role === 'instructor' ? 
                <InstructorDashboard /> : 
                <Navigate to="/login" />
            } 
          />
          <Route 
            path="/student" 
            element={
              isAuthenticated && user?.role === 'student' ? 
                <StudentView /> : 
                <Navigate to="/login" />
            } 
          />
          <Route 
            path="/classroom/:roomId" 
            element={
              isAuthenticated ? 
                <VirtualClassroom /> : 
                <Navigate to="/login" />
            } 
          />
          <Route 
            path="/reports" 
            element={
              isAuthenticated && user?.role === 'instructor' ? 
                <ReportsPage /> : 
                <Navigate to="/login" />
            } 
          />
          <Route 
            path="/" 
            element={
              isAuthenticated ? 
                (user?.role === 'instructor' ? 
                  <Navigate to="/dashboard" /> : 
                  <Navigate to="/student" />
                ) : 
                <Navigate to="/login" />
            } 
          />
        </Routes>
      </Container>
    </Box>
  );
};

export default App;