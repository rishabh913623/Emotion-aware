import React from 'react';
import {
  Box,
  Alert,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  Stack,
} from '@mui/material';
import {
  Warning,
  Error,
  Info,
  Close,
  Lightbulb,
  School,
  Person,
} from '@mui/icons-material';

interface Alert {
  type: string;
  severity: 'info' | 'warning' | 'urgent';
  message: string;
  timestamp: string;
  suggestion: string;
  student_id?: string;
}

interface Props {
  alerts: Alert[];
}

const AlertPanel: React.FC<Props> = ({ alerts }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'urgent': return 'error';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'info';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'urgent': return <Error />;
      case 'warning': return <Warning />;
      case 'info': return <Info />;
      default: return <Info />;
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'high_confusion':
      case 'student_confused':
        return <School />;
      case 'high_boredom':
      case 'low_engagement':
        return <Lightbulb />;
      case 'student_frustrated':
        return <Person />;
      default:
        return <Info />;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  if (alerts.length === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center', 
        justifyContent: 'center',
        height: 200,
        color: 'text.secondary',
        textAlign: 'center'
      }}>
        <Info sx={{ fontSize: 48, mb: 1, color: 'success.main' }} />
        <Typography variant="h6" color="success.main">
          All Clear!
        </Typography>
        <Typography variant="body2">
          No alerts for your class right now
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
      <Stack spacing={2}>
        {alerts.map((alert, index) => (
          <Alert
            key={index}
            severity={getSeverityColor(alert.severity) as any}
            icon={getAlertIcon(alert.type)}
            sx={{
              '& .MuiAlert-message': {
                width: '100%'
              }
            }}
            action={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Chip
                  label={formatTimestamp(alert.timestamp)}
                  size="small"
                  variant="outlined"
                />
              </Box>
            }
          >
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                {alert.message}
              </Typography>
              
              {alert.student_id && (
                <Chip
                  label={`Student: ${alert.student_id.slice(-4)}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                  sx={{ mb: 1 }}
                />
              )}
              
              <Typography variant="body2" color="text.secondary">
                💡 {alert.suggestion}
              </Typography>
            </Box>
          </Alert>
        ))}
      </Stack>
    </Box>
  );
};

export default AlertPanel;