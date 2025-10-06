import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Avatar,
  Typography,
  Chip,
  Box,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Person,
  TrendingUp,
  TrendingDown,
  Remove,
  QuestionMark,
  SentimentVeryDissatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  SentimentSatisfied,
  SentimentVerySatisfied,
} from '@mui/icons-material';

interface EmotionData {
  facial_emotion?: string;
  audio_emotion?: string;
  text_sentiment?: string;
  learning_state: string;
  confidence: number;
  timestamp: string;
}

interface Student {
  id: string;
  name?: string;
  current_emotion: EmotionData;
  last_updated: string;
}

interface Props {
  students: { [studentId: string]: Student };
}

const EMOTION_COLORS = {
  engaged: '#4caf50',
  curious: '#2196f3',
  neutral: '#9e9e9e',
  confused: '#ff9800',
  bored: '#f44336',
  frustrated: '#d32f2f',
};

const EMOTION_ICONS = {
  engaged: SentimentVerySatisfied,
  curious: SentimentSatisfied,
  neutral: SentimentNeutral,
  confused: QuestionMark,
  bored: SentimentDissatisfied,
  frustrated: SentimentVeryDissatisfied,
};

const StudentEmotionGrid: React.FC<Props> = ({ students }) => {
  const studentList = Object.values(students);

  const getEmotionIcon = (emotion: string) => {
    const IconComponent = EMOTION_ICONS[emotion as keyof typeof EMOTION_ICONS] || SentimentNeutral;
    return <IconComponent />;
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return { level: 'High', color: 'success' as const };
    if (confidence >= 0.6) return { level: 'Medium', color: 'warning' as const };
    return { level: 'Low', color: 'error' as const };
  };

  const getTimeSinceUpdate = (timestamp: string) => {
    const now = new Date();
    const updateTime = new Date(timestamp);
    const diffSeconds = Math.floor((now.getTime() - updateTime.getTime()) / 1000);
    
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    return `${Math.floor(diffSeconds / 3600)}h ago`;
  };

  if (studentList.length === 0) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: 300,
        color: 'text.secondary' 
      }}>
        <Typography variant="body2">No students in session</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
      <Grid container spacing={2}>
        {studentList.map((student) => {
          const emotion = student.current_emotion;
          const confidenceData = getConfidenceLevel(emotion.confidence);
          
          return (
            <Grid item xs={12} sm={6} md={4} key={student.id}>
              <Card 
                sx={{ 
                  borderLeft: 4,
                  borderColor: EMOTION_COLORS[emotion.learning_state as keyof typeof EMOTION_COLORS] || '#9e9e9e',
                  '&:hover': {
                    boxShadow: 3,
                  }
                }}
              >
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  {/* Student Header */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
                    <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
                      <Person fontSize="small" />
                    </Avatar>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle2" noWrap>
                        {student.name || `Student ${student.id.slice(-4)}`}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {getTimeSinceUpdate(emotion.timestamp)}
                      </Typography>
                    </Box>
                  </Box>

                  {/* Current Emotion */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Tooltip title={`Learning State: ${emotion.learning_state}`}>
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        color: EMOTION_COLORS[emotion.learning_state as keyof typeof EMOTION_COLORS],
                        mr: 1
                      }}>
                        {getEmotionIcon(emotion.learning_state)}
                      </Box>
                    </Tooltip>
                    <Typography variant="body2" sx={{ flex: 1, textTransform: 'capitalize' }}>
                      {emotion.learning_state}
                    </Typography>
                    <Chip
                      label={confidenceData.level}
                      color={confidenceData.color}
                      size="small"
                      variant="outlined"
                    />
                  </Box>

                  {/* Emotion Details */}
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    {emotion.facial_emotion && (
                      <Typography variant="caption" color="text.secondary">
                        Face: {emotion.facial_emotion}
                      </Typography>
                    )}
                    {emotion.audio_emotion && (
                      <Typography variant="caption" color="text.secondary">
                        Voice: {emotion.audio_emotion}
                      </Typography>
                    )}
                    {emotion.text_sentiment && (
                      <Typography variant="caption" color="text.secondary">
                        Text: {emotion.text_sentiment}
                      </Typography>
                    )}
                  </Box>

                  {/* Confidence Bar */}
                  <Box sx={{ mt: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        Confidence
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {Math.round(emotion.confidence * 100)}%
                      </Typography>
                    </Box>
                    <Box 
                      sx={{ 
                        width: '100%', 
                        height: 4, 
                        bgcolor: 'grey.300', 
                        borderRadius: 2,
                        mt: 0.5
                      }}
                    >
                      <Box
                        sx={{
                          width: `${emotion.confidence * 100}%`,
                          height: '100%',
                          bgcolor: confidenceData.color === 'success' ? 'success.main' : 
                                   confidenceData.color === 'warning' ? 'warning.main' : 'error.main',
                          borderRadius: 2,
                        }}
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default StudentEmotionGrid;