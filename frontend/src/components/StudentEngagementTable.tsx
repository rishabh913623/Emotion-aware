import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Avatar,
  Box,
  LinearProgress,
  Typography,
  Tooltip,
  IconButton,
  TableSortLabel,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Remove,
  Person,
  AccessTime,
  Psychology,
} from '@mui/icons-material';
import { visuallyHidden } from '@mui/utils';

interface EmotionDistribution {
  [emotion: string]: {
    count: number;
    percentage: number;
  };
}

interface StudentReport {
  user_id: string;
  total_records: number;
  avg_confidence: number;
  session_duration_minutes: number;
  engagement_score: number;
  emotion_distribution: EmotionDistribution;
  first_activity: string | null;
  last_activity: string | null;
}

interface EngagementData {
  student_reports: StudentReport[];
  class_summary: {
    total_students: number;
    avg_engagement: number;
    total_emotion_records: number;
  };
}

interface Props {
  data: EngagementData | null;
}

type Order = 'asc' | 'desc';
type OrderBy = 'engagement_score' | 'session_duration_minutes' | 'avg_confidence' | 'total_records';

const StudentEngagementTable: React.FC<Props> = ({ data }) => {
  const [order, setOrder] = React.useState<Order>('desc');
  const [orderBy, setOrderBy] = React.useState<OrderBy>('engagement_score');

  if (!data || !data.student_reports || data.student_reports.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Person sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No student engagement data available
        </Typography>
      </Box>
    );
  }

  const handleRequestSort = (property: OrderBy) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const sortedStudents = React.useMemo(() => {
    return [...data.student_reports].sort((a, b) => {
      let aValue = a[orderBy];
      let bValue = b[orderBy];
      
      if (order === 'desc') {
        return bValue < aValue ? -1 : bValue > aValue ? 1 : 0;
      } else {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      }
    });
  }, [data.student_reports, order, orderBy]);

  const getEngagementLevel = (score: number) => {
    if (score >= 80) return { level: 'High', color: 'success' as const, icon: TrendingUp };
    if (score >= 60) return { level: 'Medium', color: 'warning' as const, icon: Remove };
    return { level: 'Low', color: 'error' as const, icon: TrendingDown };
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) {
      return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
  };

  const getDominantEmotion = (distribution: EmotionDistribution) => {
    let maxEmotion = 'neutral';
    let maxPercentage = 0;
    
    Object.entries(distribution).forEach(([emotion, data]) => {
      if (data.percentage > maxPercentage) {
        maxEmotion = emotion;
        maxPercentage = data.percentage;
      }
    });
    
    return { emotion: maxEmotion, percentage: maxPercentage };
  };

  const getEmotionChip = (emotion: string, percentage: number) => {
    const colors: Record<string, any> = {
      engaged: 'success',
      curious: 'info',
      neutral: 'default',
      confused: 'warning',
      bored: 'error',
      frustrated: 'error',
    };

    return (
      <Chip
        label={`${emotion} (${percentage.toFixed(1)}%)`}
        color={colors[emotion] || 'default'}
        size="small"
        variant="outlined"
      />
    );
  };

  const createSortHandler = (property: OrderBy) => {
    return () => handleRequestSort(property);
  };

  return (
    <Box>
      {/* Summary Cards */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
          <Typography variant="h6" color="primary">
            {data.class_summary.total_students}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Total Students
          </Typography>
        </Paper>
        
        <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
          <Typography variant="h6" color="success.main">
            {data.class_summary.avg_engagement.toFixed(1)}%
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Average Engagement
          </Typography>
        </Paper>
        
        <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
          <Typography variant="h6" color="info.main">
            {data.class_summary.total_emotion_records.toLocaleString()}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Emotion Records
          </Typography>
        </Paper>
      </Box>

      {/* Student Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Student</TableCell>
              
              <TableCell align="right" sortDirection={orderBy === 'engagement_score' ? order : false}>
                <TableSortLabel
                  active={orderBy === 'engagement_score'}
                  direction={orderBy === 'engagement_score' ? order : 'asc'}
                  onClick={createSortHandler('engagement_score')}
                >
                  Engagement Score
                  {orderBy === 'engagement_score' ? (
                    <Box component="span" sx={visuallyHidden}>
                      {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                    </Box>
                  ) : null}
                </TableSortLabel>
              </TableCell>
              
              <TableCell align="right" sortDirection={orderBy === 'session_duration_minutes' ? order : false}>
                <TableSortLabel
                  active={orderBy === 'session_duration_minutes'}
                  direction={orderBy === 'session_duration_minutes' ? order : 'asc'}
                  onClick={createSortHandler('session_duration_minutes')}
                >
                  <AccessTime sx={{ mr: 0.5, fontSize: 16 }} />
                  Duration
                  {orderBy === 'session_duration_minutes' ? (
                    <Box component="span" sx={visuallyHidden}>
                      {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                    </Box>
                  ) : null}
                </TableSortLabel>
              </TableCell>
              
              <TableCell align="right" sortDirection={orderBy === 'avg_confidence' ? order : false}>
                <TableSortLabel
                  active={orderBy === 'avg_confidence'}
                  direction={orderBy === 'avg_confidence' ? order : 'asc'}
                  onClick={createSortHandler('avg_confidence')}
                >
                  <Psychology sx={{ mr: 0.5, fontSize: 16 }} />
                  Confidence
                  {orderBy === 'avg_confidence' ? (
                    <Box component="span" sx={visuallyHidden}>
                      {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                    </Box>
                  ) : null}
                </TableSortLabel>
              </TableCell>
              
              <TableCell>Dominant Emotion</TableCell>
              
              <TableCell align="right" sortDirection={orderBy === 'total_records' ? order : false}>
                <TableSortLabel
                  active={orderBy === 'total_records'}
                  direction={orderBy === 'total_records' ? order : 'asc'}
                  onClick={createSortHandler('total_records')}
                >
                  Data Points
                  {orderBy === 'total_records' ? (
                    <Box component="span" sx={visuallyHidden}>
                      {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                    </Box>
                  ) : null}
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>
          
          <TableBody>
            {sortedStudents.map((student) => {
              const engagementLevel = getEngagementLevel(student.engagement_score);
              const dominantEmotion = getDominantEmotion(student.emotion_distribution);
              const EngagementIcon = engagementLevel.icon;
              
              return (
                <TableRow key={student.user_id} hover>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Avatar sx={{ width: 32, height: 32, mr: 2, bgcolor: 'primary.main' }}>
                        <Person fontSize="small" />
                      </Avatar>
                      <Box>
                        <Typography variant="subtitle2">
                          Student {student.user_id.slice(-4)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          ID: {student.user_id.slice(-8)}
                        </Typography>
                      </Box>
                    </Box>
                  </TableCell>
                  
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                      <Box sx={{ width: 60, mr: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={student.engagement_score}
                          color={engagementLevel.color}
                          sx={{ height: 6, borderRadius: 3 }}
                        />
                      </Box>
                      <Typography variant="body2" sx={{ minWidth: 45 }}>
                        {student.engagement_score.toFixed(1)}%
                      </Typography>
                      <Tooltip title={`${engagementLevel.level} Engagement`}>
                        <EngagementIcon 
                          fontSize="small" 
                          color={engagementLevel.color}
                          sx={{ ml: 0.5 }}
                        />
                      </Tooltip>
                    </Box>
                  </TableCell>
                  
                  <TableCell align="right">
                    <Typography variant="body2">
                      {formatDuration(student.session_duration_minutes)}
                    </Typography>
                  </TableCell>
                  
                  <TableCell align="right">
                    <Chip
                      label={student.avg_confidence.toFixed(3)}
                      color={getConfidenceColor(student.avg_confidence)}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  
                  <TableCell>
                    {getEmotionChip(dominantEmotion.emotion, dominantEmotion.percentage)}
                  </TableCell>
                  
                  <TableCell align="right">
                    <Typography variant="body2">
                      {student.total_records.toLocaleString()}
                    </Typography>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          Engagement score calculated based on positive emotions (engaged, curious) vs. negative emotions (bored, frustrated)
        </Typography>
      </Box>
    </Box>
  );
};

export default StudentEngagementTable;