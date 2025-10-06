import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Divider,
  Alert,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Group,
  Psychology,
  Timeline,
  AccessTime,
} from '@mui/icons-material';

interface EmotionDistribution {
  [emotion: string]: {
    count: number;
    percentage: number;
  };
}

interface EngagementMetrics {
  engaged_count: number;
  disengaged_count: number;
  engagement_ratio: number;
  overall_engagement_score: number;
}

interface ActivityPatterns {
  hourly_distribution: { [hour: number]: number };
  peak_hour: number | null;
}

interface AnalyticsData {
  class_id: string;
  overview: {
    total_emotion_records: number;
    unique_students: number;
    data_collection_active: boolean;
  };
  emotion_distribution: EmotionDistribution;
  engagement_metrics: EngagementMetrics;
  activity_patterns: ActivityPatterns;
  generated_at: string;
}

interface Props {
  data: AnalyticsData | null;
}

const AnalyticsOverview: React.FC<Props> = ({ data }) => {
  if (!data) {
    return (
      <Alert severity="info">
        Loading analytics overview...
      </Alert>
    );
  }

  // Prepare emotion distribution data for pie chart
  const emotionChartData = Object.entries(data.emotion_distribution).map(([emotion, info]) => ({
    name: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: info.count,
    percentage: info.percentage,
  }));

  // Prepare hourly activity data for bar chart
  const hourlyActivityData = Object.entries(data.activity_patterns.hourly_distribution)
    .map(([hour, count]) => ({
      hour: `${hour}:00`,
      activity: count,
    }))
    .sort((a, b) => parseInt(a.hour) - parseInt(b.hour));

  const EMOTION_COLORS = {
    engaged: '#4caf50',
    curious: '#2196f3',
    neutral: '#9e9e9e',
    confused: '#ff9800',
    bored: '#f44336',
    frustrated: '#d32f2f',
  };

  const getEmotionColor = (emotion: string) => {
    return EMOTION_COLORS[emotion.toLowerCase() as keyof typeof EMOTION_COLORS] || '#9e9e9e';
  };

  const getEngagementLevel = (score: number) => {
    if (score >= 75) return { level: 'Excellent', color: 'success' as const, icon: TrendingUp };
    if (score >= 50) return { level: 'Good', color: 'info' as const, icon: TrendingUp };
    if (score >= 30) return { level: 'Fair', color: 'warning' as const, icon: TrendingDown };
    return { level: 'Needs Improvement', color: 'error' as const, icon: TrendingDown };
  };

  const engagementLevel = getEngagementLevel(data.engagement_metrics.overall_engagement_score);
  const EngagementIcon = engagementLevel.icon;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper sx={{ p: 1, border: 1, borderColor: 'divider' }}>
          <Typography variant="body2">
            {`${data.name}: ${data.value} (${data.percentage?.toFixed(1)}%)`}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Grid container spacing={3}>
      {/* Overview Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Group color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6" color="primary">
                {data.overview.unique_students}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Active Students
            </Typography>
            <Chip
              size="small"
              label={data.overview.data_collection_active ? 'Live' : 'Inactive'}
              color={data.overview.data_collection_active ? 'success' : 'default'}
              sx={{ mt: 1 }}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Psychology color="secondary" sx={{ mr: 1 }} />
              <Typography variant="h6" color="secondary">
                {data.overview.total_emotion_records.toLocaleString()}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Emotion Data Points
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <EngagementIcon color={engagementLevel.color} sx={{ mr: 1 }} />
              <Typography variant="h6" color={`${engagementLevel.color}.main`}>
                {data.engagement_metrics.overall_engagement_score.toFixed(1)}%
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Overall Engagement
            </Typography>
            <Chip
              size="small"
              label={engagementLevel.level}
              color={engagementLevel.color}
              sx={{ mt: 1 }}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <AccessTime color="info" sx={{ mr: 1 }} />
              <Typography variant="h6" color="info.main">
                {data.activity_patterns.peak_hour !== null ? `${data.activity_patterns.peak_hour}:00` : 'N/A'}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Peak Activity Hour
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Emotion Distribution */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Emotion Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={emotionChartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percentage }) => `${name}: ${percentage?.toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {emotionChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getEmotionColor(entry.name)} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Hourly Activity */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Activity by Hour
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={hourlyActivityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="activity" fill="#2196f3" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Engagement Metrics */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Detailed Engagement Metrics
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Engaged Students
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="h5" color="success.main" sx={{ mr: 1 }}>
                    {data.engagement_metrics.engaged_count}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(data.engagement_metrics.engaged_count / data.overview.unique_students) * 100}
                    color="success"
                    sx={{ flexGrow: 1, mx: 1 }}
                  />
                </Box>
              </Box>
            </Grid>

            <Grid item xs={12} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Disengaged Students
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="h5" color="error.main" sx={{ mr: 1 }}>
                    {data.engagement_metrics.disengaged_count}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(data.engagement_metrics.disengaged_count / data.overview.unique_students) * 100}
                    color="error"
                    sx={{ flexGrow: 1, mx: 1 }}
                  />
                </Box>
              </Box>
            </Grid>

            <Grid item xs={12} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Engagement Ratio
                </Typography>
                <Typography variant="h5" color="info.main">
                  {data.engagement_metrics.engagement_ratio.toFixed(2)}:1
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Engaged to Disengaged
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Overall Score
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="h5" color={`${engagementLevel.color}.main`} sx={{ mr: 1 }}>
                    {data.engagement_metrics.overall_engagement_score.toFixed(1)}%
                  </Typography>
                  <EngagementIcon color={engagementLevel.color} />
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Report Footer */}
      <Grid item xs={12}>
        <Box sx={{ textAlign: 'center', py: 2 }}>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="caption" color="text.secondary">
            Report generated on {new Date(data.generated_at).toLocaleString()} for class {data.class_id}
          </Typography>
        </Box>
      </Grid>
    </Grid>
  );
};

export default AnalyticsOverview;