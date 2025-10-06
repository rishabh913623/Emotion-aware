import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Button,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Chip,
  Divider,
} from '@mui/material';
import {
  GetApp,
  PictureAsPdf,
  TableChart,
  Analytics,
  Timeline,
  TrendingUp,
  Assessment,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs, { Dayjs } from 'dayjs';
import { useAppSelector } from '../store/hooks';
import AdvancedTimelineChart from '../components/AdvancedTimelineChart';
import StudentEngagementTable from '../components/StudentEngagementTable';
import EmotionHeatmap from '../components/EmotionHeatmap';
import AnalyticsOverview from '../components/AnalyticsOverview';
import { reportsService } from '../services/reportsService';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`reports-tabpanel-${index}`}
      aria-labelledby={`reports-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const ReportsPage: React.FC = () => {
  const { currentClassId } = useAppSelector((state) => state.dashboard);
  const [activeTab, setActiveTab] = useState(0);
  const [timeRange, setTimeRange] = useState(24); // hours
  const [dateRange, setDateRange] = useState(7); // days for heatmap
  const [startDate, setStartDate] = useState<Dayjs | null>(dayjs().subtract(7, 'days'));
  const [endDate, setEndDate] = useState<Dayjs | null>(dayjs());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Report data states
  const [timelineData, setTimelineData] = useState(null);
  const [engagementData, setEngagementData] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [analyticsData, setAnalyticsData] = useState(null);

  useEffect(() => {
    if (currentClassId) {
      loadReportsData();
    }
  }, [currentClassId, timeRange, dateRange]);

  const loadReportsData = async () => {
    if (!currentClassId) return;

    setLoading(true);
    setError(null);

    try {
      const [timeline, engagement, analytics] = await Promise.all([
        reportsService.getEmotionTimeline(currentClassId, timeRange),
        reportsService.getStudentEngagement(currentClassId),
        reportsService.getAnalyticsOverview(currentClassId),
      ]);

      setTimelineData(timeline);
      setEngagementData(engagement);
      setAnalyticsData(analytics);

      // Load heatmap data separately as it might be heavier
      const heatmap = await reportsService.getClassHeatmap(currentClassId, dateRange);
      setHeatmapData(heatmap);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reports data');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleExportPDF = async () => {
    if (!currentClassId) return;

    try {
      setLoading(true);
      await reportsService.exportPDFReport(currentClassId);
    } catch (err) {
      setError('Failed to export PDF report');
    } finally {
      setLoading(false);
    }
  };

  const handleExportExcel = async () => {
    if (!currentClassId) return;

    try {
      setLoading(true);
      await reportsService.exportExcelReport(currentClassId);
    } catch (err) {
      setError('Failed to export Excel report');
    } finally {
      setLoading(false);
    }
  };

  if (!currentClassId) {
    return (
      <Box sx={{ textAlign: 'center', mt: 8 }}>
        <Analytics sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          No Active Class Session
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Start a class session to view reports and analytics.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Reports & Analytics
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<PictureAsPdf />}
            onClick={handleExportPDF}
            disabled={loading}
          >
            Export PDF
          </Button>
          <Button
            variant="outlined"
            startIcon={<TableChart />}
            onClick={handleExportExcel}
            disabled={loading}
          >
            Export Excel
          </Button>
        </Box>
      </Box>

      {/* Class Info */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6">Class: {currentClassId}</Typography>
            <Typography variant="body2" color="text.secondary">
              Report generated on {dayjs().format('MMMM D, YYYY at h:mm A')}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                onChange={(e) => setTimeRange(Number(e.target.value))}
                label="Time Range"
              >
                <MenuItem value={1}>Last Hour</MenuItem>
                <MenuItem value={6}>Last 6 Hours</MenuItem>
                <MenuItem value={24}>Last 24 Hours</MenuItem>
                <MenuItem value={72}>Last 3 Days</MenuItem>
                <MenuItem value={168}>Last Week</MenuItem>
              </Select>
            </FormControl>
            
            <Chip
              icon={<TrendingUp />}
              label={`${timeRange}h data`}
              color="primary"
              variant="outlined"
            />
          </Box>
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="reports tabs">
          <Tab label="Overview" icon={<Assessment />} />
          <Tab label="Timeline Analysis" icon={<Timeline />} />
          <Tab label="Student Engagement" icon={<TrendingUp />} />
          <Tab label="Emotion Heatmap" icon={<Analytics />} />
        </Tabs>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Tab Content */}
      <TabPanel value={activeTab} index={0}>
        <AnalyticsOverview data={analyticsData} />
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Emotion Timeline - Last {timeRange} Hours
              </Typography>
              <AdvancedTimelineChart data={timelineData} />
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Student Engagement Analysis
              </Typography>
              <StudentEngagementTable data={engagementData} />
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Class Emotion Heatmap
                </Typography>
                
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Days</InputLabel>
                  <Select
                    value={dateRange}
                    onChange={(e) => setDateRange(Number(e.target.value))}
                    label="Days"
                  >
                    <MenuItem value={1}>Today</MenuItem>
                    <MenuItem value={3}>Last 3 Days</MenuItem>
                    <MenuItem value={7}>Last Week</MenuItem>
                    <MenuItem value={14}>Last 2 Weeks</MenuItem>
                    <MenuItem value={30}>Last Month</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <EmotionHeatmap data={heatmapData} days={dateRange} />
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Quick Actions Footer */}
      <Paper sx={{ p: 2, mt: 4, bgcolor: 'grey.50' }}>
        <Typography variant="subtitle2" gutterBottom>
          Quick Actions
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="body2" gutterBottom>
                  Export comprehensive PDF report with all charts and analysis
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<PictureAsPdf />}
                  onClick={handleExportPDF}
                  disabled={loading}
                  fullWidth
                >
                  Generate PDF Report
                </Button>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="body2" gutterBottom>
                  Export raw data and statistics in Excel format
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<TableChart />}
                  onClick={handleExportExcel}
                  disabled={loading}
                  fullWidth
                >
                  Download Excel Data
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default ReportsPage;