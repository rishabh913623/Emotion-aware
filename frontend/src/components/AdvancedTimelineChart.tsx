import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';
import {
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  FormControlLabel,
  Switch,
  Paper,
  Alert,
} from '@mui/material';

interface TimelineDataPoint {
  timestamp: string;
  learning_state: string;
  count: number;
  percentage: number;
}

interface TimelineData {
  timeline_data: TimelineDataPoint[];
  total_records: number;
  time_range_hours: number;
  unique_students: number;
}

interface Props {
  data: TimelineData | null;
}

const AdvancedTimelineChart: React.FC<Props> = ({ data }) => {
  const [chartType, setChartType] = React.useState('line');
  const [showPercentage, setShowPercentage] = React.useState(true);
  const [stackedView, setStackedView] = React.useState(false);

  if (!data || !data.timeline_data || data.timeline_data.length === 0) {
    return (
      <Alert severity="info">
        No timeline data available for the selected period.
      </Alert>
    );
  }

  // Process data for visualization
  const processedData = React.useMemo(() => {
    const timePoints = new Map<string, Record<string, number>>();
    
    // Group data by timestamp
    data.timeline_data.forEach(point => {
      const timestamp = point.timestamp;
      if (!timePoints.has(timestamp)) {
        timePoints.set(timestamp, {
          timestamp,
          engaged: 0,
          curious: 0,
          neutral: 0,
          confused: 0,
          bored: 0,
          frustrated: 0,
        });
      }
      
      const timeData = timePoints.get(timestamp)!;
      const valueKey = showPercentage ? 'percentage' : 'count';
      timeData[point.learning_state] = point[valueKey];
    });
    
    return Array.from(timePoints.values()).sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [data.timeline_data, showPercentage]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const total = payload.reduce((sum: number, entry: any) => sum + entry.value, 0);
      
      return (
        <Paper sx={{ p: 2, border: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2" gutterBottom>
            {formatTimestamp(label)}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography 
              key={index} 
              variant="body2" 
              sx={{ color: entry.color }}
            >
              {`${entry.name}: ${entry.value}${showPercentage ? '%' : ''}`}
            </Typography>
          ))}
          {!showPercentage && (
            <Typography variant="caption" color="text.secondary">
              Total: {total}
            </Typography>
          )}
        </Paper>
      );
    }
    return null;
  };

  const chartColors = {
    engaged: '#4caf50',
    curious: '#2196f3',
    neutral: '#9e9e9e',
    confused: '#ff9800',
    bored: '#f44336',
    frustrated: '#d32f2f',
  };

  return (
    <Box>
      {/* Chart Controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={(e, value) => value && setChartType(value)}
            size="small"
          >
            <ToggleButton value="line">Line</ToggleButton>
            <ToggleButton value="area">Area</ToggleButton>
            <ToggleButton value="bar">Bar</ToggleButton>
          </ToggleButtonGroup>
          
          <FormControlLabel
            control={
              <Switch
                checked={showPercentage}
                onChange={(e) => setShowPercentage(e.target.checked)}
                size="small"
              />
            }
            label="Show %"
          />
          
          {(chartType === 'area' || chartType === 'bar') && (
            <FormControlLabel
              control={
                <Switch
                  checked={stackedView}
                  onChange={(e) => setStackedView(e.target.checked)}
                  size="small"
                />
              }
              label="Stacked"
            />
          )}
        </Box>

        <Box sx={{ textAlign: 'right' }}>
          <Typography variant="caption" color="text.secondary">
            {data.unique_students} students • {data.total_records} data points
          </Typography>
        </Box>
      </Box>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        {chartType === 'line' && (
          <LineChart data={processedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={formatTimestamp}
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={showPercentage ? [0, 100] : [0, 'dataMax']}
              tick={{ fontSize: 12 }}
              label={{ 
                value: showPercentage ? 'Percentage (%)' : 'Count', 
                angle: -90, 
                position: 'insideLeft' 
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {Object.entries(chartColors).map(([emotion, color]) => (
              <Line
                key={emotion}
                type="monotone"
                dataKey={emotion}
                stroke={color}
                strokeWidth={2}
                dot={false}
                name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
              />
            ))}
          </LineChart>
        )}

        {chartType === 'area' && (
          <AreaChart data={processedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={formatTimestamp}
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={showPercentage ? [0, 100] : [0, 'dataMax']}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {Object.entries(chartColors).map(([emotion, color]) => (
              <Area
                key={emotion}
                type="monotone"
                dataKey={emotion}
                stackId={stackedView ? "1" : emotion}
                stroke={color}
                fill={color}
                fillOpacity={0.6}
                name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
              />
            ))}
          </AreaChart>
        )}

        {chartType === 'bar' && (
          <BarChart data={processedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={formatTimestamp}
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={showPercentage ? [0, 100] : [0, 'dataMax']}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {Object.entries(chartColors).map(([emotion, color]) => (
              <Bar
                key={emotion}
                dataKey={emotion}
                stackId={stackedView ? "1" : undefined}
                fill={color}
                name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
              />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>

      {/* Chart Summary */}
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="caption" color="text.secondary">
          Data collected over {data.time_range_hours} hours
        </Typography>
        
        <Typography variant="caption" color="text.secondary">
          Last updated: {new Date().toLocaleTimeString()}
        </Typography>
      </Box>
    </Box>
  );
};

export default AdvancedTimelineChart;