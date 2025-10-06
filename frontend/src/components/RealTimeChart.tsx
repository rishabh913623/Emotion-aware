import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Box, Typography, FormControlLabel, Switch } from '@mui/material';
import { dashboardService } from '../services/dashboardService';

interface Props {
  classId: string | null;
}

interface DataPoint {
  timestamp: string;
  engaged: number;
  confused: number;
  bored: number;
  curious: number;
  frustrated: number;
  neutral: number;
}

const RealTimeChart: React.FC<Props> = ({ classId }) => {
  const [data, setData] = useState<DataPoint[]>([]);
  const [showAllLines, setShowAllLines] = useState(false);

  useEffect(() => {
    if (!classId) return;

    const fetchData = async () => {
      try {
        // In a real implementation, this would fetch historical data
        // For now, we'll simulate real-time data
        const now = new Date();
        const mockData: DataPoint[] = [];

        for (let i = 9; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 60000); // Every minute
          mockData.push({
            timestamp: time.toLocaleTimeString(),
            engaged: Math.random() * 40 + 20,
            confused: Math.random() * 20 + 5,
            bored: Math.random() * 15 + 5,
            curious: Math.random() * 30 + 10,
            frustrated: Math.random() * 10 + 2,
            neutral: Math.random() * 25 + 15,
          });
        }

        setData(mockData);
      } catch (error) {
        console.error('Failed to fetch timeline data:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [classId]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 2, 
          border: 1, 
          borderColor: 'divider',
          borderRadius: 1,
          boxShadow: 2
        }}>
          <Typography variant="subtitle2" gutterBottom>
            {label}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography 
              key={index} 
              variant="body2" 
              sx={{ color: entry.color }}
            >
              {`${entry.name}: ${entry.value.toFixed(1)}%`}
            </Typography>
          ))}
        </Box>
      );
    }
    return null;
  };

  if (!classId) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100%',
        color: 'text.secondary' 
      }}>
        <Typography variant="body2">Start a class session to see real-time data</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="subtitle1" color="text.secondary">
          Last 10 Minutes
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showAllLines}
              onChange={(e) => setShowAllLines(e.target.checked)}
              size="small"
            />
          }
          label="Show All"
        />
      </Box>

      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            domain={[0, 100]}
            tick={{ fontSize: 12 }}
            label={{ value: 'Percentage', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Always show engagement line */}
          <Line
            type="monotone"
            dataKey="engaged"
            stroke="#4caf50"
            strokeWidth={3}
            name="Engaged"
            dot={false}
          />
          
          {/* Show curious line if engagement mode or show all */}
          {showAllLines && (
            <Line
              type="monotone"
              dataKey="curious"
              stroke="#2196f3"
              strokeWidth={2}
              name="Curious"
              dot={false}
            />
          )}
          
          {/* Always show confusion (important metric) */}
          <Line
            type="monotone"
            dataKey="confused"
            stroke="#ff9800"
            strokeWidth={2}
            name="Confused"
            dot={false}
          />
          
          {showAllLines && (
            <>
              <Line
                type="monotone"
                dataKey="bored"
                stroke="#f44336"
                strokeWidth={2}
                name="Bored"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="frustrated"
                stroke="#d32f2f"
                strokeWidth={2}
                name="Frustrated"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="neutral"
                stroke="#9e9e9e"
                strokeWidth={1}
                name="Neutral"
                dot={false}
              />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default RealTimeChart;