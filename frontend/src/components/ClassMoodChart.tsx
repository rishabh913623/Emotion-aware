import React from 'react';
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
import { Box, Typography, Switch, FormControlLabel } from '@mui/material';

interface ClassMood {
  engaged: number;
  confused: number;
  bored: number;
  frustrated: number;
  curious: number;
  neutral: number;
}

interface Props {
  classMood: ClassMood;
}

const COLORS = {
  engaged: '#4caf50',
  curious: '#2196f3',
  neutral: '#9e9e9e',
  confused: '#ff9800',
  bored: '#f44336',
  frustrated: '#d32f2f',
};

const EMOTION_LABELS = {
  engaged: 'Engaged',
  curious: 'Curious',
  neutral: 'Neutral',
  confused: 'Confused',
  bored: 'Bored',
  frustrated: 'Frustrated',
};

const ClassMoodChart: React.FC<Props> = ({ classMood }) => {
  const [showPieChart, setShowPieChart] = React.useState(true);

  const chartData = Object.entries(classMood).map(([emotion, percentage]) => ({
    name: EMOTION_LABELS[emotion as keyof typeof EMOTION_LABELS],
    value: Math.round(percentage * 10) / 10,
    color: COLORS[emotion as keyof typeof COLORS],
  })).filter(item => item.value > 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 1, 
          border: 1, 
          borderColor: 'divider',
          borderRadius: 1 
        }}>
          <Typography variant="body2">
            {`${payload[0].name}: ${payload[0].value}%`}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Box sx={{ height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="subtitle1" color="text.secondary">
          Current Class Mood Distribution
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showPieChart}
              onChange={(e) => setShowPieChart(e.target.checked)}
              size="small"
            />
          }
          label={showPieChart ? 'Pie Chart' : 'Bar Chart'}
        />
      </Box>

      <ResponsiveContainer width="100%" height="85%">
        {showPieChart ? (
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        ) : (
          <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" fill="#8884d8">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>

      {chartData.length === 0 && (
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '80%',
          color: 'text.secondary' 
        }}>
          <Typography variant="body2">No emotion data available</Typography>
        </Box>
      )}
    </Box>
  );
};

export default ClassMoodChart;