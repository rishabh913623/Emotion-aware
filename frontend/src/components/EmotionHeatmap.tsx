import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Alert,
  CircularProgress,
} from '@mui/material';
import { ThermalCamera } from '@mui/icons-material';

interface HeatmapData {
  heatmap_image?: string;
  format?: string;
  days: number;
  message?: string;
}

interface Props {
  data: HeatmapData | null;
  days: number;
}

const EmotionHeatmap: React.FC<Props> = ({ data, days }) => {
  const [imageLoading, setImageLoading] = React.useState(true);
  const [imageError, setImageError] = React.useState(false);

  React.useEffect(() => {
    setImageLoading(true);
    setImageError(false);
  }, [data]);

  const handleImageLoad = () => {
    setImageLoading(false);
  };

  const handleImageError = () => {
    setImageLoading(false);
    setImageError(true);
  };

  if (!data) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (data.message || !data.heatmap_image) {
    return (
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <ThermalCamera sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary" gutterBottom>
          No Heatmap Data Available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {data.message || `No emotion data found for the last ${days} days`}
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Emotion intensity heatmap showing student learning states over time (last {days} days)
      </Typography>

      <Paper 
        elevation={2} 
        sx={{ 
          p: 2, 
          textAlign: 'center',
          bgcolor: 'grey.50',
          minHeight: 400,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        {imageLoading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CircularProgress sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary">
              Generating heatmap...
            </Typography>
          </Box>
        )}
        
        {imageError && (
          <Alert severity="error" sx={{ width: '100%' }}>
            Failed to load heatmap. Please try again.
          </Alert>
        )}

        <img
          src={`data:image/png;base64,${data.heatmap_image}`}
          alt="Emotion Heatmap"
          style={{
            maxWidth: '100%',
            maxHeight: '400px',
            objectFit: 'contain',
            display: imageLoading || imageError ? 'none' : 'block'
          }}
          onLoad={handleImageLoad}
          onError={handleImageError}
        />
      </Paper>

      <Box sx={{ mt: 2 }}>
        <Typography variant="caption" color="text.secondary">
          • Darker colors indicate higher activity levels<br/>
          • Each row represents a different learning state<br/>
          • Each column represents an hour of the day<br/>
          • Generated from {days === 1 ? 'today' : `last ${days} days`} of emotion data
        </Typography>
      </Box>
    </Box>
  );
};

export default EmotionHeatmap;