import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface EmotionData {
  facial_emotion?: string;
  audio_emotion?: string;
  text_sentiment?: string;
  learning_state: string;
  confidence: number;
  timestamp: string;
}

interface StudentState {
  id: string;
  name?: string;
  current_emotion: EmotionData;
  last_updated: string;
}

interface ClassMood {
  engaged: number;
  confused: number;
  bored: number;
  frustrated: number;
  curious: number;
  neutral: number;
}

interface Alert {
  type: string;
  severity: 'info' | 'warning' | 'urgent';
  message: string;
  timestamp: string;
  suggestion: string;
  student_id?: string;
}

interface DashboardState {
  currentClassId: string | null;
  classMood: ClassMood;
  students: { [studentId: string]: StudentState };
  totalStudents: number;
  alerts: Alert[];
  lastUpdated: string | null;
  isConnected: boolean;
  loading: boolean;
  error: string | null;
}

const initialState: DashboardState = {
  currentClassId: null,
  classMood: {
    engaged: 0,
    confused: 0,
    bored: 0,
    frustrated: 0,
    curious: 0,
    neutral: 0,
  },
  students: {},
  totalStudents: 0,
  alerts: [],
  lastUpdated: null,
  isConnected: false,
  loading: false,
  error: null,
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    setCurrentClass: (state, action: PayloadAction<string>) => {
      state.currentClassId = action.payload;
    },
    updateClassState: (state, action: PayloadAction<{
      class_mood: ClassMood;
      students: { [studentId: string]: StudentState };
      total_students: number;
      last_updated: string;
    }>) => {
      state.classMood = action.payload.class_mood;
      state.students = action.payload.students;
      state.totalStudents = action.payload.total_students;
      state.lastUpdated = action.payload.last_updated;
    },
    updateStudentEmotion: (state, action: PayloadAction<{
      student_id: string;
      emotion_data: EmotionData;
      class_mood: ClassMood;
    }>) => {
      const { student_id, emotion_data, class_mood } = action.payload;
      
      if (!state.students[student_id]) {
        state.students[student_id] = {
          id: student_id,
          current_emotion: emotion_data,
          last_updated: emotion_data.timestamp,
        };
      } else {
        state.students[student_id].current_emotion = emotion_data;
        state.students[student_id].last_updated = emotion_data.timestamp;
      }
      
      state.classMood = class_mood;
      state.lastUpdated = emotion_data.timestamp;
    },
    updateAlerts: (state, action: PayloadAction<Alert[]>) => {
      state.alerts = action.payload;
    },
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearDashboard: (state) => {
      state.currentClassId = null;
      state.students = {};
      state.totalStudents = 0;
      state.alerts = [];
      state.lastUpdated = null;
      state.isConnected = false;
    },
  },
});

export const {
  setCurrentClass,
  updateClassState,
  updateStudentEmotion,
  updateAlerts,
  setConnectionStatus,
  setLoading,
  setError,
  clearDashboard,
} = dashboardSlice.actions;

export default dashboardSlice.reducer;