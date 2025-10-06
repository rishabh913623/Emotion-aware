import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface ClassroomState {
  isInSession: boolean;
  sessionId: string | null;
  participants: string[];
  mediaStream: MediaStream | null;
  isVideoEnabled: boolean;
  isAudioEnabled: boolean;
  isSharingScreen: boolean;
  chatMessages: ChatMessage[];
  loading: boolean;
  error: string | null;
}

interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  message: string;
  timestamp: string;
  sentiment?: string;
}

const initialState: ClassroomState = {
  isInSession: false,
  sessionId: null,
  participants: [],
  mediaStream: null,
  isVideoEnabled: false,
  isAudioEnabled: false,
  isSharingScreen: false,
  chatMessages: [],
  loading: false,
  error: null,
};

const classroomSlice = createSlice({
  name: 'classroom',
  initialState,
  reducers: {
    joinSession: (state, action: PayloadAction<{ sessionId: string }>) => {
      state.isInSession = true;
      state.sessionId = action.payload.sessionId;
    },
    leaveSession: (state) => {
      state.isInSession = false;
      state.sessionId = null;
      state.participants = [];
      state.mediaStream = null;
      state.isVideoEnabled = false;
      state.isAudioEnabled = false;
      state.isSharingScreen = false;
      state.chatMessages = [];
    },
    updateParticipants: (state, action: PayloadAction<string[]>) => {
      state.participants = action.payload;
    },
    setMediaStream: (state, action: PayloadAction<MediaStream | null>) => {
      // Note: MediaStream is not serializable, handle this in components
      // This is just for tracking state
    },
    toggleVideo: (state) => {
      state.isVideoEnabled = !state.isVideoEnabled;
    },
    toggleAudio: (state) => {
      state.isAudioEnabled = !state.isAudioEnabled;
    },
    toggleScreenShare: (state) => {
      state.isSharingScreen = !state.isSharingScreen;
    },
    addChatMessage: (state, action: PayloadAction<ChatMessage>) => {
      state.chatMessages.push(action.payload);
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const {
  joinSession,
  leaveSession,
  updateParticipants,
  setMediaStream,
  toggleVideo,
  toggleAudio,
  toggleScreenShare,
  addChatMessage,
  setLoading,
  setError,
} = classroomSlice.actions;

export default classroomSlice.reducer;