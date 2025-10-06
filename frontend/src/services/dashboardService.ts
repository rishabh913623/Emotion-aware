import {
  setConnectionStatus,
  updateClassState,
  updateStudentEmotion,
  updateAlerts,
  setError,
} from '../store/slices/dashboardSlice';
import { AppDispatch } from '../store/store';

class DashboardService {
  private websocket: WebSocket | null = null;
  private dispatch: AppDispatch | null = null;
  private classId: string | null = null;
  private instructorId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  setDispatch(dispatch: AppDispatch) {
    this.dispatch = dispatch;
  }

  async connect(classId: string, instructorId: string): Promise<void> {
    this.classId = classId;
    this.instructorId = instructorId;

    const wsUrl = `ws://localhost:8000/api/dashboard/ws/dashboard/${classId}/${instructorId}`;
    
    try {
      this.websocket = new WebSocket(wsUrl);
      
      this.websocket.onopen = () => {
        console.log('Dashboard WebSocket connected');
        this.dispatch?.({
          type: setConnectionStatus.type,
          payload: true,
        });
        this.reconnectAttempts = 0;
      };

      this.websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onclose = (event) => {
        console.log('Dashboard WebSocket disconnected:', event.code, event.reason);
        this.dispatch?.({
          type: setConnectionStatus.type,
          payload: false,
        });
        
        // Attempt to reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => this.reconnect(), 3000 * (this.reconnectAttempts + 1));
        }
      };

      this.websocket.onerror = (error) => {
        console.error('Dashboard WebSocket error:', error);
        this.dispatch?.({
          type: setError.type,
          payload: 'WebSocket connection error',
        });
      };

    } catch (error) {
      console.error('Failed to connect to dashboard WebSocket:', error);
      throw error;
    }
  }

  private async reconnect(): Promise<void> {
    if (!this.classId || !this.instructorId) return;
    
    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    try {
      await this.connect(this.classId, this.instructorId);
    } catch (error) {
      console.error('Reconnection failed:', error);
    }
  }

  private handleMessage(message: any): void {
    if (!this.dispatch) return;

    switch (message.type) {
      case 'class_state':
        this.dispatch({
          type: updateClassState.type,
          payload: message.data,
        });
        break;

      case 'emotion_update':
        this.dispatch({
          type: updateStudentEmotion.type,
          payload: {
            student_id: message.student_id,
            emotion_data: message.emotion_data,
            class_mood: message.class_mood,
          },
        });
        
        // Refresh alerts after emotion update
        this.refreshAlerts();
        break;

      case 'pong':
        // Keep-alive response
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }

  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.classId = null;
    this.instructorId = null;
    this.reconnectAttempts = 0;
  }

  // API Methods
  async startClassSession(classId: string, instructorId: string): Promise<any> {
    const response = await fetch(`/api/dashboard/api/class/${classId}/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ instructor_id: instructorId }),
    });

    if (!response.ok) {
      throw new Error('Failed to start class session');
    }

    return response.json();
  }

  async endClassSession(classId: string, instructorId: string): Promise<any> {
    const response = await fetch(`/api/dashboard/api/class/${classId}/end`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ instructor_id: instructorId }),
    });

    if (!response.ok) {
      throw new Error('Failed to end class session');
    }

    return response.json();
  }

  async getCurrentClassState(classId: string): Promise<any> {
    const response = await fetch(`/api/dashboard/api/class/${classId}/current-state`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch class state');
    }

    return response.json();
  }

  async getStudentHistory(classId: string, studentId: string, minutes = 30): Promise<any> {
    const response = await fetch(
      `/api/dashboard/api/class/${classId}/student/${studentId}/history?minutes=${minutes}`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch student history');
    }

    return response.json();
  }

  async getClassAlerts(classId: string): Promise<any> {
    const response = await fetch(`/api/dashboard/api/class/${classId}/alerts`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch class alerts');
    }

    return response.json();
  }

  async getClassSummary(classId: string): Promise<any> {
    const response = await fetch(`/api/dashboard/api/class/${classId}/summary`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch class summary');
    }

    return response.json();
  }

  private async refreshAlerts(): Promise<void> {
    if (!this.classId || !this.dispatch) return;

    try {
      const alertsData = await this.getClassAlerts(this.classId);
      this.dispatch({
        type: updateAlerts.type,
        payload: alertsData.alerts,
      });
    } catch (error) {
      console.error('Failed to refresh alerts:', error);
    }
  }

  // Keep-alive ping
  sendPing(): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({ type: 'ping' }));
    }
  }
}

export const dashboardService = new DashboardService();