/**
 * API Client for SafeWatch AI Backend
 * Connects frontend to Python FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ApiResponse<T> {
  data?: T;
  error?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // ==================== HELPER METHODS ====================

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
      };
    }
  }

  // ==================== CAMERA APIs ====================

  async getCameras() {
    return this.request<any[]>('/api/cameras/list');
  }

  async getCamera(cameraId: string) {
    return this.request<any>(`/api/cameras/${cameraId}`);
  }

  async addCamera(cameraData: {
    camera_id: string;
    name: string;
    location: string;
    stream_url: string;
    camera_type: string;
    fps?: number;
    resolution?: string;
  }) {
    return this.request<any>('/api/cameras/add', {
      method: 'POST',
      body: JSON.stringify(cameraData),
    });
  }

  async deleteCamera(cameraId: string) {
    return this.request<any>(`/api/cameras/${cameraId}`, {
      method: 'DELETE',
    });
  }

  async startCamera(cameraId: string) {
    return this.request<any>(`/api/cameras/${cameraId}/start`, {
      method: 'POST',
    });
  }

  async stopCamera(cameraId: string) {
    return this.request<any>(`/api/cameras/${cameraId}/stop`, {
      method: 'POST',
    });
  }

  // ==================== DETECTION APIs ====================

  async getRecentDetections(limit: number = 100, cameraId?: string) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (cameraId) params.append('camera_id', cameraId);
    return this.request<any[]>(`/api/detections/recent?${params}`);
  }

  async getDetectionStats(startDate?: string, endDate?: string) {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    return this.request<any>(`/api/detections/stats?${params}`);
  }

  async getDetectionsByCamera(cameraId: string, hours: number = 24) {
    return this.request<any>(`/api/detections/by-camera/${cameraId}?hours=${hours}`);
  }

  // ==================== ALERT APIs ====================

  async getAlerts(limit: number = 50, status?: string, alertLevel?: number) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (status) params.append('status', status);
    if (alertLevel) params.append('alert_level', alertLevel.toString());
    return this.request<any[]>(`/api/alerts/list?${params}`);
  }

  async getAlert(alertId: string) {
    return this.request<any>(`/api/alerts/${alertId}`);
  }

  async acknowledgeAlert(alertId: string) {
    return this.request<any>(`/api/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    });
  }

  async resolveAlert(alertId: string) {
    return this.request<any>(`/api/alerts/${alertId}/resolve`, {
      method: 'POST',
    });
  }

  async getAlertStats(days: number = 7) {
    return this.request<any>(`/api/alerts/stats/summary?days=${days}`);
  }

  // ==================== FACE RECOGNITION APIs ====================

  async registerPerson(formData: FormData) {
    return fetch(`${this.baseUrl}/api/faces/register`, {
      method: 'POST',
      body: formData,
    }).then(res => res.json());
  }

  async getPersons(classification?: string) {
    const params = classification ? `?classification=${classification}` : '';
    return this.request<any[]>(`/api/faces/list${params}`);
  }

  async getPerson(personId: string) {
    return this.request<any>(`/api/faces/${personId}`);
  }

  async updatePerson(personId: string, data: { name?: string; classification?: string }) {
    return this.request<any>(`/api/faces/${personId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deletePerson(personId: string) {
    return this.request<any>(`/api/faces/${personId}`, {
      method: 'DELETE',
    });
  }

  async getUnknownPersons(limit: number = 100) {
    return this.request<any[]>(`/api/faces/unknown/list?limit=${limit}`);
  }

  async classifyUnknownPerson(unknownId: number, data: { name: string; classification: string }) {
    return this.request<any>(`/api/faces/unknown/${unknownId}/classify`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async clusterUnknownFaces() {
    return this.request<any>('/api/faces/cluster-unknown', {
      method: 'POST',
    });
  }

  // ==================== ANALYTICS APIs ====================

  async getDashboardAnalytics(period: 'today' | 'week' | 'month' | 'year' = 'today') {
    return this.request<any>(`/api/analytics/dashboard?period=${period}`);
  }

  async getTimelineData(period: string) {
    return this.request<any>(`/api/analytics/timeline?period=${period}`);
  }

  async getActivityHeatmap(days: number = 7) {
    return this.request<any>(`/api/analytics/activity-heatmap?days=${days}`);
  }

  // ==================== SETTINGS APIs ====================

  async getSystemSettings() {
    return this.request<any>('/api/settings/system');
  }

  async getNotificationConfig() {
    return this.request<any>('/api/settings/notification-config');
  }

  async getStorageInfo() {
    return this.request<any>('/api/settings/storage-info');
  }

  // ==================== HEALTH CHECK ====================

  async healthCheck() {
    return this.request<any>('/health');
  }

  // ==================== WEBSOCKET CONNECTIONS ====================

  createDetectionWebSocket(onMessage: (data: any) => void, onError?: (error: any) => void) {
    const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/detections`);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    return ws;
  }

  getCameraStreamUrl(cameraId: string): string {
    return `${this.baseUrl.replace('http', 'ws')}/ws/camera/${cameraId}`;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export class for testing
export { ApiClient };
