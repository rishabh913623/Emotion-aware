class ReportsService {
  private baseURL = '/api/reports';

  async getEmotionTimeline(classId: string, hours: number = 24) {
    const response = await fetch(
      `${this.baseURL}/api/class/${classId}/timeline?hours=${hours}`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch emotion timeline');
    }
    
    return response.json();
  }

  async getStudentEngagement(classId: string) {
    const response = await fetch(
      `${this.baseURL}/api/class/${classId}/student-engagement`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch student engagement data');
    }
    
    return response.json();
  }

  async getClassHeatmap(classId: string, days: number = 7) {
    const response = await fetch(
      `${this.baseURL}/api/class/${classId}/heatmap?days=${days}`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch class heatmap');
    }
    
    return response.json();
  }

  async getAnalyticsOverview(classId: string) {
    const response = await fetch(
      `${this.baseURL}/api/class/${classId}/analytics/overview`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch analytics overview');
    }
    
    return response.json();
  }

  async exportPDFReport(classId: string) {
    const response = await fetch(
      `${this.baseURL}/api/class/${classId}/export/pdf`
    );
    
    if (!response.ok) {
      throw new Error('Failed to export PDF report');
    }
    
    // Create blob and trigger download
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `class_report_${classId}_${new Date().toISOString().split('T')[0]}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }

  async exportExcelReport(classId: string) {
    const response = await fetch(
      `${this.baseURL}/api/class/${classId}/export/excel`
    );
    
    if (!response.ok) {
      throw new Error('Failed to export Excel report');
    }
    
    // Create blob and trigger download
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `emotion_data_${classId}_${new Date().toISOString().split('T')[0]}.xlsx`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

export const reportsService = new ReportsService();