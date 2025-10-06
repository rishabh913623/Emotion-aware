"""
Reports and Visualization API for Emotion-Aware Virtual Classroom
Week 7: Advanced charts, time-series analysis, PDF/Excel export
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np

from backend.core.database import get_db, EmotionData, ClassSession, User, Classroom
from backend.core.config import get_settings, LEARNING_STATES

router = APIRouter()
settings = get_settings()

class ReportsService:
    """Service for generating reports and visualizations"""
    
    def __init__(self):
        self.plt_style = 'seaborn-v0_8'
        plt.style.use('default')  # Use default style as fallback
        
    def generate_emotion_timeline(self, class_id: str, db: Session, hours: int = 24) -> Dict:
        """Generate emotion timeline chart data"""
        
        # Query emotion data for the specified period
        since_time = datetime.now() - timedelta(hours=hours)
        
        emotion_query = db.query(EmotionData).filter(
            EmotionData.classroom_id == class_id,
            EmotionData.timestamp >= since_time
        ).order_by(EmotionData.timestamp).all()
        
        if not emotion_query:
            return {"message": "No data available for the specified period"}
        
        # Convert to DataFrame for easier processing
        data = []
        for record in emotion_query:
            data.append({
                'timestamp': record.timestamp,
                'learning_state': record.predicted_state or 'neutral',
                'confidence': record.fusion_confidence or 0.0,
                'user_id': record.user_id
            })
        
        df = pd.DataFrame(data)
        
        # Group by 5-minute intervals
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample and count emotions per interval
        timeline_data = []
        for state in LEARNING_STATES.keys():
            state_data = df[df['learning_state'] == state]
            if len(state_data) > 0:
                resampled = state_data.resample('5T').size()
                for timestamp, count in resampled.items():
                    if count > 0:
                        timeline_data.append({
                            'timestamp': timestamp.isoformat(),
                            'learning_state': state,
                            'count': int(count),
                            'percentage': 0  # Will calculate later
                        })
        
        # Calculate percentages for each time interval
        timestamp_totals = {}
        for item in timeline_data:
            ts = item['timestamp']
            if ts not in timestamp_totals:
                timestamp_totals[ts] = 0
            timestamp_totals[ts] += item['count']
        
        for item in timeline_data:
            total = timestamp_totals[item['timestamp']]
            if total > 0:
                item['percentage'] = round((item['count'] / total) * 100, 2)
        
        return {
            'timeline_data': timeline_data,
            'total_records': len(emotion_query),
            'time_range_hours': hours,
            'unique_students': len(df['user_id'].unique())
        }
    
    def generate_student_engagement_report(self, class_id: str, db: Session) -> Dict:
        """Generate detailed student engagement analysis"""
        
        # Get all students who participated
        students_query = db.query(
            EmotionData.user_id,
            func.count(EmotionData.id).label('total_records'),
            func.avg(EmotionData.fusion_confidence).label('avg_confidence'),
            func.min(EmotionData.timestamp).label('first_activity'),
            func.max(EmotionData.timestamp).label('last_activity')
        ).filter(
            EmotionData.classroom_id == class_id
        ).group_by(EmotionData.user_id).all()
        
        student_reports = []
        
        for student in students_query:
            user_id = student.user_id
            
            # Get emotion distribution for this student
            emotion_dist = db.query(
                EmotionData.predicted_state,
                func.count(EmotionData.id).label('count')
            ).filter(
                EmotionData.classroom_id == class_id,
                EmotionData.user_id == user_id
            ).group_by(EmotionData.predicted_state).all()
            
            # Calculate engagement metrics
            total_records = student.total_records
            emotion_distribution = {}
            engagement_score = 0
            
            for emotion, count in emotion_dist:
                if emotion:
                    percentage = (count / total_records) * 100
                    emotion_distribution[emotion] = {
                        'count': count,
                        'percentage': round(percentage, 2)
                    }
                    
                    # Weight emotions for engagement score
                    if emotion in ['engaged', 'curious']:
                        engagement_score += percentage * 1.0
                    elif emotion == 'neutral':
                        engagement_score += percentage * 0.5
                    elif emotion in ['confused']:
                        engagement_score += percentage * 0.3
                    # bored and frustrated get 0 weight
            
            # Calculate session duration
            if student.first_activity and student.last_activity:
                duration = student.last_activity - student.first_activity
                duration_minutes = int(duration.total_seconds() / 60)
            else:
                duration_minutes = 0
            
            student_reports.append({
                'user_id': user_id,
                'total_records': total_records,
                'avg_confidence': round(float(student.avg_confidence or 0), 3),
                'session_duration_minutes': duration_minutes,
                'engagement_score': round(engagement_score, 2),
                'emotion_distribution': emotion_distribution,
                'first_activity': student.first_activity.isoformat() if student.first_activity else None,
                'last_activity': student.last_activity.isoformat() if student.last_activity else None
            })
        
        # Sort by engagement score
        student_reports.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        return {
            'student_reports': student_reports,
            'class_summary': {
                'total_students': len(student_reports),
                'avg_engagement': round(np.mean([s['engagement_score'] for s in student_reports]), 2) if student_reports else 0,
                'total_emotion_records': sum(s['total_records'] for s in student_reports)
            }
        }
    
    def generate_class_mood_heatmap(self, class_id: str, db: Session, days: int = 7) -> str:
        """Generate class mood heatmap as base64 image"""
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Query hourly emotion data
        emotion_data = db.query(
            func.date_trunc('hour', EmotionData.timestamp).label('hour'),
            EmotionData.predicted_state,
            func.count(EmotionData.id).label('count')
        ).filter(
            EmotionData.classroom_id == class_id,
            EmotionData.timestamp >= since_date
        ).group_by(
            func.date_trunc('hour', EmotionData.timestamp),
            EmotionData.predicted_state
        ).all()
        
        if not emotion_data:
            return None
        
        # Prepare data for heatmap
        df_data = []
        for record in emotion_data:
            if record.predicted_state:
                df_data.append({
                    'hour': record.hour,
                    'emotion': record.predicted_state,
                    'count': record.count
                })
        
        if not df_data:
            return None
            
        df = pd.DataFrame(df_data)
        
        # Pivot for heatmap
        heatmap_data = df.pivot_table(
            index='emotion',
            columns='hour',
            values='count',
            fill_value=0
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='d',
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Number of Students'}
        )
        plt.title(f'Class Emotion Heatmap - Last {days} Days')
        plt.xlabel('Time (Hour)')
        plt.ylabel('Learning State')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_pdf_report(self, class_id: str, data: Dict, db: Session) -> BytesIO:
        """Generate comprehensive PDF report"""
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkgreen
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Emotion-Aware Virtual Classroom Report", title_style))
        story.append(Spacer(1, 20))
        
        # Class Information
        class_info = f"""
        <b>Class ID:</b> {class_id}<br/>
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Report Period:</b> Last 24 hours
        """
        story.append(Paragraph(class_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Class Summary
        story.append(Paragraph("Class Summary", subtitle_style))
        
        if 'class_summary' in data:
            summary = data['class_summary']
            summary_data = [
                ['Metric', 'Value'],
                ['Total Students', str(summary.get('total_students', 0))],
                ['Average Engagement Score', f"{summary.get('avg_engagement', 0):.2f}%"],
                ['Total Emotion Records', str(summary.get('total_emotion_records', 0))]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
        
        # Student Engagement Details
        if 'student_reports' in data:
            story.append(Paragraph("Top Students by Engagement", subtitle_style))
            
            top_students = data['student_reports'][:5]  # Top 5 students
            
            student_data = [['Student ID', 'Engagement Score', 'Session Duration (min)', 'Avg Confidence']]
            
            for student in top_students:
                student_data.append([
                    student['user_id'][-8:],  # Last 8 chars
                    f"{student['engagement_score']:.1f}%",
                    str(student['session_duration_minutes']),
                    f"{student['avg_confidence']:.3f}"
                ])
            
            student_table = Table(student_data)
            student_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(student_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

# Initialize service
reports_service = ReportsService()

@router.get("/api/class/{class_id}/timeline")
async def get_emotion_timeline(
    class_id: str,
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    db: Session = Depends(get_db)
):
    """Get emotion timeline data for charts"""
    
    try:
        timeline_data = reports_service.generate_emotion_timeline(class_id, db, hours)
        return timeline_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate timeline: {str(e)}")

@router.get("/api/class/{class_id}/student-engagement")
async def get_student_engagement_report(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed student engagement analysis"""
    
    try:
        engagement_data = reports_service.generate_student_engagement_report(class_id, db)
        return engagement_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate engagement report: {str(e)}")

@router.get("/api/class/{class_id}/heatmap")
async def get_class_mood_heatmap(
    class_id: str,
    days: int = Query(7, ge=1, le=30),  # 1 day to 1 month
    db: Session = Depends(get_db)
):
    """Get class mood heatmap as base64 image"""
    
    try:
        heatmap_image = reports_service.generate_class_mood_heatmap(class_id, db, days)
        
        if not heatmap_image:
            return {"message": "No data available for heatmap generation"}
        
        return {
            "heatmap_image": heatmap_image,
            "format": "base64_png",
            "days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")

@router.get("/api/class/{class_id}/export/pdf")
async def export_pdf_report(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Export comprehensive PDF report"""
    
    try:
        # Gather all report data
        engagement_data = reports_service.generate_student_engagement_report(class_id, db)
        timeline_data = reports_service.generate_emotion_timeline(class_id, db, 24)
        
        combined_data = {**engagement_data, **timeline_data}
        
        # Generate PDF
        pdf_buffer = reports_service.generate_pdf_report(class_id, combined_data, db)
        
        # Return PDF as response
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=class_report_{class_id}_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")

@router.get("/api/class/{class_id}/export/excel")
async def export_excel_report(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Export data as Excel file"""
    
    try:
        # Get raw emotion data
        emotion_data = db.query(EmotionData).filter(
            EmotionData.classroom_id == class_id
        ).all()
        
        if not emotion_data:
            raise HTTPException(status_code=404, detail="No data found for this class")
        
        # Convert to DataFrame
        data_rows = []
        for record in emotion_data:
            data_rows.append({
                'timestamp': record.timestamp,
                'user_id': record.user_id,
                'facial_emotion': record.facial_emotion,
                'audio_emotion': record.audio_emotion,
                'text_sentiment': record.text_sentiment,
                'learning_state': record.predicted_state,
                'confidence': record.fusion_confidence,
                'facial_confidence': record.facial_confidence,
                'audio_confidence': record.audio_confidence,
                'text_confidence': record.text_confidence
            })
        
        df = pd.DataFrame(data_rows)
        
        # Create Excel buffer
        excel_buffer = BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Raw data sheet
            df.to_excel(writer, sheet_name='Raw_Emotion_Data', index=False)
            
            # Summary statistics sheet
            summary_stats = df.groupby('learning_state').agg({
                'confidence': ['count', 'mean', 'std'],
                'user_id': 'nunique'
            }).round(3)
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Per-student summary
            student_summary = df.groupby(['user_id', 'learning_state']).size().unstack(fill_value=0)
            student_summary.to_excel(writer, sheet_name='Student_Summary')
        
        excel_buffer.seek(0)
        
        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=emotion_data_{class_id}_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel report: {str(e)}")

@router.get("/api/class/{class_id}/analytics/overview")
async def get_analytics_overview(
    class_id: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive analytics overview"""
    
    try:
        # Get basic statistics
        total_records = db.query(func.count(EmotionData.id)).filter(
            EmotionData.classroom_id == class_id
        ).scalar()
        
        unique_students = db.query(func.count(func.distinct(EmotionData.user_id))).filter(
            EmotionData.classroom_id == class_id
        ).scalar()
        
        # Get emotion distribution
        emotion_dist = db.query(
            EmotionData.predicted_state,
            func.count(EmotionData.id).label('count')
        ).filter(
            EmotionData.classroom_id == class_id
        ).group_by(EmotionData.predicted_state).all()
        
        emotion_distribution = {}
        for emotion, count in emotion_dist:
            if emotion:
                percentage = (count / total_records) * 100 if total_records > 0 else 0
                emotion_distribution[emotion] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
        
        # Get time-based patterns
        hourly_activity = db.query(
            func.extract('hour', EmotionData.timestamp).label('hour'),
            func.count(EmotionData.id).label('count')
        ).filter(
            EmotionData.classroom_id == class_id
        ).group_by(func.extract('hour', EmotionData.timestamp)).all()
        
        activity_by_hour = {int(hour): count for hour, count in hourly_activity}
        
        # Calculate engagement metrics
        engaged_emotions = ['engaged', 'curious']
        disengaged_emotions = ['bored', 'frustrated']
        
        engaged_count = sum(
            emotion_distribution.get(emotion, {}).get('count', 0)
            for emotion in engaged_emotions
        )
        
        disengaged_count = sum(
            emotion_distribution.get(emotion, {}).get('count', 0)
            for emotion in disengaged_emotions
        )
        
        engagement_ratio = engaged_count / max(disengaged_count, 1)
        
        return {
            'class_id': class_id,
            'overview': {
                'total_emotion_records': total_records,
                'unique_students': unique_students,
                'data_collection_active': total_records > 0
            },
            'emotion_distribution': emotion_distribution,
            'engagement_metrics': {
                'engaged_count': engaged_count,
                'disengaged_count': disengaged_count,
                'engagement_ratio': round(engagement_ratio, 2),
                'overall_engagement_score': round((engaged_count / max(total_records, 1)) * 100, 2)
            },
            'activity_patterns': {
                'hourly_distribution': activity_by_hour,
                'peak_hour': max(activity_by_hour.items(), key=lambda x: x[1])[0] if activity_by_hour else None
            },
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics overview: {str(e)}")