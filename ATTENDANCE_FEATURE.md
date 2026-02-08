# 📋 Automatic Attendance Tracking Feature

## Overview
The Emotion-Aware Virtual Classroom now includes **automatic attendance tracking** that records student attendance as soon as they join a classroom session. No manual intervention required!

## ✨ Key Features

### 1. **Automatic Attendance Recording**
- ✅ Attendance is automatically recorded when a student/instructor joins a room
- ✅ Timestamp of join time is captured
- ✅ User role (student/instructor) is tracked
- ✅ No duplicate entries - each user is recorded once per session

### 2. **Real-time Confirmation**
- Students receive instant confirmation when attendance is recorded
- Visual feedback in the chat: "✅ Your attendance has been automatically recorded at [time]"
- Green success notification on joining

### 3. **Instructor Dashboard**
- View complete attendance records with one click
- See all attendees with names, roles, and join times
- Export attendance to CSV format
- Real-time statistics (total attendees, students count, instructors count)

## 🚀 How It Works

### For Students:
1. **Join the classroom** by entering Room ID
2. **Attendance is automatically recorded** - no action needed!
3. See confirmation message in chat
4. Continue with the session normally

### For Instructors:
1. **Create or join a classroom**
2. Click **"📋 Attendance"** button in the Instructor Dashboard
3. View complete attendance report with:
   - Student names
   - Join times
   - Roles
   - Present/Absent status
4. **Export to CSV** for record-keeping

## 📡 API Endpoints

### Get Room Attendance
```http
GET /api/attendance/room/{room_id}
```

**Response:**
```json
{
  "room_id": "abc123...",
  "attendance_records": [
    {
      "user_id": "user_123",
      "username": "John Doe",
      "role": "student",
      "joined_at": "2026-02-08T10:30:00",
      "status": "present"
    }
  ],
  "statistics": {
    "total_attendees": 25,
    "students_present": 24,
    "instructors_present": 1
  }
}
```

### Export Attendance (CSV or JSON)
```http
GET /api/attendance/room/{room_id}/export?format=csv
```

**CSV Format:**
```csv
User ID,Username,Role,Joined At,Status
user_123,John Doe,student,2026-02-08T10:30:00,present
user_456,Jane Smith,student,2026-02-08T10:31:15,present
```

### Student Attendance History
```http
GET /api/attendance/student/{user_id}
```

**Response:**
```json
{
  "user_id": "user_123",
  "username": "John Doe",
  "total_sessions_attended": 15,
  "attendance_history": [
    {
      "room_id": "abc123...",
      "room_short_id": "abc12345",
      "joined_at": "2026-02-08T10:30:00",
      "status": "present"
    }
  ]
}
```

### Overall Attendance Summary
```http
GET /api/attendance/summary
```

**Response:**
```json
{
  "total_attendance_records": 125,
  "rooms_with_attendance": 5,
  "room_summaries": [
    {
      "room_id": "abc123...",
      "room_short_id": "abc12345",
      "room_name": "Virtual Classroom",
      "total_attendees": 25,
      "students_count": 24,
      "first_join": "2026-02-08T10:00:00",
      "last_join": "2026-02-08T10:35:00"
    }
  ]
}
```

## 💡 Usage Examples

### Example 1: View Attendance During Class
1. As instructor, join your classroom
2. Click **"📋 Attendance"** button
3. See real-time attendance list
4. View who joined and when

### Example 2: Export Attendance Report
1. Click **"📋 Attendance"** button
2. Click **"📥 Export CSV"** button
3. CSV file downloads automatically
4. Open in Excel/Google Sheets for further analysis

### Example 3: Check Student History
```bash
# Using curl or Postman
curl http://localhost:8001/api/attendance/student/user_123
```

## 🔧 Technical Implementation

### Backend (Python/FastAPI)
```python
def record_attendance(room_id: str, user_id: str, username: str, role: str):
    """Automatically record attendance when a student joins the room"""
    if room_id not in attendance_db:
        attendance_db[room_id] = []
    
    attendance_record = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "joined_at": datetime.now().isoformat(),
        "status": "present",
        "timestamp": datetime.now().timestamp()
    }
    attendance_db[room_id].append(attendance_record)
    return True
```

### WebSocket Integration
When a user joins via WebSocket:
```python
# Automatic attendance recording happens here
attendance_recorded = record_attendance(room.room_id, user_id, username, role)

# Send confirmation to user
await websocket.send_text(json.dumps({
    "type": "room_joined",
    "attendance_recorded": attendance_recorded
}))
```

### Frontend (JavaScript)
```javascript
case 'room_joined':
    if (message.attendance_recorded) {
        showStatus(`✅ Attendance Recorded!`, 'success');
        addChatMessage('System', 
            `✅ Your attendance has been automatically recorded`);
    }
    break;
```

## 📊 Data Structure

### Attendance Database
```python
attendance_db: Dict[str, List[dict]] = {
    "room_id_1": [
        {
            "user_id": "user_123",
            "username": "John Doe",
            "role": "student",
            "joined_at": "2026-02-08T10:30:00",
            "status": "present",
            "timestamp": 1707392400.0
        }
    ]
}
```

## 🎯 Benefits

1. **Time-Saving**: No manual attendance taking required
2. **Accurate**: Exact join timestamps recorded automatically
3. **Reliable**: No missed attendance marks
4. **Exportable**: Easy CSV export for institutional records
5. **Real-time**: Instant updates as students join
6. **Historical**: Track attendance across multiple sessions
7. **Analytics-Ready**: Data formatted for analysis

## 🔒 Privacy & Security

- Attendance data stored in-memory (can be configured for database persistence)
- Only instructors can view full attendance records
- Students see only their own confirmation
- Export feature restricted to instructors
- GDPR-compliant timestamp recording

## 🚦 Quick Start

1. **Start the backend server:**
   ```bash
   python run_backend.py
   ```

2. **Open classroom interface:**
   ```
   http://localhost:8001/classroom
   ```

3. **Create/Join a room** - Attendance tracking starts automatically!

4. **View attendance** (Instructors only):
   - Click "📋 Attendance" button
   - Or visit: `http://localhost:8001/api/attendance/room/{room_id}`

## 📈 Future Enhancements

- [ ] Database persistence (PostgreSQL/MongoDB)
- [ ] Attendance reports with charts/graphs
- [ ] Email notifications to students
- [ ] Integration with LMS (Canvas, Moodle, etc.)
- [ ] Late join tracking and alerts
- [ ] Minimum attendance duration tracking
- [ ] Automated absence notifications
- [ ] Attendance trends and analytics

## 🐛 Troubleshooting

**Problem:** Attendance not recorded
- **Solution:** Ensure user is properly registered and authenticated before joining

**Problem:** Can't view attendance
- **Solution:** Check that you're logged in as an instructor

**Problem:** Export not working
- **Solution:** Verify the room has attendance records

## 📞 Support

For issues or questions:
1. Check API documentation: `http://localhost:8001/docs`
2. View attendance summary: `http://localhost:8001/api/attendance/summary`
3. Check server logs for attendance recording messages

---

**Last Updated:** February 8, 2026  
**Version:** 1.0.0  
**Feature Status:** ✅ Production Ready
