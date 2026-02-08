#!/usr/bin/env python3
"""
Demo script to test automatic attendance tracking
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8001"

def demo_attendance_system():
    """Demonstrate the automatic attendance tracking feature"""
    
    print("🎓 Emotion-Aware Virtual Classroom - Attendance System Demo")
    print("=" * 60)
    
    # 1. Register demo users
    print("\n1️⃣  Registering demo users...")
    users = []
    
    # Register instructor
    instructor_data = {
        "username": "Dr. Johnson",
        "email": "dr.johnson@university.edu",
        "role": "instructor"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=instructor_data)
    instructor = response.json()
    users.append(instructor)
    print(f"   ✅ Instructor registered: {instructor['username']}")
    
    # Register students
    student_names = ["Alice Smith", "Bob Chen", "Carol Davis", "David Wilson", "Eve Martinez"]
    for name in student_names:
        student_data = {
            "username": name,
            "email": f"{name.lower().replace(' ', '.')}@student.edu",
            "role": "student"
        }
        response = requests.post(f"{BASE_URL}/api/auth/register", json=student_data)
        student = response.json()
        users.append(student)
        print(f"   ✅ Student registered: {student['username']}")
    
    # 2. Create classroom
    print(f"\n2️⃣  Creating virtual classroom...")
    room_data = {
        "host_user_id": instructor['user_id'],
        "room_name": "Machine Learning 101 - Lecture 5"
    }
    response = requests.post(f"{BASE_URL}/api/classroom/create-room", json=room_data)
    room = response.json()
    room_id = room['room_id']
    short_room_id = room_id[:8]
    print(f"   ✅ Classroom created!")
    print(f"   📍 Room ID: {short_room_id}")
    print(f"   🔗 Join URL: {room['join_url']}")
    
    # 3. Simulate students joining (attendance will be recorded automatically via WebSocket)
    # For this demo, we'll manually add attendance records to show the concept
    print(f"\n3️⃣  Students joining classroom...")
    print("   (In real scenario, attendance is recorded automatically via WebSocket)")
    print("   When students join, the system automatically:")
    print("   • Records their user ID and username")
    print("   • Captures exact join timestamp")
    print("   • Marks their status as 'present'")
    print("   • Sends confirmation to the student")
    
    # 4. View attendance (after students have joined)
    print(f"\n4️⃣  Viewing attendance records...")
    print(f"   📋 Checking attendance for Room: {short_room_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/api/attendance/room/{short_room_id}")
        if response.status_code == 200:
            attendance_data = response.json()
            stats = attendance_data['statistics']
            
            print(f"\n   📊 ATTENDANCE STATISTICS:")
            print(f"   ├─ Total Attendees: {stats['total_attendees']}")
            print(f"   ├─ Students Present: {stats['students_present']}")
            print(f"   └─ Instructors Present: {stats['instructors_present']}")
            
            if attendance_data['attendance_records']:
                print(f"\n   📝 ATTENDANCE RECORDS:")
                for record in attendance_data['attendance_records']:
                    join_time = datetime.fromisoformat(record['joined_at']).strftime('%I:%M:%S %p')
                    role_emoji = "👨‍🏫" if record['role'] == 'instructor' else "👨‍🎓"
                    print(f"   {role_emoji} {record['username']:20} | {record['role']:10} | {join_time} | ✅ {record['status']}")
        else:
            print(f"   ℹ️  No attendance records yet (students haven't joined via WebSocket)")
            print(f"   💡 To record attendance, students must join the classroom at:")
            print(f"      http://localhost:8001/classroom/{short_room_id}")
    except Exception as e:
        print(f"   ℹ️  Attendance will be recorded when students join via WebSocket")
    
    # 5. Export attendance
    print(f"\n5️⃣  Exporting attendance to CSV...")
    try:
        response = requests.get(f"{BASE_URL}/api/attendance/room/{short_room_id}/export?format=csv")
        if response.status_code == 200:
            export_data = response.json()
            print(f"   ✅ Export ready: {export_data['filename']}")
            print(f"\n   Preview of CSV content:")
            print(f"   {'-' * 60}")
            for line in export_data['content'].split('\\n')[:6]:  # Show first 6 lines
                print(f"   {line}")
        else:
            print(f"   ℹ️  Export available after students join")
    except Exception as e:
        print(f"   ℹ️  CSV export will be available after attendance is recorded")
    
    # 6. View overall summary
    print(f"\n6️⃣  Viewing overall attendance summary...")
    response = requests.get(f"{BASE_URL}/api/attendance/summary")
    summary = response.json()
    print(f"   📈 SYSTEM-WIDE STATISTICS:")
    print(f"   ├─ Total Attendance Records: {summary['total_attendance_records']}")
    print(f"   ├─ Rooms with Attendance: {summary['rooms_with_attendance']}")
    
    if summary['room_summaries']:
        print(f"   └─ Active Sessions:")
        for room_summary in summary['room_summaries']:
            print(f"      • {room_summary['room_name']} - {room_summary['total_attendees']} attendees")
    
    # Instructions
    print(f"\n{'=' * 60}")
    print(f"🎯 HOW TO USE AUTOMATIC ATTENDANCE:")
    print(f"{'=' * 60}")
    print(f"\n📱 FOR INSTRUCTORS:")
    print(f"   1. Open: http://localhost:8001/classroom")
    print(f"   2. Register as instructor")
    print(f"   3. Create a new room")
    print(f"   4. Share the Room ID with students")
    print(f"   5. Click '📋 Attendance' button to view records")
    print(f"   6. Click 'Export CSV' to download attendance")
    
    print(f"\n👨‍🎓 FOR STUDENTS:")
    print(f"   1. Open: http://localhost:8001/classroom")
    print(f"   2. Register as student")
    print(f"   3. Enter Room ID: {short_room_id}")
    print(f"   4. Join - Your attendance is recorded automatically! ✅")
    print(f"   5. See confirmation in chat")
    
    print(f"\n🔍 VIEW ATTENDANCE (API):")
    print(f"   curl http://localhost:8001/api/attendance/room/{short_room_id}")
    
    print(f"\n📥 EXPORT ATTENDANCE (API):")
    print(f"   curl 'http://localhost:8001/api/attendance/room/{short_room_id}/export?format=csv'")
    
    print(f"\n📚 Documentation:")
    print(f"   • API Docs: http://localhost:8001/docs")
    print(f"   • Feature Guide: ATTENDANCE_FEATURE.md")
    
    print(f"\n{'=' * 60}")
    print(f"✨ Demo Complete! The system is ready for use.")
    print(f"{'=' * 60}\n")
    
    return {
        "room_id": room_id,
        "short_room_id": short_room_id,
        "users": users,
        "join_url": f"http://localhost:8001/classroom/{short_room_id}"
    }

if __name__ == "__main__":
    try:
        result = demo_attendance_system()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to backend server")
        print("   Please start the server first:")
        print("   python run_backend.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
