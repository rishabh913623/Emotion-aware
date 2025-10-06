#!/usr/bin/env python3
"""
Week 10 - Testing & Demo
Comprehensive stress testing for the emotion-aware virtual classroom system
Tests system with 100+ concurrent students and various load scenarios
"""

import asyncio
import aiohttp
import json
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test configuration
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
MAX_CONCURRENT_STUDENTS = 100
TEST_DURATION_MINUTES = 10
CLASS_ID = "StressTest-CS101-Fall2024"

class StressTestRunner:
    def __init__(self):
        self.results = {
            'total_students': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'total_emotion_updates': 0,
            'avg_response_time': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        self.active_sessions = []
        
    async def simulate_student_session(self, session_id: int, session: aiohttp.ClientSession):
        """Simulate a single student session with emotion data"""
        student_id = f"student_{session_id:03d}"
        
        try:
            # 1. Student login/authentication
            auth_start = time.time()
            auth_data = {
                "username": student_id,
                "password": "test_password_123"
            }
            
            async with session.post(f"{BACKEND_URL}/api/auth/login", json=auth_data) as response:
                if response.status == 200:
                    auth_token = (await response.json()).get('access_token')
                else:
                    # Create user if not exists
                    create_data = {
                        "username": student_id,
                        "email": f"{student_id}@test.edu",
                        "password": "test_password_123",
                        "role": "student"
                    }
                    async with session.post(f"{BACKEND_URL}/api/auth/register", json=create_data) as reg_response:
                        if reg_response.status == 201:
                            async with session.post(f"{BACKEND_URL}/api/auth/login", json=auth_data) as login_response:
                                auth_token = (await login_response.json()).get('access_token')
                        else:
                            raise Exception(f"Failed to create/login user: {reg_response.status}")
            
            auth_time = time.time() - auth_start
            
            # 2. Connect to WebSocket for real-time emotion data
            headers = {"Authorization": f"Bearer {auth_token}"}
            ws_url = f"{WS_URL}/api/emotion/ws/emotion_stream/{CLASS_ID}/{student_id}"
            
            async with session.ws_connect(ws_url, headers=headers) as ws:
                self.results['successful_connections'] += 1
                
                # Simulate student activity for test duration
                end_time = time.time() + (TEST_DURATION_MINUTES * 60)
                emotion_count = 0
                
                while time.time() < end_time:
                    # Generate realistic emotion data
                    emotion_data = self.generate_emotion_data(session_id)
                    
                    # Send emotion update
                    update_start = time.time()
                    await ws.send_str(json.dumps({
                        'type': 'emotion_update',
                        'data': emotion_data,
                        'timestamp': datetime.now().isoformat()
                    }))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(ws.receive(), timeout=5.0)
                        if response.type == aiohttp.WSMsgType.TEXT:
                            update_time = time.time() - update_start
                            self.results['avg_response_time'] += update_time
                            emotion_count += 1
                    except asyncio.TimeoutError:
                        self.results['errors'].append(f"Timeout for student {student_id}")
                    
                    # Simulate realistic student interaction patterns
                    await asyncio.sleep(random.uniform(2.0, 8.0))  # 2-8 seconds between updates
                
                self.results['total_emotion_updates'] += emotion_count
                
        except Exception as e:
            self.results['failed_connections'] += 1
            self.results['errors'].append(f"Student {student_id}: {str(e)}")
    
    def generate_emotion_data(self, session_id: int) -> dict:
        """Generate realistic emotion data for testing"""
        # Simulate different student personality types
        personality_type = session_id % 4
        
        if personality_type == 0:  # Engaged student
            emotions = {
                'happy': random.uniform(0.6, 0.9),
                'engaged': random.uniform(0.7, 0.95),
                'neutral': random.uniform(0.1, 0.3),
                'confused': random.uniform(0.0, 0.2),
                'bored': random.uniform(0.0, 0.1),
                'frustrated': random.uniform(0.0, 0.1)
            }
        elif personality_type == 1:  # Struggling student
            emotions = {
                'confused': random.uniform(0.5, 0.8),
                'frustrated': random.uniform(0.3, 0.7),
                'neutral': random.uniform(0.2, 0.4),
                'engaged': random.uniform(0.1, 0.3),
                'happy': random.uniform(0.0, 0.2),
                'bored': random.uniform(0.1, 0.3)
            }
        elif personality_type == 2:  # Bored student
            emotions = {
                'bored': random.uniform(0.6, 0.9),
                'neutral': random.uniform(0.3, 0.6),
                'engaged': random.uniform(0.0, 0.2),
                'happy': random.uniform(0.0, 0.1),
                'confused': random.uniform(0.0, 0.2),
                'frustrated': random.uniform(0.1, 0.3)
            }
        else:  # Variable student
            emotions = {
                emotion: random.uniform(0.0, 0.8)
                for emotion in ['happy', 'engaged', 'neutral', 'confused', 'bored', 'frustrated']
            }
        
        # Normalize emotions to sum to 1
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        return {
            'facial_emotions': emotions,
            'audio_emotions': emotions,  # Simplified for testing
            'text_sentiment': {
                'positive': emotions.get('happy', 0) + emotions.get('engaged', 0),
                'negative': emotions.get('frustrated', 0) + emotions.get('bored', 0),
                'neutral': emotions.get('neutral', 0)
            },
            'learning_state': self.map_to_learning_state(emotions),
            'confidence': random.uniform(0.7, 0.95)
        }
    
    def map_to_learning_state(self, emotions: dict) -> str:
        """Map emotions to learning states"""
        if emotions['engaged'] > 0.5:
            return 'engaged'
        elif emotions['confused'] > 0.4:
            return 'confused'
        elif emotions['bored'] > 0.5:
            return 'bored'
        elif emotions['frustrated'] > 0.4:
            return 'frustrated'
        else:
            return 'neutral'
    
    async def test_api_endpoints(self, session: aiohttp.ClientSession):
        """Test API endpoint performance under load"""
        print("🔧 Testing API endpoints...")
        
        endpoints = [
            f"/api/dashboard/api/class/{CLASS_ID}/current-state",
            f"/api/dashboard/api/class/{CLASS_ID}/summary",
            f"/api/reports/api/class/{CLASS_ID}/analytics/overview",
            f"/api/reports/api/class/{CLASS_ID}/timeline?hours=1"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            try:
                async with session.get(f"{BACKEND_URL}{endpoint}") as response:
                    response_time = time.time() - start_time
                    print(f"  ✓ {endpoint}: {response.status} ({response_time:.3f}s)")
            except Exception as e:
                print(f"  ❌ {endpoint}: {str(e)}")
    
    async def run_stress_test(self):
        """Run the complete stress test"""
        print(f"🚀 Starting Stress Test - {MAX_CONCURRENT_STUDENTS} concurrent students")
        print("=" * 80)
        
        self.results['start_time'] = datetime.now()
        self.results['total_students'] = MAX_CONCURRENT_STUDENTS
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Test API endpoints first
            await self.test_api_endpoints(session)
            
            print(f"\n🎯 Simulating {MAX_CONCURRENT_STUDENTS} concurrent student sessions...")
            
            # Create tasks for all student sessions
            tasks = []
            for i in range(MAX_CONCURRENT_STUDENTS):
                task = asyncio.create_task(self.simulate_student_session(i, session))
                tasks.append(task)
                
                # Add small delay between connections to avoid overwhelming the server
                if i % 10 == 0:
                    await asyncio.sleep(0.1)
            
            # Wait for all sessions to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.results['end_time'] = datetime.now()
        
        # Calculate final metrics
        if self.results['total_emotion_updates'] > 0:
            self.results['avg_response_time'] /= self.results['total_emotion_updates']
        
        self.print_results()
    
    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 80)
        print("📊 STRESS TEST RESULTS")
        print("=" * 80)
        
        duration = (self.results['end_time'] - self.results['start_time']).total_seconds()
        
        print(f"Test Duration: {duration:.1f} seconds")
        print(f"Total Students: {self.results['total_students']}")
        print(f"Successful Connections: {self.results['successful_connections']}")
        print(f"Failed Connections: {self.results['failed_connections']}")
        print(f"Connection Success Rate: {(self.results['successful_connections']/self.results['total_students']*100):.1f}%")
        print(f"Total Emotion Updates: {self.results['total_emotion_updates']}")
        print(f"Average Response Time: {self.results['avg_response_time']:.3f}s")
        print(f"Updates per Second: {(self.results['total_emotion_updates']/duration):.2f}")
        
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.results['errors']) > 10:
                print(f"  ... and {len(self.results['errors']) - 10} more errors")
        
        # Performance evaluation
        print("\n🎯 PERFORMANCE EVALUATION:")
        
        success_rate = self.results['successful_connections'] / self.results['total_students']
        response_time = self.results['avg_response_time']
        
        if success_rate >= 0.95 and response_time < 1.0:
            print("🟢 EXCELLENT: System handles load very well")
        elif success_rate >= 0.90 and response_time < 2.0:
            print("🟡 GOOD: System performs well under load")
        elif success_rate >= 0.80 and response_time < 3.0:
            print("🟠 ACCEPTABLE: System shows some strain but functional")
        else:
            print("🔴 POOR: System struggles with current load")
        
        print(f"\nRecommendations:")
        if response_time > 2.0:
            print("- Consider adding more backend replicas")
            print("- Optimize database queries and add caching")
        if success_rate < 0.90:
            print("- Increase connection limits and timeouts")
            print("- Add circuit breakers for fault tolerance")
        
        print("\n✅ Stress test completed successfully!")

async def main():
    """Main test function"""
    try:
        tester = StressTestRunner()
        await tester.run_stress_test()
        return True
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Week 10 - Comprehensive Stress Testing")
    success = asyncio.run(main())
    sys.exit(0 if success else 1)