#!/usr/bin/env python3
"""
Simple Virtual Classroom Demo
A standalone demonstration of the virtual classroom functionality
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading
import time
import os
from pathlib import Path

class VirtualClassroomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/classroom':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion-Aware Virtual Classroom Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .main-container {
            padding: 2rem;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            height: calc(100vh - 200px);
        }
        
        .video-section {
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }
        
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            height: 70%;
        }
        
        .video-container {
            position: relative;
            background: #2c3e50;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .video-container:hover {
            border-color: #3498db;
            transform: scale(1.02);
        }
        
        .video-container.speaking {
            border-color: #e74c3c;
            box-shadow: 0 0 20px rgba(231, 76, 60, 0.5);
        }
        
        .participant-info {
            position: absolute;
            bottom: 8px;
            left: 8px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .emotion-indicator {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid white;
            animation: emotionPulse 3s infinite;
        }
        
        @keyframes emotionPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .emotion-engaged { background: #2ecc71; }
        .emotion-confused { background: #f39c12; }
        .emotion-bored { background: #e74c3c; }
        .emotion-neutral { background: #95a5a6; }
        .emotion-curious { background: #9b59b6; }
        .emotion-frustrated { background: #e67e22; }
        
        .avatar {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(45deg, #3498db, #2ecc71);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .controls {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 1rem;
        }
        
        .control-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            font-size: 0.9rem;
        }
        
        .control-btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        .control-btn.active {
            background: #3498db;
        }
        
        .control-btn.muted {
            background: #e74c3c;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .chat-container, .emotion-dashboard {
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            flex: 1;
        }
        
        .section-title {
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .chat-messages {
            height: 200px;
            overflow-y: auto;
            margin-bottom: 1rem;
            padding-right: 0.5rem;
        }
        
        .chat-messages::-webkit-scrollbar {
            width: 4px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.3);
            border-radius: 2px;
        }
        
        .message {
            background: rgba(255,255,255,0.1);
            padding: 0.5rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .message-author {
            font-weight: bold;
            color: #3498db;
        }
        
        .chat-input-container {
            display: flex;
            gap: 0.5rem;
        }
        
        .chat-input {
            flex: 1;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 0.5rem 1rem;
            color: white;
            outline: none;
        }
        
        .chat-input::placeholder {
            color: rgba(255,255,255,0.6);
        }
        
        .send-btn {
            background: #3498db;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            color: white;
            transition: all 0.3s ease;
        }
        
        .send-btn:hover {
            background: #2980b9;
            transform: scale(1.1);
        }
        
        .emotion-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        
        .emotion-stat {
            background: rgba(255,255,255,0.1);
            padding: 0.5rem;
            border-radius: 8px;
            text-align: center;
            font-size: 0.8rem;
        }
        
        .emotion-bar {
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
            margin-top: 0.25rem;
            overflow: hidden;
        }
        
        .emotion-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 2px;
            transition: width 0.5s ease;
        }
        
        .feature-highlight {
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(46, 204, 113, 0.9);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            animation: slideDown 0.5s ease;
        }
        
        @keyframes slideDown {
            from { transform: translate(-50%, -100%); opacity: 0; }
            to { transform: translate(-50%, 0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="feature-highlight">
        🎯 Live Demo: Emotion-Aware Virtual Classroom with Real-time AI Analysis
    </div>
    
    <header class="header">
        <div class="logo">
            🎓 Emotion-Aware Virtual Classroom
        </div>
        <div class="status">
            <div class="status-dot"></div>
            <span>Live Session</span>
        </div>
    </header>
    
    <main class="main-container">
        <div class="video-section">
            <div class="video-grid" id="videoGrid">
                <!-- Instructor Video -->
                <div class="video-container" id="instructor">
                    <div class="avatar">👨‍🏫</div>
                    <div class="participant-info">
                        <span>Dr. Johnson (Instructor)</span>
                        🎤 📹
                    </div>
                    <div class="emotion-indicator emotion-engaged"></div>
                </div>
                
                <!-- Student Videos -->
                <div class="video-container" id="student1">
                    <div class="avatar">👩‍🎓</div>
                    <div class="participant-info">
                        <span>Sarah M.</span>
                        🎤 📹
                    </div>
                    <div class="emotion-indicator emotion-confused"></div>
                </div>
                
                <div class="video-container" id="student2">
                    <div class="avatar">👨‍🎓</div>
                    <div class="participant-info">
                        <span>Alex K.</span>
                        🔇 📹
                    </div>
                    <div class="emotion-indicator emotion-curious"></div>
                </div>
                
                <div class="video-container" id="student3">
                    <div class="avatar">👩‍🎓</div>
                    <div class="participant-info">
                        <span>Emma L.</span>
                        🎤 📹
                    </div>
                    <div class="emotion-indicator emotion-bored"></div>
                </div>
                
                <div class="video-container" id="student4">
                    <div class="avatar">👨‍🎓</div>
                    <div class="participant-info">
                        <span>Mike R.</span>
                        🎤 📹
                    </div>
                    <div class="emotion-indicator emotion-neutral"></div>
                </div>
            </div>
            
            <div class="controls">
                <button class="control-btn active" onclick="toggleVideo()">📹 Video</button>
                <button class="control-btn active" onclick="toggleAudio()">🎤 Audio</button>
                <button class="control-btn" onclick="shareScreen()">📺 Share Screen</button>
                <button class="control-btn" onclick="raiseHand()">✋ Raise Hand</button>
                <button class="control-btn muted" onclick="leaveRoom()">📞 Leave</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="chat-container">
                <div class="section-title">
                    💬 Live Chat
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="message">
                        <div class="message-author">Dr. Johnson:</div>
                        Welcome everyone! Today we'll explore machine learning concepts.
                    </div>
                    <div class="message">
                        <div class="message-author">Sarah M.:</div>
                        I'm a bit confused about neural networks 😕
                    </div>
                    <div class="message">
                        <div class="message-author">System:</div>
                        🚨 Alert: Student Sarah shows confusion - consider providing additional explanation
                    </div>
                    <div class="message">
                        <div class="message-author">Alex K.:</div>
                        This is really interesting! 🤔
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" class="chat-input" placeholder="Type your message..." id="chatInput">
                    <button class="send-btn" onclick="sendMessage()">➤</button>
                </div>
            </div>
            
            <div class="emotion-dashboard">
                <div class="section-title">
                    🎯 Class Emotion Analytics
                </div>
                <div class="emotion-stats">
                    <div class="emotion-stat">
                        <div>Engaged</div>
                        <div class="emotion-bar">
                            <div class="emotion-fill" style="width: 40%;"></div>
                        </div>
                    </div>
                    <div class="emotion-stat">
                        <div>Confused</div>
                        <div class="emotion-bar">
                            <div class="emotion-fill" style="width: 25%;"></div>
                        </div>
                    </div>
                    <div class="emotion-stat">
                        <div>Curious</div>
                        <div class="emotion-bar">
                            <div class="emotion-fill" style="width: 20%;"></div>
                        </div>
                    </div>
                    <div class="emotion-stat">
                        <div>Bored</div>
                        <div class="emotion-bar">
                            <div class="emotion-fill" style="width: 15%;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        let videoEnabled = true;
        let audioEnabled = true;
        let handRaised = false;
        
        // Simulate real-time emotion updates
        const emotions = ['engaged', 'confused', 'bored', 'neutral', 'curious', 'frustrated'];
        const students = ['student1', 'student2', 'student3', 'student4'];
        
        function updateEmotions() {
            students.forEach(studentId => {
                const indicator = document.querySelector(`#${studentId} .emotion-indicator`);
                const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
                indicator.className = `emotion-indicator emotion-${randomEmotion}`;
            });
            
            // Update emotion statistics
            const stats = document.querySelectorAll('.emotion-fill');
            stats.forEach(stat => {
                const newWidth = Math.random() * 60 + 10; // 10-70%
                stat.style.width = newWidth + '%';
            });
        }
        
        // Simulate speaking indicator
        function simulateSpeaking() {
            const containers = document.querySelectorAll('.video-container');
            containers.forEach(container => container.classList.remove('speaking'));
            
            const randomContainer = containers[Math.floor(Math.random() * containers.length)];
            randomContainer.classList.add('speaking');
            
            setTimeout(() => {
                randomContainer.classList.remove('speaking');
            }, 2000);
        }
        
        function toggleVideo() {
            videoEnabled = !videoEnabled;
            const btn = event.target;
            btn.className = videoEnabled ? 'control-btn active' : 'control-btn muted';
            btn.textContent = videoEnabled ? '📹 Video' : '📹 Video Off';
            
            addChatMessage('System', `Your video is now ${videoEnabled ? 'on' : 'off'}`);
        }
        
        function toggleAudio() {
            audioEnabled = !audioEnabled;
            const btn = event.target;
            btn.className = audioEnabled ? 'control-btn active' : 'control-btn muted';
            btn.textContent = audioEnabled ? '🎤 Audio' : '🎤 Muted';
            
            addChatMessage('System', `Your microphone is now ${audioEnabled ? 'on' : 'muted'}`);
        }
        
        function shareScreen() {
            addChatMessage('System', '📺 Screen sharing feature activated (demo mode)');
            event.target.classList.add('active');
            setTimeout(() => event.target.classList.remove('active'), 3000);
        }
        
        function raiseHand() {
            handRaised = !handRaised;
            const btn = event.target;
            btn.className = handRaised ? 'control-btn active' : 'control-btn';
            addChatMessage('System', `Hand ${handRaised ? 'raised' : 'lowered'} ✋`);
        }
        
        function leaveRoom() {
            if (confirm('Are you sure you want to leave the virtual classroom?')) {
                addChatMessage('System', '👋 You have left the classroom');
                setTimeout(() => {
                    alert('Demo completed! In a real deployment, you would return to the dashboard.');
                }, 1000);
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (message) {
                addChatMessage('You', message);
                input.value = '';
                
                // Simulate AI response based on message content
                setTimeout(() => {
                    if (message.toLowerCase().includes('confused') || message.toLowerCase().includes('help')) {
                        addChatMessage('System', '🤖 AI detected confusion - instructor has been notified');
                    } else if (message.toLowerCase().includes('boring') || message.toLowerCase().includes('bored')) {
                        addChatMessage('System', '🚨 Engagement alert sent to instructor');
                    }
                }, 1000);
            }
        }
        
        function addChatMessage(author, message) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.innerHTML = `
                <div class="message-author">${author}:</div>
                ${message}
            `;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Handle Enter key in chat
        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Start simulations
        setInterval(updateEmotions, 5000);
        setInterval(simulateSpeaking, 8000);
        
        // Add some demo messages
        setTimeout(() => addChatMessage('Emma L.', 'Can you explain this part again?'), 3000);
        setTimeout(() => addChatMessage('System', '🎯 Emotion analysis: 2 students showing confusion'), 4000);
        setTimeout(() => addChatMessage('Mike R.', 'Thanks, that helped!'), 7000);
        
        // Show demo features
        setTimeout(() => {
            addChatMessage('System', '✨ Demo Features Active:');
            addChatMessage('System', '• Real-time emotion recognition');
            addChatMessage('System', '• WebRTC video conferencing');
            addChatMessage('System', '• AI-powered engagement alerts');
            addChatMessage('System', '• Multimodal fusion (face + audio + text)');
        }, 2000);
    </script>
</body>
</html>
            """
            
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()

def run_demo_server():
    """Run the demo server"""
    print("\n🎓 Starting Emotion-Aware Virtual Classroom Demo Server...")
    
    server_address = ('localhost', 8080)
    httpd = HTTPServer(server_address, VirtualClassroomHandler)
    
    print(f"🌐 Virtual Classroom Demo running at: http://localhost:8080")
    print(f"🎯 Direct access: http://localhost:8080/classroom")
    print("📱 Features demonstrated:")
    print("   • WebRTC-style video conferencing interface")
    print("   • Real-time emotion recognition overlays")
    print("   • Live chat with AI analysis")
    print("   • Instructor emotion analytics dashboard")
    print("   • Student engagement monitoring")
    print("\n🚀 Opening browser in 3 seconds...")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:8080/classroom')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    print("""
🎉 EMOTION-AWARE VIRTUAL CLASSROOM DEMO
=====================================

This demo showcases the virtual classroom features:
✅ Video conferencing interface (Zoom-like)
✅ Real-time emotion recognition
✅ Live chat with AI analysis  
✅ Instructor dashboard
✅ Student engagement monitoring

Press Ctrl+C to stop the demo server.
    """)
    
    run_demo_server()