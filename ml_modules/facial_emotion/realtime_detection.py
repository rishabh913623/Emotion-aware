"""
Real-time Facial Emotion Detection
Week 2: Live webcam emotion recognition with face tracking
"""

import cv2
import torch
import numpy as np
import argparse
import time
from collections import deque
import json

from advanced_model import RealTimeEmotionDetector, EmotionRecognitionService

class EmotionWebcamApp:
    """Real-time emotion detection from webcam"""
    
    def __init__(self, model_path=None, camera_id=0, confidence_threshold=0.6):
        # Initialize emotion detector
        self.detector = RealTimeEmotionDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Camera setup
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # UI colors for emotions
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (255, 255, 0), # Cyan
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128),  # Brown
            'neutral': (128, 128, 128) # Gray
        }
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.frame_count = 0
        
    def draw_emotion_info(self, frame, results):
        """Draw emotion detection results on frame"""
        
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Get color for emotion
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label and confidence
            label = f"{emotion}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background for text
            cv2.rectangle(frame, (x, y-label_size[1]-10), 
                         (x+label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw emotion probabilities as bars
            if 'probabilities' in result:
                self.draw_emotion_bars(frame, result['probabilities'], x, y+h+10)
    
    def draw_emotion_bars(self, frame, probabilities, x, y):
        """Draw emotion probability bars"""
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        bar_width = 60
        bar_height = 10
        
        for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
            bar_x = x
            bar_y = y + i * (bar_height + 5)
            
            # Draw background bar
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Draw probability bar
            prob_width = int(bar_width * prob)
            color = self.emotion_colors.get(emotion.lower(), (255, 255, 255))
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + prob_width, bar_y + bar_height), 
                         color, -1)
            
            # Draw emotion name
            cv2.putText(frame, f"{emotion}: {prob:.2f}", 
                       (bar_x + bar_width + 5, bar_y + bar_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def draw_performance_info(self, frame):
        """Draw performance information on frame"""
        # Calculate FPS
        if len(self.fps_queue) > 1:
            fps = len(self.fps_queue) / (self.fps_queue[-1] - self.fps_queue[0])
        else:
            fps = 0
        
        # Get detector stats
        stats = self.detector.get_performance_stats()
        
        # Draw performance info
        info_text = [
            f"FPS: {fps:.1f}",
            f"Processing: {stats['avg_processing_time']*1000:.1f}ms",
            f"Device: {stats['device']}",
            f"Frames: {self.frame_count}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25
    
    def run(self):
        """Main application loop"""
        print("Starting emotion detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset stats")
        
        while True:
            start_time = time.time()
            
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect emotions
            results = self.detector.detect_emotions_in_frame(frame)
            
            # Draw results
            self.draw_emotion_info(frame, results)
            self.draw_performance_info(frame)
            
            # Draw instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save, 'r' to reset", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update performance tracking
            self.fps_queue.append(time.time())
            self.frame_count += 1
            
            # Show frame
            cv2.imshow('Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"emotion_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset performance stats
                self.detector.reset_performance_stats()
                self.fps_queue.clear()
                self.frame_count = 0
                print("Performance stats reset")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        final_stats = self.detector.get_performance_stats()
        print("\\nFinal Performance Statistics:")
        print(f"Total frames processed: {final_stats['total_frames_processed']}")
        print(f"Average processing time: {final_stats['avg_processing_time']*1000:.1f}ms")
        print(f"Final FPS: {final_stats['fps']:.1f}")

class EmotionVideoProcessor:
    """Process video file for emotion detection"""
    
    def __init__(self, model_path=None, confidence_threshold=0.6):
        self.detector = RealTimeEmotionDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
    
    def process_video(self, input_path, output_path=None, save_results=True):
        """Process video file and optionally save annotated video"""
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        results_data = []
        frame_number = 0
        
        app = EmotionWebcamApp()  # For drawing functions
        
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect emotions
                results = self.detector.detect_emotions_in_frame(frame)
                
                # Store results
                frame_data = {
                    'frame_number': frame_number,
                    'timestamp': frame_number / fps,
                    'detections': results
                }
                results_data.append(frame_data)
                
                # Annotate frame if output video requested
                if out is not None:
                    app.draw_emotion_info(frame, results)
                    out.write(frame)
                
                frame_number += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        
        # Save results to JSON
        if save_results:
            results_path = input_path.replace('.mp4', '_emotion_results.json')
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to: {results_path}")
        
        return results_data

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Detection')
    
    parser.add_argument('--model_path', type=str, 
                       help='Path to trained emotion model')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video'], 
                       default='webcam', help='Detection mode')
    parser.add_argument('--input_video', type=str,
                       help='Input video file path (for video mode)')
    parser.add_argument('--output_video', type=str,
                       help='Output video file path (optional)')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Confidence threshold for emotion detection')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        # Real-time webcam detection
        app = EmotionWebcamApp(
            model_path=args.model_path,
            camera_id=args.camera_id,
            confidence_threshold=args.confidence_threshold
        )
        app.run()
        
    elif args.mode == 'video':
        if not args.input_video:
            print("Error: --input_video required for video mode")
            return
        
        # Video file processing
        processor = EmotionVideoProcessor(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold
        )
        
        results = processor.process_video(
            args.input_video, 
            args.output_video
        )
        
        print(f"\\nProcessed {len(results)} frames")
        
        # Print summary statistics
        total_detections = sum(len(frame['detections']) for frame in results)
        print(f"Total emotion detections: {total_detections}")
        
        if total_detections > 0:
            emotion_counts = {}
            for frame in results:
                for detection in frame['detections']:
                    emotion = detection['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print("\\nEmotion distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                percentage = (count / total_detections) * 100
                print(f"{emotion}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()