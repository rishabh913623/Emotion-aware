"""
Text Sentiment Analysis Module
Week 4: HuggingFace Transformer-based sentiment analysis for virtual classroom chat
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, PreTrainedTokenizer, PreTrainedModel
)
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import time
import re
import string
from collections import deque
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """Text preprocessing for chat messages"""
    
    def __init__(self):
        # Common abbreviations and internet slang
        self.abbreviations = {
            "u": "you", "ur": "your", "youre": "you are", "cant": "cannot",
            "wont": "will not", "dont": "do not", "isnt": "is not",
            "arent": "are not", "wasnt": "was not", "werent": "were not",
            "lol": "laugh out loud", "omg": "oh my god", "wtf": "what the hell",
            "btw": "by the way", "imo": "in my opinion", "tbh": "to be honest",
            "idk": "i do not know", "thx": "thanks", "thnx": "thanks",
            "gonna": "going to", "wanna": "want to", "gotta": "got to",
            "kinda": "kind of", "sorta": "sort of", "dunno": "do not know",
            "yep": "yes", "nope": "no", "ok": "okay", "k": "okay"
        }
        
        # Emoticons to sentiment mapping
        self.emoticons = {
            ":)": "happy", ":]": "happy", "=)": "happy", ":D": "very happy",
            ":(": "sad", ":[": "sad", "=(": "sad", ":'/": "sad",
            ":P": "playful", ":p": "playful", ";)": "playful", ";D": "playful",
            ":o": "surprised", ":O": "surprised", ":-o": "surprised",
            ":/": "confused", ":\\": "confused", ":|": "neutral",
            "<3": "love", "</3": "heartbroken", "^_^": "happy",
            "-_-": "annoyed", ">:(": "angry", "XD": "very happy"
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', '', text)
        
        # Replace abbreviations
        words = text.split()
        words = [self.abbreviations.get(word, word) for word in words]
        text = ' '.join(words)
        
        # Handle emoticons (extract sentiment before removing them)
        emoticon_sentiments = []
        for emoticon, sentiment in self.emoticons.items():
            if emoticon in text:
                emoticon_sentiments.append(sentiment)
                text = text.replace(emoticon, f" {sentiment} ")
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Union[int, float, bool]]:
        """Extract text features for enhanced analysis"""
        features = {}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Emotional indicators
        positive_words = ['good', 'great', 'awesome', 'amazing', 'excellent', 'fantastic', 
                         'wonderful', 'perfect', 'love', 'like', 'happy', 'glad', 'excited']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                         'sad', 'angry', 'frustrated', 'confused', 'difficult', 'hard']
        
        words = text.lower().split()
        features['positive_word_count'] = sum(1 for word in words if word in positive_words)
        features['negative_word_count'] = sum(1 for word in words if word in negative_words)
        
        # Learning-related keywords
        learning_positive = ['understand', 'clear', 'makes sense', 'got it', 'interesting', 
                           'cool', 'learned', 'helpful', 'easy', 'simple']
        learning_negative = ['confused', 'lost', 'dont understand', 'hard', 'difficult', 
                           'unclear', 'boring', 'complicated', 'stuck']
        
        features['learning_positive_count'] = sum(1 for phrase in learning_positive if phrase in text.lower())
        features['learning_negative_count'] = sum(1 for phrase in learning_negative if phrase in text.lower())
        
        return features

class EducationalSentimentAnalyzer:
    """Enhanced sentiment analyzer for educational contexts"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "auto",
        confidence_threshold: float = 0.7
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load pre-trained model
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Initialize model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            print(f"Loaded sentiment model: {model_name}")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to simpler approach
            self.sentiment_pipeline = None
            self.model = None
            self.tokenizer = None
        
        # Text preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Learning-focused sentiment mapping
        self.learning_sentiment_map = {
            'POSITIVE': 'engaged',
            'NEGATIVE': 'frustrated',
            'engaged': 'engaged',
            'frustrated': 'frustrated',
            'confused': 'confused',
            'bored': 'bored',
            'curious': 'curious',
            'neutral': 'neutral'
        }
        
        # Temporal smoothing
        self.sentiment_history = deque(maxlen=5)
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
    
    def analyze_sentiment_basic(self, text: str) -> Dict[str, Union[str, float]]:
        """Basic rule-based sentiment analysis (fallback)"""
        
        features = self.preprocessor.extract_features(text)
        
        # Simple scoring based on keywords
        positive_score = features['positive_word_count'] + features['learning_positive_count']
        negative_score = features['negative_word_count'] + features['learning_negative_count']
        
        # Consider punctuation
        if features['exclamation_count'] > 0:
            if positive_score > negative_score:
                positive_score += 1
            else:
                negative_score += 1
        
        # Determine sentiment
        if positive_score > negative_score:
            sentiment = 'POSITIVE'
            confidence = min(0.6 + 0.1 * (positive_score - negative_score), 1.0)
        elif negative_score > positive_score:
            sentiment = 'NEGATIVE'
            confidence = min(0.6 + 0.1 * (negative_score - positive_score), 1.0)
        else:
            sentiment = 'NEUTRAL'
            confidence = 0.5
        
        # Map to learning states
        if features['learning_negative_count'] > 0:
            learning_state = 'confused'
        elif features['learning_positive_count'] > 0:
            learning_state = 'engaged'
        else:
            learning_state = self.learning_sentiment_map.get(sentiment, 'neutral')
        
        return {
            'sentiment': sentiment,
            'learning_state': learning_state,
            'confidence': confidence,
            'features': features
        }
    
    def analyze_sentiment_transformer(self, text: str) -> Dict[str, Union[str, float, List]]:
        """Advanced transformer-based sentiment analysis"""
        
        try:
            # Get predictions from transformer
            results = self.sentiment_pipeline(text)
            
            # Extract sentiment scores
            sentiment_scores = {result['label']: result['score'] for result in results[0]}
            
            # Get primary sentiment
            primary_result = max(results[0], key=lambda x: x['score'])
            sentiment = primary_result['label']
            confidence = primary_result['score']
            
            # Get text features for enhanced analysis
            features = self.preprocessor.extract_features(text)
            
            # Map to learning states with context awareness
            learning_state = self._map_to_learning_state(sentiment, confidence, features, text)
            
            # Enhanced confidence calculation
            enhanced_confidence = self._calculate_enhanced_confidence(
                confidence, features, sentiment_scores
            )
            
            return {
                'sentiment': sentiment,
                'learning_state': learning_state,
                'confidence': enhanced_confidence,
                'raw_confidence': confidence,
                'sentiment_scores': sentiment_scores,
                'features': features
            }
            
        except Exception as e:
            print(f"Transformer analysis error: {e}")
            # Fall back to basic analysis
            return self.analyze_sentiment_basic(text)
    
    def _map_to_learning_state(self, sentiment: str, confidence: float, features: Dict, text: str) -> str:
        """Map sentiment to learning-specific states"""
        
        text_lower = text.lower()
        
        # Direct learning indicators
        confusion_indicators = [
            'confused', 'lost', 'dont understand', 'what', 'how', 'help',
            'stuck', 'unclear', 'not sure', 'dont get it'
        ]
        
        engagement_indicators = [
            'understand', 'got it', 'makes sense', 'clear', 'interesting',
            'cool', 'awesome', 'learned something', 'helpful', 'good explanation'
        ]
        
        boredom_indicators = [
            'boring', 'tired', 'sleepy', 'when will this end', 'slow',
            'drag', 'repetitive', 'already know'
        ]
        
        curiosity_indicators = [
            'interesting', 'want to know', 'tell me more', 'how does',
            'why', 'what if', 'curious', 'explore', 'learn more'
        ]
        
        # Check for specific indicators
        if any(indicator in text_lower for indicator in confusion_indicators):
            return 'confused'
        elif any(indicator in text_lower for indicator in engagement_indicators):
            return 'engaged'
        elif any(indicator in text_lower for indicator in boredom_indicators):
            return 'bored'
        elif any(indicator in text_lower for indicator in curiosity_indicators):
            return 'curious'
        
        # Use learning-specific feature counts
        if features['learning_negative_count'] > 0:
            return 'confused'
        elif features['learning_positive_count'] > 0:
            return 'engaged'
        
        # Fall back to general sentiment mapping
        return self.learning_sentiment_map.get(sentiment, 'neutral')
    
    def _calculate_enhanced_confidence(self, base_confidence: float, features: Dict, sentiment_scores: Dict) -> float:
        """Calculate enhanced confidence score"""
        
        # Start with base confidence
        confidence = base_confidence
        
        # Boost confidence for clear indicators
        if features['exclamation_count'] > 0:
            confidence = min(confidence * 1.1, 1.0)
        
        if features['question_count'] > 0:
            confidence = min(confidence * 1.05, 1.0)
        
        # Boost for learning-specific language
        if features['learning_positive_count'] > 0 or features['learning_negative_count'] > 0:
            confidence = min(confidence * 1.15, 1.0)
        
        # Consider sentiment distribution
        if len(sentiment_scores) >= 2:
            sorted_scores = sorted(sentiment_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]
                confidence = confidence * (0.5 + 0.5 * margin)
        
        return min(confidence, 1.0)
    
    def smooth_predictions(self, current_sentiment: str, current_confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to predictions"""
        
        self.sentiment_history.append((current_sentiment, current_confidence))
        
        if len(self.sentiment_history) < 3:
            return current_sentiment, current_confidence
        
        # Weighted voting with recency bias
        sentiment_votes = {}
        total_weight = 0
        
        for i, (sentiment, confidence) in enumerate(self.sentiment_history):
            weight = (i + 1) * confidence  # Recent predictions with higher confidence get more weight
            
            if sentiment not in sentiment_votes:
                sentiment_votes[sentiment] = 0
            sentiment_votes[sentiment] += weight
            total_weight += weight
        
        # Find best sentiment
        if total_weight > 0:
            best_sentiment = max(sentiment_votes, key=sentiment_votes.get)
            smoothed_confidence = sentiment_votes[best_sentiment] / total_weight
            
            # Boost confidence for stable predictions
            stability_boost = len([s for s, c in self.sentiment_history if s == best_sentiment]) / len(self.sentiment_history)
            smoothed_confidence = min(smoothed_confidence * (0.8 + 0.4 * stability_boost), 1.0)
            
            return best_sentiment, smoothed_confidence
        
        return current_sentiment, current_confidence
    
    def analyze_text(self, text: str, use_smoothing: bool = True) -> Dict[str, Union[str, float, Dict]]:
        """Main text analysis function"""
        
        start_time = time.time()
        
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return {
                'sentiment': 'neutral',
                'learning_state': 'neutral',
                'confidence': 0.5,
                'text_length': 0,
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'cleaned_text': cleaned_text
            }
        
        # Perform analysis
        if self.sentiment_pipeline:
            result = self.analyze_sentiment_transformer(cleaned_text)
        else:
            result = self.analyze_sentiment_basic(cleaned_text)
        
        # Apply smoothing if requested
        if use_smoothing:
            smoothed_sentiment, smoothed_confidence = self.smooth_predictions(
                result['learning_state'], result['confidence']
            )
            result['smoothed_learning_state'] = smoothed_sentiment
            result['smoothed_confidence'] = smoothed_confidence
        
        # Add metadata
        result['text_length'] = len(text)
        result['cleaned_text'] = cleaned_text
        result['processing_time'] = time.time() - start_time
        result['timestamp'] = time.time()
        
        # Track performance
        self.processing_times.append(result['processing_time'])
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'avg_processing_time': avg_processing_time,
            'model_name': self.model_name,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'total_processed': len(self.processing_times)
        }
    
    def reset_performance_stats(self):
        """Reset performance counters"""
        self.processing_times.clear()
        self.sentiment_history.clear()

class TextSentimentService:
    """High-level service for text sentiment analysis in virtual classroom"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.analyzer = EducationalSentimentAnalyzer(model_name=model_name)
        self.active_sessions = {}  # Track active chat sessions
        
    def start_session(self, session_id: str, user_id: str):
        """Start text analysis session"""
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'start_time': time.time(),
            'messages_processed': 0,
            'sentiment_history': []
        }
    
    def end_session(self, session_id: str):
        """End text analysis session"""
        if session_id in self.active_sessions:
            session_data = self.active_sessions.pop(session_id)
            return {
                'session_id': session_id,
                'duration': time.time() - session_data['start_time'],
                'total_messages': session_data['messages_processed'],
                'sentiment_summary': self._generate_sentiment_summary(session_data['sentiment_history'])
            }
        return None
    
    def process_message(self, text: str, session_id: str) -> Dict:
        """Process chat message for sentiment analysis"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Analyze sentiment
        result = self.analyzer.analyze_text(text)
        
        # Update session data
        session = self.active_sessions[session_id]
        session['messages_processed'] += 1
        
        if result['learning_state'] not in ['error', 'neutral'] or result['confidence'] > 0.6:
            session['sentiment_history'].append(result)
        
        result['session_id'] = session_id
        result['message_number'] = session['messages_processed']
        
        return result
    
    def _generate_sentiment_summary(self, sentiment_history: List[Dict]) -> Dict:
        """Generate summary statistics for sentiment history"""
        
        if not sentiment_history:
            return {}
        
        # Count learning states
        state_counts = {}
        total_messages = len(sentiment_history)
        
        for result in sentiment_history:
            state = result.get('smoothed_learning_state', result.get('learning_state', 'neutral'))
            if state not in state_counts:
                state_counts[state] = 0
            state_counts[state] += 1
        
        # Calculate percentages
        state_percentages = {
            state: (count / total_messages) * 100
            for state, count in state_counts.items()
        }
        
        # Find dominant state
        dominant_state = max(state_counts.items(), key=lambda x: x[1]) if state_counts else ('neutral', 0)
        
        # Calculate average confidence
        avg_confidence = np.mean([
            s.get('smoothed_confidence', s.get('confidence', 0.5)) 
            for s in sentiment_history
        ])
        
        return {
            'total_messages': total_messages,
            'learning_state_distribution': state_percentages,
            'dominant_state': dominant_state[0],
            'dominant_state_percentage': (dominant_state[1] / total_messages) * 100,
            'average_confidence': float(avg_confidence)
        }
    
    def analyze_batch_messages(self, messages: List[str]) -> List[Dict]:
        """Analyze multiple messages in batch"""
        
        results = []
        for message in messages:
            result = self.analyzer.analyze_text(message, use_smoothing=False)
            results.append(result)
        
        return results

# Global service instance
text_sentiment_service = TextSentimentService()