"""
Multimodal Fusion Module
Week 5: Advanced fusion of facial, audio, and text emotion recognition for learning state prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import time
import json
from dataclasses import dataclass

@dataclass
class ModalityInput:
    """Structure for individual modality inputs"""
    emotion: str
    confidence: float
    raw_probabilities: Optional[np.ndarray] = None
    features: Optional[Dict] = None
    timestamp: Optional[float] = None

@dataclass
class FusionOutput:
    """Structure for fusion output"""
    learning_state: str
    confidence: float
    modality_contributions: Dict[str, float]
    raw_emotions: Dict[str, str]
    fusion_scores: Dict[str, float]
    timestamp: float

class LearningStateMapper:
    """Maps emotions from different modalities to learning-centered states"""
    
    def __init__(self):
        # Define learning states
        self.learning_states = [
            'engaged',      # Actively participating, focused, understanding
            'confused',     # Having difficulty, need help
            'bored',        # Disengaged, lack of interest
            'frustrated',   # Struggling, stressed about material
            'curious',      # Interested, asking questions, exploring
            'neutral'       # Baseline state, no strong indicators
        ]
        
        # Emotion to learning state mappings for each modality
        self.facial_mapping = {
            'happy': {'engaged': 0.7, 'curious': 0.3},
            'surprise': {'curious': 0.6, 'engaged': 0.4},
            'neutral': {'neutral': 1.0},
            'sad': {'bored': 0.6, 'frustrated': 0.4},
            'angry': {'frustrated': 0.8, 'confused': 0.2},
            'fear': {'confused': 0.7, 'frustrated': 0.3},
            'disgust': {'bored': 0.5, 'frustrated': 0.5}
        }
        
        self.audio_mapping = {
            'happy': {'engaged': 0.8, 'curious': 0.2},
            'sad': {'bored': 0.7, 'frustrated': 0.3},
            'angry': {'frustrated': 0.9, 'confused': 0.1},
            'fear': {'confused': 0.8, 'frustrated': 0.2},
            'neutral': {'neutral': 1.0},
            'disgust': {'bored': 0.6, 'frustrated': 0.4}
        }
        
        self.text_mapping = {
            'engaged': {'engaged': 1.0},
            'confused': {'confused': 1.0},
            'bored': {'bored': 1.0},
            'frustrated': {'frustrated': 1.0},
            'curious': {'curious': 1.0},
            'neutral': {'neutral': 1.0},
            'positive': {'engaged': 0.6, 'curious': 0.4},
            'negative': {'frustrated': 0.5, 'confused': 0.3, 'bored': 0.2}
        }
    
    def map_emotion_to_states(self, emotion: str, modality: str, confidence: float) -> Dict[str, float]:
        """Map an emotion from a specific modality to learning state probabilities"""
        
        if modality == 'facial':
            mapping = self.facial_mapping
        elif modality == 'audio':
            mapping = self.audio_mapping
        elif modality == 'text':
            mapping = self.text_mapping
        else:
            return {'neutral': 1.0}
        
        # Get base mapping
        base_states = mapping.get(emotion, {'neutral': 1.0})
        
        # Scale by confidence
        scaled_states = {}
        for state, prob in base_states.items():
            scaled_states[state] = prob * confidence
        
        # Add neutral baseline for low confidence
        neutral_weight = 1.0 - confidence
        if 'neutral' not in scaled_states:
            scaled_states['neutral'] = 0
        scaled_states['neutral'] += neutral_weight
        
        return scaled_states

class TemporalFusionModel:
    """Temporal fusion model for combining multimodal inputs over time"""
    
    def __init__(self, window_size: int = 10, decay_factor: float = 0.9):
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # Temporal buffers for each modality
        self.facial_history = deque(maxlen=window_size)
        self.audio_history = deque(maxlen=window_size)
        self.text_history = deque(maxlen=window_size)
        
        # State transition probabilities (learned from data or expert knowledge)
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Previous state for temporal consistency
        self.previous_state = 'neutral'
        
    def _initialize_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize state transition probabilities"""
        
        states = ['engaged', 'confused', 'bored', 'frustrated', 'curious', 'neutral']
        
        # Define realistic transition probabilities
        transitions = {
            'engaged': {'engaged': 0.6, 'curious': 0.2, 'neutral': 0.1, 'confused': 0.05, 'bored': 0.03, 'frustrated': 0.02},
            'confused': {'confused': 0.4, 'frustrated': 0.3, 'engaged': 0.15, 'neutral': 0.1, 'bored': 0.03, 'curious': 0.02},
            'bored': {'bored': 0.5, 'neutral': 0.3, 'confused': 0.1, 'frustrated': 0.05, 'engaged': 0.03, 'curious': 0.02},
            'frustrated': {'frustrated': 0.4, 'confused': 0.25, 'bored': 0.15, 'neutral': 0.1, 'engaged': 0.05, 'curious': 0.05},
            'curious': {'curious': 0.5, 'engaged': 0.3, 'neutral': 0.1, 'confused': 0.05, 'bored': 0.03, 'frustrated': 0.02},
            'neutral': {'neutral': 0.4, 'engaged': 0.2, 'curious': 0.15, 'confused': 0.1, 'bored': 0.1, 'frustrated': 0.05}
        }
        
        return transitions
    
    def add_modality_input(self, modality: str, emotion: str, confidence: float, timestamp: float):
        """Add new input from a modality"""
        
        input_data = {
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        if modality == 'facial':
            self.facial_history.append(input_data)
        elif modality == 'audio':
            self.audio_history.append(input_data)
        elif modality == 'text':
            self.text_history.append(input_data)
    
    def get_temporal_weights(self, history: deque, current_time: float) -> List[float]:
        """Calculate temporal weights for historical data"""
        
        if not history:
            return []
        
        weights = []
        for i, data in enumerate(history):
            # Time-based decay
            time_diff = current_time - data['timestamp']
            time_weight = np.exp(-time_diff / 10.0)  # 10 second half-life
            
            # Recency-based weight
            recency_weight = (i + 1) / len(history)
            
            # Confidence-based weight
            confidence_weight = data['confidence']
            
            # Combined weight
            combined_weight = time_weight * recency_weight * confidence_weight
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights

class MultimodalFusionEngine:
    """Main fusion engine that combines all modalities"""
    
    def __init__(self, 
                 fusion_strategy: str = "weighted_voting",
                 modality_weights: Optional[Dict[str, float]] = None,
                 confidence_threshold: float = 0.3):
        
        self.fusion_strategy = fusion_strategy
        self.confidence_threshold = confidence_threshold
        
        # Default modality weights (can be learned from data)
        self.modality_weights = modality_weights or {
            'facial': 0.4,
            'audio': 0.3,
            'text': 0.3
        }
        
        # Initialize components
        self.state_mapper = LearningStateMapper()
        self.temporal_model = TemporalFusionModel()
        
        # Performance tracking
        self.fusion_history = deque(maxlen=100)
        
    def fuse_multimodal_input(self, 
                             facial_input: Optional[ModalityInput] = None,
                             audio_input: Optional[ModalityInput] = None,
                             text_input: Optional[ModalityInput] = None) -> FusionOutput:
        """
        Main fusion function that combines inputs from all modalities
        """
        
        current_time = time.time()
        
        # Collect modality inputs
        modality_inputs = {}
        
        if facial_input and facial_input.confidence >= self.confidence_threshold:
            modality_inputs['facial'] = facial_input
            self.temporal_model.add_modality_input(
                'facial', facial_input.emotion, facial_input.confidence, current_time
            )
        
        if audio_input and audio_input.confidence >= self.confidence_threshold:
            modality_inputs['audio'] = audio_input
            self.temporal_model.add_modality_input(
                'audio', audio_input.emotion, audio_input.confidence, current_time
            )
        
        if text_input and text_input.confidence >= self.confidence_threshold:
            modality_inputs['text'] = text_input
            self.temporal_model.add_modality_input(
                'text', text_input.emotion, text_input.confidence, current_time
            )
        
        # Perform fusion based on strategy
        if self.fusion_strategy == "weighted_voting":
            fusion_result = self._weighted_voting_fusion(modality_inputs, current_time)
        elif self.fusion_strategy == "neural_fusion":
            fusion_result = self._neural_fusion(modality_inputs, current_time)
        elif self.fusion_strategy == "probabilistic_fusion":
            fusion_result = self._probabilistic_fusion(modality_inputs, current_time)
        else:
            fusion_result = self._simple_fusion(modality_inputs, current_time)
        
        # Store fusion history
        self.fusion_history.append(fusion_result)
        
        return fusion_result
    
    def _weighted_voting_fusion(self, modality_inputs: Dict, current_time: float) -> FusionOutput:
        """Weighted voting fusion strategy"""
        
        # Initialize learning state scores
        state_scores = {state: 0.0 for state in self.state_mapper.learning_states}
        modality_contributions = {}
        raw_emotions = {}
        
        total_weight = 0.0
        
        # Process each modality
        for modality, input_data in modality_inputs.items():
            # Map emotion to learning states
            state_probs = self.state_mapper.map_emotion_to_states(
                input_data.emotion, modality, input_data.confidence
            )
            
            # Apply modality weight
            modality_weight = self.modality_weights.get(modality, 1.0)
            weighted_contribution = modality_weight * input_data.confidence
            
            # Add to state scores
            for state, prob in state_probs.items():
                state_scores[state] += prob * weighted_contribution
            
            total_weight += weighted_contribution
            modality_contributions[modality] = weighted_contribution
            raw_emotions[modality] = input_data.emotion
        
        # Normalize scores
        if total_weight > 0:
            for state in state_scores:
                state_scores[state] /= total_weight
            
            # Normalize modality contributions
            for modality in modality_contributions:
                modality_contributions[modality] /= total_weight
        
        # Add temporal consistency
        state_scores = self._apply_temporal_consistency(state_scores)
        
        # Find best state
        best_state = max(state_scores.items(), key=lambda x: x[1])
        final_state = best_state[0]
        confidence = best_state[1]
        
        # Update previous state
        self.temporal_model.previous_state = final_state
        
        return FusionOutput(
            learning_state=final_state,
            confidence=float(confidence),
            modality_contributions=modality_contributions,
            raw_emotions=raw_emotions,
            fusion_scores=dict(state_scores),
            timestamp=current_time
        )
    
    def _apply_temporal_consistency(self, state_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal consistency using transition probabilities"""
        
        previous_state = self.temporal_model.previous_state
        transitions = self.temporal_model.transition_matrix.get(previous_state, {})
        
        # Blend current scores with transition probabilities
        temporal_weight = 0.2  # Weight for temporal consistency
        current_weight = 1.0 - temporal_weight
        
        adjusted_scores = {}
        for state in state_scores:
            current_score = state_scores[state] * current_weight
            transition_score = transitions.get(state, 0.0) * temporal_weight
            adjusted_scores[state] = current_score + transition_score
        
        return adjusted_scores
    
    def _simple_fusion(self, modality_inputs: Dict, current_time: float) -> FusionOutput:
        """Simple majority voting fusion"""
        
        if not modality_inputs:
            return FusionOutput(
                learning_state='neutral',
                confidence=0.5,
                modality_contributions={},
                raw_emotions={},
                fusion_scores={'neutral': 1.0},
                timestamp=current_time
            )
        
        # Simple majority vote
        emotion_votes = {}
        total_confidence = 0
        
        for modality, input_data in modality_inputs.items():
            emotion = input_data.emotion
            confidence = input_data.confidence
            
            if emotion not in emotion_votes:
                emotion_votes[emotion] = 0
            emotion_votes[emotion] += confidence
            total_confidence += confidence
        
        # Find majority emotion
        best_emotion = max(emotion_votes.items(), key=lambda x: x[1])
        
        # Map to learning state (simplified)
        if best_emotion[0] in ['happy', 'engaged']:
            learning_state = 'engaged'
        elif best_emotion[0] in ['confused', 'fear']:
            learning_state = 'confused'
        elif best_emotion[0] in ['sad', 'bored']:
            learning_state = 'bored'
        elif best_emotion[0] in ['angry', 'frustrated']:
            learning_state = 'frustrated'
        else:
            learning_state = 'neutral'
        
        confidence = best_emotion[1] / total_confidence if total_confidence > 0 else 0.5
        
        return FusionOutput(
            learning_state=learning_state,
            confidence=confidence,
            modality_contributions={m: 1.0/len(modality_inputs) for m in modality_inputs},
            raw_emotions={m: inp.emotion for m, inp in modality_inputs.items()},
            fusion_scores={learning_state: confidence},
            timestamp=current_time
        )
    
    def _neural_fusion(self, modality_inputs: Dict, current_time: float) -> FusionOutput:
        """Neural network-based fusion (placeholder for future implementation)"""
        # This would involve a trained neural network to combine features
        # For now, fall back to weighted voting
        return self._weighted_voting_fusion(modality_inputs, current_time)
    
    def _probabilistic_fusion(self, modality_inputs: Dict, current_time: float) -> FusionOutput:
        """Probabilistic fusion using Bayesian inference (placeholder)"""
        # This would use Bayesian methods to combine probability distributions
        # For now, fall back to weighted voting
        return self._weighted_voting_fusion(modality_inputs, current_time)
    
    def get_fusion_statistics(self) -> Dict:
        """Get statistics about fusion performance"""
        
        if not self.fusion_history:
            return {}
        
        # Calculate state distribution
        state_counts = {}
        confidence_scores = []
        
        for result in self.fusion_history:
            state = result.learning_state
            if state not in state_counts:
                state_counts[state] = 0
            state_counts[state] += 1
            confidence_scores.append(result.confidence)
        
        # Calculate percentages
        total_decisions = len(self.fusion_history)
        state_percentages = {
            state: (count / total_decisions) * 100
            for state, count in state_counts.items()
        }
        
        return {
            'total_decisions': total_decisions,
            'state_distribution': state_percentages,
            'average_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores),
            'modality_weights': self.modality_weights,
            'fusion_strategy': self.fusion_strategy
        }
    
    def update_modality_weights(self, new_weights: Dict[str, float]):
        """Update modality weights based on performance feedback"""
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.modality_weights = {
                modality: weight / total_weight
                for modality, weight in new_weights.items()
            }

class MultimodalFusionService:
    """High-level service for multimodal emotion fusion in virtual classroom"""
    
    def __init__(self, fusion_strategy: str = "weighted_voting"):
        self.fusion_engine = MultimodalFusionEngine(fusion_strategy=fusion_strategy)
        self.active_sessions = {}  # Track active fusion sessions
        
    def start_session(self, session_id: str, user_id: str):
        """Start multimodal fusion session"""
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'start_time': time.time(),
            'fusion_count': 0,
            'state_history': []
        }
    
    def end_session(self, session_id: str):
        """End multimodal fusion session"""
        if session_id in self.active_sessions:
            session_data = self.active_sessions.pop(session_id)
            return {
                'session_id': session_id,
                'duration': time.time() - session_data['start_time'],
                'total_fusions': session_data['fusion_count'],
                'session_summary': self._generate_session_summary(session_data['state_history'])
            }
        return None
    
    def process_multimodal_input(self,
                                facial_emotion: Optional[str] = None,
                                facial_confidence: Optional[float] = None,
                                audio_emotion: Optional[str] = None,
                                audio_confidence: Optional[float] = None,
                                text_sentiment: Optional[str] = None,
                                text_confidence: Optional[float] = None,
                                session_id: Optional[str] = None) -> Dict:
        """Process multimodal input and return fusion result"""
        
        # Create modality inputs
        facial_input = None
        if facial_emotion and facial_confidence:
            facial_input = ModalityInput(
                emotion=facial_emotion,
                confidence=facial_confidence,
                timestamp=time.time()
            )
        
        audio_input = None
        if audio_emotion and audio_confidence:
            audio_input = ModalityInput(
                emotion=audio_emotion,
                confidence=audio_confidence,
                timestamp=time.time()
            )
        
        text_input = None
        if text_sentiment and text_confidence:
            text_input = ModalityInput(
                emotion=text_sentiment,
                confidence=text_confidence,
                timestamp=time.time()
            )
        
        # Perform fusion
        fusion_result = self.fusion_engine.fuse_multimodal_input(
            facial_input=facial_input,
            audio_input=audio_input,
            text_input=text_input
        )
        
        # Update session data if session is active
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['fusion_count'] += 1
            session['state_history'].append({
                'learning_state': fusion_result.learning_state,
                'confidence': fusion_result.confidence,
                'timestamp': fusion_result.timestamp,
                'modalities_used': list(fusion_result.raw_emotions.keys())
            })
        
        # Convert to dictionary for API response
        return {
            'learning_state': fusion_result.learning_state,
            'confidence': fusion_result.confidence,
            'modality_contributions': fusion_result.modality_contributions,
            'raw_emotions': fusion_result.raw_emotions,
            'fusion_scores': fusion_result.fusion_scores,
            'timestamp': fusion_result.timestamp,
            'session_id': session_id
        }
    
    def _generate_session_summary(self, state_history: List[Dict]) -> Dict:
        """Generate summary for a fusion session"""
        
        if not state_history:
            return {}
        
        # Calculate state distribution
        state_counts = {}
        confidence_scores = []
        modality_usage = {}
        
        for entry in state_history:
            state = entry['learning_state']
            confidence = entry['confidence']
            modalities = entry['modalities_used']
            
            # Count states
            if state not in state_counts:
                state_counts[state] = 0
            state_counts[state] += 1
            
            confidence_scores.append(confidence)
            
            # Count modality usage
            for modality in modalities:
                if modality not in modality_usage:
                    modality_usage[modality] = 0
                modality_usage[modality] += 1
        
        total_entries = len(state_history)
        
        # Calculate percentages
        state_percentages = {
            state: (count / total_entries) * 100
            for state, count in state_counts.items()
        }
        
        modality_percentages = {
            modality: (count / total_entries) * 100
            for modality, count in modality_usage.items()
        }
        
        # Find dominant state
        dominant_state = max(state_counts.items(), key=lambda x: x[1])
        
        return {
            'total_fusion_decisions': total_entries,
            'state_distribution': state_percentages,
            'dominant_state': dominant_state[0],
            'dominant_state_percentage': (dominant_state[1] / total_entries) * 100,
            'average_confidence': float(np.mean(confidence_scores)),
            'modality_usage': modality_percentages,
            'session_duration': state_history[-1]['timestamp'] - state_history[0]['timestamp']
        }

# Global fusion service instance
multimodal_fusion_service = MultimodalFusionService()