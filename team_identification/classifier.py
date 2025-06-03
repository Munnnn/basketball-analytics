"""
Unified team classification interface
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from core import Track, TeamClassifier
from .ml_classifier import MLTeamClassifier
from .color_classifier import ColorBasedClassifier
from .basketball_rules import BasketballTeamBalancer
from .crop_manager import CropManager


class TeamClassifierInterface(TeamClassifier):
    """Base interface for team classifiers"""
    
    def __init__(self):
        self.initialized = False
        
    def is_initialized(self) -> bool:
        """Check if classifier is ready"""
        return self.initialized


class UnifiedTeamClassifier(TeamClassifierInterface):
    """Unified team classifier that combines ML and color-based approaches"""
    
    def __init__(self, 
                 use_ml: bool = True,
                 use_color_fallback: bool = True,
                 enforce_basketball_rules: bool = True,
                 device: str = 'cuda',
                 max_crops: int = 5000):
        """
        Initialize unified team classifier
        
        Args:
            use_ml: Whether to use ML-based classification
            use_color_fallback: Whether to use color-based fallback
            enforce_basketball_rules: Whether to enforce 5v5 balancing
            device: Device for ML inference
            max_crops: Maximum crops to store
        """
        super().__init__()
        
        self.use_ml = use_ml
        self.use_color_fallback = use_color_fallback
        self.enforce_basketball_rules = enforce_basketball_rules
        
        # Initialize components
        self.crop_manager = CropManager(max_crops=max_crops)
        
        if use_ml:
            self.ml_classifier = MLTeamClassifier(device=device)
        else:
            self.ml_classifier = None
            
        if use_color_fallback:
            self.color_classifier = ColorBasedClassifier()
        else:
            self.color_classifier = None
            
        if enforce_basketball_rules:
            self.team_balancer = BasketballTeamBalancer()
        else:
            self.team_balancer = None
            
        # Statistics
        self.classification_stats = {
            'ml_classifications': 0,
            'color_classifications': 0,
            'balance_corrections': 0
        }
        
    def classify(self, tracks: List[Track], frame: np.ndarray) -> Dict[int, int]:
        """
        Classify tracks into teams
        
        Args:
            tracks: List of tracks to classify
            frame: Current video frame
            
        Returns:
            Dictionary mapping track_id to team_id
        """
        if not tracks:
            return {}
            
        # Extract crops for active tracks
        crops = []
        track_ids = []
        
        for track in tracks:
            if track.current_bbox is not None:
                crop = self.crop_manager.extract_crop(frame, track.current_bbox)
                if crop is not None:
                    crops.append(crop)
                    track_ids.append(track.id)
                    
        if not crops:
            return {}
            
        # Try ML classification first
        predictions = None
        confidence = 0.0
        
        if self.ml_classifier and self.ml_classifier.is_initialized():
            try:
                predictions = self.ml_classifier.predict(crops)
                confidence = self.ml_classifier.get_confidence()
                self.classification_stats['ml_classifications'] += len(predictions)
            except Exception as e:
                logging.warning(f"ML classification failed: {e}")
                predictions = None
                
        # Fallback to color-based classification
        if predictions is None and self.color_classifier:
            try:
                predictions = self.color_classifier.predict(crops)
                self.classification_stats['color_classifications'] += len(predictions)
            except Exception as e:
                logging.warning(f"Color classification failed: {e}")
                predictions = np.zeros(len(crops), dtype=int)
                
        if predictions is None:
            # Default assignment
            predictions = np.array([i % 2 for i in range(len(crops))])
            
        # Apply basketball rules for team balancing
        if self.team_balancer and self.enforce_basketball_rules:
            original = predictions.copy()
            predictions = self.team_balancer.balance_teams(predictions, crops)
            
            if not np.array_equal(original, predictions):
                self.classification_stats['balance_corrections'] += 1
                
        # Create result mapping
        result = {}
        for track_id, team_id in zip(track_ids, predictions):
            result[track_id] = int(team_id)
            
        return result
        
    def fit(self, crops: List[np.ndarray]) -> None:
        """Train classifier on player crops"""
        if len(crops) < 10:
            logging.warning(f"Not enough crops for training: {len(crops)}")
            return
            
        # Store crops for later training
        for crop in crops:
            self.crop_manager.add_crop(crop)
            
        # Try to initialize classifiers
        if self.crop_manager.get_crop_count() >= 50:
            if self.ml_classifier and not self.ml_classifier.is_initialized():
                try:
                    training_crops = self.crop_manager.get_training_crops()
                    self.ml_classifier.fit(training_crops)
                    self.initialized = True
                    logging.info("ML classifier initialized successfully")
                except Exception as e:
                    logging.warning(f"ML classifier training failed: {e}")
                    
            if self.color_classifier and not self.color_classifier.is_initialized():
                try:
                    training_crops = self.crop_manager.get_training_crops()
                    self.color_classifier.fit(training_crops)
                    if not self.initialized:
                        self.initialized = True
                    logging.info("Color classifier initialized successfully")
                except Exception as e:
                    logging.warning(f"Color classifier training failed: {e}")
                    
    def update_with_feedback(self, track_id: int, correct_team_id: int):
        """Update classifier with correction feedback"""
        # Store feedback for adaptive learning
        pass
        
    def get_statistics(self) -> Dict[str, int]:
        """Get classification statistics"""
        return self.classification_stats.copy()
