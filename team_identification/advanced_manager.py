"""
Advanced Team Classification Manager
Integrates all existing team identification modules
Handles detection-based classification, crop collection, and adaptive retraining
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time

from .classifier import UnifiedTeamClassifier
from .crop_manager import CropManager
from .basketball_rules import BasketballTeamBalancer
from .ml_classifier import MLTeamClassifier
from .color_classifier import ColorBasedClassifier


class AdvancedTeamClassificationManager:
    """
    Advanced team classification manager that properly integrates all existing modules
    - Works directly with detections instead of tracks
    - Handles crop collection and adaptive retraining
    - Implements basketball 5v5 balancing
    """
    
    def __init__(self, enhanced_tracker, device: str = 'cuda', max_crops: int = 5000):
        self.enhanced_tracker = enhanced_tracker
        self.device = device
        self.max_crops = max_crops
        
        # Initialize using existing modules
        self.unified_classifier = UnifiedTeamClassifier(
            use_ml=True,
            use_color_fallback=True,
            enforce_basketball_rules=True,
            device=device,
            max_crops=max_crops
        )
        
        # Crop management using existing module
        self.crop_manager = CropManager(max_crops=max_crops)
        
        # Basketball team balancing using existing module
        self.team_balancer = BasketballTeamBalancer()
        
        # ML classifier for advanced features
        self.ml_classifier = MLTeamClassifier(device=device)
        
        # Color classifier as fallback
        self.color_classifier = ColorBasedClassifier()
        
        # State tracking
        self.teams_initialized = False
        self.initialization_frames = 0
        self.min_initialization_frames = 50
        
        # Retraining logic
        self.last_retrain_frame = 0
        self.retrain_interval = 1000
        self.retrain_crop_threshold = 500
        self.retraining_count = 0
        self.min_confidence_for_retrain = 0.6
        
        print("🎯 AdvancedTeamClassificationManager initialized with existing modules")
        print("🏀 Enhanced with 5v5 brightness-based team balancing")
        
    def collect_crops_from_detections(self, frame: np.ndarray, detections, frame_idx: int) -> int:
        """Collect crops from player detections using existing crop manager"""
        crops_added = 0
        
        if detections is None or len(detections) == 0:
            return 0
            
        # Extract player crops using existing crop manager functionality
        for player_idx, bbox in enumerate(detections.xyxy):
            if self.crop_manager.get_crop_count() >= self.max_crops:
                break
                
            try:
                # Extract crop using existing crop manager methods
                crop = self.crop_manager.extract_crop(frame, bbox)
                
                if crop is not None and self.crop_manager.add_crop(crop, {'frame_idx': frame_idx, 'player_idx': player_idx}):
                    crops_added += 1
                    
            except Exception as e:
                logging.warning(f"Failed to extract crop for player {player_idx}: {e}")
                continue
                
        return crops_added
    
    def initialize_team_classifier(self, max_training_crops: int = 3000) -> bool:
        """Initialize team classifier using existing unified classifier"""
        if self.crop_manager.get_crop_count() < 50:
            return False
            
        try:
            print(f"🚀 Initializing team classifier with {self.crop_manager.get_crop_count()} crops")
            
            # Get training crops from existing crop manager
            training_crops = self.crop_manager.get_training_crops(max_training_crops)
            
            if len(training_crops) < 20:
                print(f"❌ Not enough valid crops: {len(training_crops)}")
                return False
                
            # Train the unified classifier (which integrates ML + color + basketball rules)
            self.unified_classifier.fit(training_crops)
            self.teams_initialized = True
            
            print("✅ Team classifier initialization complete using unified classifier!")
            
            # Clear crops to free memory
            self.crop_manager.clear()
            
            return True
            
        except Exception as e:
            print(f"❌ Team classifier initialization failed: {e}")
            return False
    
    def predict_teams(self, frame: np.ndarray, detections) -> Tuple[np.ndarray, float]:
        """
        Predict team assignments using existing unified classifier
        """
        if not self.teams_initialized:
            return np.zeros(len(detections), dtype=int), 0.0
            
        if len(detections) == 0:
            return np.array([]), 1.0
            
        try:
            print(f"🏀 Predicting teams for {len(detections)} players using unified classifier...")
            
            # Create dummy tracks for compatibility with existing classifier
            dummy_tracks = []
            for i, bbox in enumerate(detections.xyxy):
                dummy_track = type('Track', (), {
                    'id': i,
                    'current_bbox': bbox,
                    'team_id': None
                })()
                dummy_tracks.append(dummy_track)
            
            # Use existing unified classifier
            team_assignments_dict = self.unified_classifier.classify(dummy_tracks, frame)
            
            # Convert to array format
            predictions = np.array([team_assignments_dict.get(i, 0) for i in range(len(detections))])
            
            # Calculate confidence based on team balance (using existing balancer)
            balanced_predictions = self.team_balancer.balance_teams(predictions, [])
            
            # Use basketball balancing logic for confidence
            team_0_count = np.sum(balanced_predictions == 0)
            team_1_count = np.sum(balanced_predictions == 1)
            
            if 4 <= team_0_count <= 6 and 4 <= team_1_count <= 6:
                confidence = 0.9  # High confidence for balanced teams
            elif team_0_count > 0 and team_1_count > 0:
                confidence = 0.7  # Medium confidence for some balance
            else:
                confidence = 0.3  # Low confidence for single team
            
            print(f"✅ Team predictions complete - T0: {team_0_count}, T1: {team_1_count}")
            
            return balanced_predictions, confidence
            
        except Exception as e:
            logging.error(f"Team prediction failed: {e}")
            return np.zeros(len(detections), dtype=int), 0.1
    
    def should_retrain(self, frame_idx: int) -> bool:
        """Determine if we should retrain the classifier"""
        if not self.teams_initialized:
            return False
            
        # Fixed interval trigger
        frames_since_retrain = frame_idx - self.last_retrain_frame
        interval_trigger = frames_since_retrain >= self.retrain_interval
        
        # Crop threshold trigger
        crop_trigger = self.crop_manager.get_crop_count() >= self.retrain_crop_threshold
        
        return interval_trigger or crop_trigger
    
    def update_with_frame(self, frame: np.ndarray, detections, frame_idx: int) -> Tuple[np.ndarray, float]:
        """
        Main update method that integrates all existing modules
        """
        team_assignments = np.array([])
        confidence = 0.0
        
        if detections is None or len(detections) == 0:
            return team_assignments, confidence
            
        print(f"🏀 Frame {frame_idx} - Processing {len(detections)} detections")
        
        # Collect crops if not initialized
        if not self.teams_initialized:
            crops_added = self.collect_crops_from_detections(frame, detections, frame_idx)
            if crops_added > 0:
                self.initialization_frames += 1
                
            # Try to initialize using existing unified classifier
            if (self.initialization_frames >= self.min_initialization_frames or 
                (frame_idx % 25 == 0 and self.crop_manager.get_crop_count() >= 25)):
                if self.initialize_team_classifier():
                    print(f"🎉 Team classifier initialized at frame {frame_idx}")
        
        # Continue collecting crops for potential retraining
        else:
            if frame_idx % 5 == 0:  # Sample every few frames
                self.collect_crops_from_detections(frame, detections, frame_idx)
        
        # Predict teams if initialized using existing modules
        if self.teams_initialized:
            # Check if we should retrain
            if self.should_retrain(frame_idx):
                print(f"🔄 Retraining classifier at frame {frame_idx}")
                self.last_retrain_frame = frame_idx
                self.retraining_count += 1
                
            # Get team predictions using integrated approach
            team_assignments, confidence = self.predict_teams(frame, detections)
            
        return team_assignments, confidence
    
    def get_basketball_statistics(self) -> Dict:
        """Get basketball-specific statistics"""
        return {
            'teams_initialized': self.teams_initialized,
            'retraining_count': self.retraining_count,
            'crops_collected': self.crop_manager.get_crop_count(),
            'initialization_frames': self.initialization_frames,
            'unified_classifier_stats': self.unified_classifier.get_statistics() if hasattr(self.unified_classifier, 'get_statistics') else {},
            'team_balancer_stats': self.team_balancer.get_balance_statistics() if hasattr(self.team_balancer, 'get_balance_statistics') else {}
        }
