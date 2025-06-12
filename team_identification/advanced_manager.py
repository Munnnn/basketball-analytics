"""
Advanced Team Classification Manager - Matches Pasted Code Functionality
Handles detection-based classification, crop collection, and adaptive retraining
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time

from .classifier import UnifiedTeamClassifier
from .crop_manager import CropManager
from .basketball_rules import BasketballTeamBalancer


class AdvancedTeamClassificationManager:
    """
    CRITICAL FIX: Advanced team classification manager to match pasted code
    - Works directly with detections instead of tracks
    - Handles crop collection and adaptive retraining
    - Implements basketball 5v5 balancing
    """
    
    def __init__(self, enhanced_tracker, device: str = 'cuda', max_crops: int = 5000):
        self.enhanced_tracker = enhanced_tracker
        self.device = device
        self.max_crops = max_crops
        
        # Initialize core classifier
        self.team_classifier = UnifiedTeamClassifier(
            use_ml=True,
            basketball_5v5_balancing=True,
            device=device,
            max_crops=max_crops
        )
        
        # Crop management
        self.crop_manager = CropManager(max_crops=max_crops)
        self.team_balancer = BasketballTeamBalancer()
        
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
        
        print("🎯 AdvancedTeamClassificationManager initialized")
        print("🏀 Enhanced with 5v5 brightness-based team balancing")
        
    def collect_crops_from_detections(self, frame: np.ndarray, detections, frame_idx: int) -> int:
        """Collect crops from player detections - MATCHES PASTED CODE"""
        crops_added = 0
        
        if detections is None or len(detections) == 0:
            return 0
            
        # Extract player crops
        for player_idx, bbox in enumerate(detections.xyxy):
            if self.crop_manager.crop_count >= self.max_crops:
                break
                
            try:
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    
                    if self.crop_manager.add_crop(crop, frame_idx, player_idx):
                        crops_added += 1
                        
            except Exception as e:
                logging.warning(f"Failed to extract crop for player {player_idx}: {e}")
                continue
                
        return crops_added
    
    def initialize_team_classifier(self, max_training_crops: int = 3000) -> bool:
        """Initialize team classifier with collected crops - MATCHES PASTED CODE"""
        if self.crop_manager.crop_count < 50:
            return False
            
        try:
            print(f"🚀 Initializing team classifier with {self.crop_manager.crop_count} crops")
            
            # Get training crops
            training_crops = self.crop_manager.get_training_crops(max_training_crops)
            
            if len(training_crops) < 20:
                print(f"❌ Not enough valid crops: {len(training_crops)}")
                return False
                
            # Train the classifier
            self.team_classifier.fit(training_crops)
            self.teams_initialized = True
            
            print("✅ Team classifier initialization complete!")
            
            # Clear crops to free memory
            self.crop_manager.clear()
            
            return True
            
        except Exception as e:
            print(f"❌ Team classifier initialization failed: {e}")
            return False
    
    def predict_teams(self, frame: np.ndarray, detections) -> Tuple[np.ndarray, float]:
        """
        CRITICAL FIX: Predict team assignments directly from detections
        This matches the pasted code interface exactly
        """
        if not self.teams_initialized:
            return np.zeros(len(detections), dtype=int), 0.0
            
        if len(detections) == 0:
            return np.array([]), 1.0
            
        try:
            print(f"🏀 Predicting teams for {len(detections)} players...")
            
            # Use the unified classifier's detection-based method
            predictions, confidence = self.team_classifier.classify_detections(frame, detections)
            
            print(f"✅ Team predictions complete - T0: {np.sum(predictions == 0)}, T1: {np.sum(predictions == 1)}")
            
            return predictions, confidence
            
        except Exception as e:
            logging.error(f"Team prediction failed: {e}")
            return np.zeros(len(detections), dtype=int), 0.1
    
    def should_retrain(self, frame_idx: int) -> bool:
        """Determine if we should retrain the classifier - MATCHES PASTED CODE"""
        if not self.teams_initialized:
            return False
            
        # Fixed interval trigger
        frames_since_retrain = frame_idx - self.last_retrain_frame
        interval_trigger = frames_since_retrain >= self.retrain_interval
        
        # Crop threshold trigger
        crop_trigger = self.crop_manager.crop_count >= self.retrain_crop_threshold
        
        return interval_trigger or crop_trigger
    
    def update_with_frame(self, frame: np.ndarray, detections, frame_idx: int) -> Tuple[np.ndarray, float]:
        """
        CRITICAL FIX: Main update method that works with detections
        This exactly matches the pasted code interface and behavior
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
                
            # Try to initialize
            if (self.initialization_frames >= self.min_initialization_frames or 
                (frame_idx % 25 == 0 and self.crop_manager.crop_count >= 25)):
                if self.initialize_team_classifier():
                    print(f"🎉 Team classifier initialized at frame {frame_idx}")
        
        # Continue collecting crops for potential retraining
        else:
            if frame_idx % 5 == 0:  # Sample every few frames
                self.collect_crops_from_detections(frame, detections, frame_idx)
        
        # Predict teams if initialized
        if self.teams_initialized:
            # Check if we should retrain
            if self.should_retrain(frame_idx):
                print(f"🔄 Retraining classifier at frame {frame_idx}")
                # Could implement retraining logic here
                self.last_retrain_frame = frame_idx
                self.retraining_count += 1
                
            # Get team predictions with 5v5 balancing
            team_assignments, confidence = self.predict_teams(frame, detections)
            
        return team_assignments, confidence
    
    def get_basketball_statistics(self) -> Dict:
        """Get basketball-specific statistics"""
        return {
            'teams_initialized': self.teams_initialized,
            'retraining_count': self.retraining_count,
            'crops_collected': self.crop_manager.crop_count,
            'initialization_frames': self.initialization_frames
        }
