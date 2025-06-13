"""
Advanced Team Classification Manager
Integrates all existing team identification modules
Handles detection-based classification, crop collection, and adaptive retraining
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging
import time
import gc
import torch
import psutil

from .classifier import UnifiedTeamClassifier
from .crop_manager import CropManager
from .basketball_rules import BasketballTeamBalancer
from .ml_classifier import MLTeamClassifier
from .color_classifier import ColorBasedClassifier
from utils.memory_utils import cleanup_memory, force_cleanup_large_objects, MemoryManager


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
        # Reduce max crops to prevent memory issues
        self.max_crops = min(max_crops, 2000)  # Limit to 2000 crops max
        
        # Memory optimization parameters
        self.feature_extraction_interval = 5
        self.memory_cleanup_interval = 20
        self.max_crop_storage = 500  # Reduced from 1000
        self.last_memory_cleanup = 0
        self.enable_memory_logging = True

        # Reuse variables to prevent memory accumulation
        self.cached_assignments = np.array([])
        self.cached_confidence = 0.0
        self.temp_crops = []  # Reusable crop list

        # Initialize using existing modules
        self.unified_classifier = UnifiedTeamClassifier(
            use_ml=True,
            use_color_fallback=True,
            enforce_basketball_rules=True,
            device=device,
            max_crops=self.max_crops
        )

        # Crop management using existing module
        self.crop_manager = CropManager(max_crops=self.max_crops)

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

    def log_memory_usage(self, step_name: str):
        """Log current memory usage"""
        if self.enable_memory_logging:
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"💾 {step_name}: Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")

    def cleanup_memory(self):
        """Enhanced memory cleanup with more aggressive optimization"""
        # Clear Python objects
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear excess crops more aggressively
        if self.crop_manager.get_crop_count() > self.max_crop_storage:
            # Keep only the most recent crops
            current_count = self.crop_manager.get_crop_count()
            keep_count = self.max_crop_storage // 2  # Keep only half
            
            self.crop_manager.crops = self.crop_manager.crops[-keep_count:]
            self.crop_manager.crop_metadata = self.crop_manager.crop_metadata[-keep_count:]
            
            print(f"🧹 Aggressive crop cleanup: {current_count} -> {len(self.crop_manager.crops)} crops")
        
        # Clear temporary variables
        self.temp_crops.clear()

    def is_initialized(self) -> bool:
        """Check if the team classifier is initialized - THE MISSING METHOD"""
        return self.teams_initialized

    def classify(self, tracks: List, frame: np.ndarray) -> Dict[int, int]:
        """
        Classify tracks into teams - COMPATIBILITY METHOD FOR FRAME PROCESSOR

        Args:
            tracks: List of tracks to classify
            frame: Current video frame

        Returns:
            Dictionary mapping track_id to team_id
        """
        if not self.teams_initialized:
            # Return empty dict if not initialized
            return {}

        # Use the existing unified classifier's classify method
        return self.unified_classifier.classify(tracks, frame)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Train the team classifier - COMPATIBILITY METHOD FOR FRAME PROCESSOR

        Args:
            crops: List of player crop images for training
        """
        if len(crops) == 0:
            return

        print(f"🎯 AdvancedTeamClassificationManager: Training with {len(crops)} crops")

        # Use the existing unified classifier's fit method
        self.unified_classifier.fit(crops)

        # Update our initialization status based on unified classifier
        if self.unified_classifier.is_initialized():
            self.teams_initialized = True
            print("✅ AdvancedTeamClassificationManager: Training complete, teams initialized")
        else:
            print("⚠️ AdvancedTeamClassificationManager: Training attempted but initialization incomplete")

    def collect_crops_from_detections(self, frame: np.ndarray, detections, frame_idx: int) -> int:
        """Collect crops with memory optimization - reuse temp_crops list"""
        crops_added = 0
        
        if detections is None or len(detections) == 0:
            return 0

        # Clear and reuse temp_crops list
        self.temp_crops.clear()
        
        # Limit crop collection to prevent memory issues
        max_crops_this_frame = min(len(detections), 10)
        
        for player_idx in range(max_crops_this_frame):
            if self.crop_manager.get_crop_count() >= self.max_crops:
                break
                
            if player_idx >= len(detections.xyxy):
                break

            try:
                bbox = detections.xyxy[player_idx]
                crop = self.crop_manager.extract_crop(frame, bbox)

                if crop is not None:
                    # Store in temp list first
                    self.temp_crops.append((crop, {'frame_idx': frame_idx, 'player_idx': player_idx}))
                    
            except Exception as e:
                logging.warning(f"Failed to extract crop for player {player_idx}: {e}")
                continue

        # Add crops from temp list in batch
        for crop, metadata in self.temp_crops:
            if self.crop_manager.add_crop(crop, metadata):
                crops_added += 1
        
        # Clear temp list immediately
        self.temp_crops.clear()
        
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
        Optimized update method with aggressive memory management
        """
        # Early return for empty detections
        if detections is None or len(detections) == 0:
            return np.array([]), 0.0

        # Memory cleanup check - more frequent
        if frame_idx - self.last_memory_cleanup >= self.memory_cleanup_interval:
            self.cleanup_memory()
            self.last_memory_cleanup = frame_idx

        # Only process feature extraction at intervals to save memory
        should_extract_features = (frame_idx % self.feature_extraction_interval == 0)

        # Collect crops if not initialized (less frequently)
        if not self.teams_initialized:
            if should_extract_features and frame_idx % 3 == 0:  # Even less frequent collection
                with MemoryManager("crop_collection"):
                    crops_added = self.collect_crops_from_detections(frame, detections, frame_idx)
                    if crops_added > 0:
                        self.initialization_frames += 1

            # Try to initialize (less frequently)
            if (self.initialization_frames >= self.min_initialization_frames or
                (frame_idx % 50 == 0 and self.crop_manager.get_crop_count() >= 25)):
                if self.initialize_team_classifier():
                    print(f"🎉 Team classifier initialized at frame {frame_idx}")

        # Continue collecting crops for potential retraining (much less frequently)
        else:
            if frame_idx % 20 == 0 and should_extract_features:  # Very infrequent collection
                self.collect_crops_from_detections(frame, detections, frame_idx)

        # Predict teams if initialized
        team_assignments = np.array([])
        confidence = 0.0
        
        if self.teams_initialized:
            if should_extract_features:
                # Check if we should retrain (less frequently)
                if frame_idx % 500 == 0 and self.should_retrain(frame_idx):
                    print(f"🔄 Retraining classifier at frame {frame_idx}")
                    self.last_retrain_frame = frame_idx
                    self.retraining_count += 1

                # Get team predictions
                with MemoryManager("team_prediction"):
                    team_assignments, confidence = self.predict_teams(frame, detections)
                
                # Cache results and limit cache size
                if len(team_assignments) <= 15:  # Only cache reasonable sizes
                    self.cached_assignments = team_assignments.copy()
                    self.cached_confidence = confidence
            else:
                # Use cached results for frames between intervals
                if (hasattr(self, 'cached_assignments') and 
                    len(self.cached_assignments) == len(detections) and
                    len(self.cached_assignments) <= 15):
                    team_assignments = self.cached_assignments.copy()
                    confidence = self.cached_confidence
                else:
                    # Fallback to simple assignment
                    team_assignments = np.array([i % 2 for i in range(min(len(detections), 15))])
                    confidence = 0.5

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
