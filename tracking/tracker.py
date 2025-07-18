"""
Main tracking orchestrator
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict

from core import Detection, Track, TrackState, Tracker
from .algorithms import TrackAssociator, KalmanTracker
from .features import AppearanceExtractor
from .memory import MemoryOptimizer

from team_identification.advanced_manager import AdvancedTeamClassificationManager
from analytics.possession.enhanced_tracker import EnhancedPossessionTracker
from analytics.possession.context import PossessionContext

class EnhancedTracker(Tracker):
    """Enhanced multi-object tracker for basketball analytics with FIXED integration"""
    
    def __init__(self, 
                 track_activation_threshold: float = 0.4,
                 lost_track_buffer: int = 90,
                 minimum_matching_threshold: float = 0.2,
                 minimum_consecutive_frames: int = 3,
                 max_tracks: int = 15,
                 use_kalman: bool = True,
                 use_appearance: bool = False,
                 use_basketball_logic: bool = True,
                 enable_team_identification: bool = True,
                 enable_enhanced_possession: bool = True):
        """
        Initialize enhanced tracker
        
        Args:
            track_activation_threshold: Minimum confidence to activate track
            lost_track_buffer: Frames to keep lost track before removal
            minimum_matching_threshold: Minimum IoU for matching
            minimum_consecutive_frames: Minimum detections to confirm track
            max_tracks: Maximum number of simultaneous tracks
            use_kalman: Whether to use Kalman filtering
            use_appearance: Whether to use appearance features
        """
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.max_tracks = max_tracks
        self.use_kalman = use_kalman
        self.use_appearance = use_appearance

        # Basketball enhancements
        self.use_basketball_logic = use_basketball_logic
        self.enable_team_identification = enable_team_identification
                     
        # Tracking state
        self.frame_id = 0
        self.next_track_id = 1
        self.active_tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.removed_tracks: Dict[int, Track] = {}
        
        # Core components
        self.associator = TrackAssociator(
            minimum_matching_threshold=minimum_matching_threshold
        )
        self.kalman_filters: Dict[int, KalmanTracker] = {}
        
        if use_appearance:
            self.appearance_extractor = AppearanceExtractor()
        else:
            self.appearance_extractor = None

        # Add advanced team classification manager integration
        if enable_team_identification:
            self.team_classification_manager = AdvancedTeamClassificationManager(
                enhanced_tracker=self,
                device='cuda',
                max_crops=5000
            )
        else:
            self.team_classification_manager = None
            
        # Add enhanced possession tracking integration  
        if enable_enhanced_possession:
            self.enhanced_possession_tracker = EnhancedPossessionTracker(
                basketball_enhanced=True,
                context_tracking=True
            )
            self.possession_context = PossessionContext(context_window=3)
        else:
            self.enhanced_possession_tracker = None
            self.possession_context = None

        # Memory optimizer
        self.memory_optimizer = MemoryOptimizer()

    def update_basketball(self, detections: List[Detection], frame: Optional[np.ndarray] = None, frame_idx: int = 0) -> List[Track]:
        """
        CRITICAL FIX: Basketball-specific update method that works with detections
        This matches the pasted code's integration pattern exactly
        """
        self.frame_id = frame_idx
        
        # Filter low confidence detections
        detections = [d for d in detections if d.confidence >= self.track_activation_threshold]
        
        if not detections:
            self._age_tracks()
            return list(self.active_tracks.values())
        
        # CRITICAL FIX: Team identification using detection-based approach
        team_assignments = np.array([])
        team_confidence = 0.0
        
        if self.team_classification_manager and frame is not None:
            # Convert detections to supervision format for compatibility
            detection_data = self._convert_detections_to_sv_format(detections)
            
            # Get team assignments using detection-based approach
            team_assignments, team_confidence = self.team_classification_manager.update_with_frame(
                frame, detection_data, frame_idx
            )
            
            if len(team_assignments) > 0:
                print(f"🎯 Team assignments: {len(team_assignments)} players classified, confidence: {team_confidence:.2f}")
        
        # Extract appearance features if enabled
        if self.use_appearance and frame is not None:
            features = self.appearance_extractor.extract_batch(frame, np.array([d.bbox for d in detections]))
        else:
            features = None
        
        # Associate detections with existing tracks
        matches, unmatched_detections, unmatched_tracks = self.associator.associate(
            detections, 
            list(self.active_tracks.values()) + list(self.lost_tracks.values()),
            features
        )
        
        # Update matched tracks with team information
        for track_id, det_idx in matches:
            track = self._get_track(track_id)
            if track:
                self._update_track_with_team_info(track, detections[det_idx], team_assignments, det_idx, team_confidence)
                
                # Move from lost to active if needed
                if track_id in self.lost_tracks:
                    self.active_tracks[track_id] = self.lost_tracks.pop(track_id)
        
        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            if track_id in self.active_tracks:
                track = self.active_tracks[track_id]
                track.state = TrackState.LOST
                self.lost_tracks[track_id] = self.active_tracks.pop(track_id)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            if len(self.active_tracks) + len(self.lost_tracks) < self.max_tracks:
                new_track = self._create_track_with_team_info(detections[det_idx], team_assignments, det_idx, team_confidence)
        
        # Clean up old lost tracks
        self._cleanup_lost_tracks()
        
        # Periodic memory cleanup
        if self.frame_id % 50 == 0:
            self.memory_optimizer.cleanup()
        
        return self.get_active_tracks()
    
    def _convert_detections_to_sv_format(self, detections: List[Detection]):
        """Convert detection format for team classification compatibility"""
        import supervision as sv
        
        if not detections:
            return sv.Detections.empty()
        
        xyxy = np.array([d.bbox for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_id = np.array([d.class_id for d in detections])
        
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
    
    def _update_track_with_team_info(self, track: Track, detection: Detection, 
                                   team_assignments: np.ndarray, det_idx: int, team_confidence: float):
        """Update track with team information from detection-based classification"""
        # Standard track update
        self._update_track(track, detection)
        
        # Add team information if available
        if len(team_assignments) > det_idx:
            predicted_team = int(team_assignments[det_idx])
            
            if not hasattr(track, 'team_id') or track.team_id is None:
                track.team_id = predicted_team
                track.team_confidence = team_confidence
            else:
                # Smooth team transitions
                if predicted_team == track.team_id:
                    # Same team - smooth confidence
                    alpha = 0.3
                    track.team_confidence = alpha * team_confidence + (1 - alpha) * track.team_confidence
                else:
                    # Different team - only switch if highly confident
                    if team_confidence > 0.8 and track.team_confidence < 0.6:
                        track.team_id = predicted_team
                        track.team_confidence = team_confidence
                    else:
                        track.team_confidence *= 0.95
    
    def _create_track_with_team_info(self, detection: Detection, team_assignments: np.ndarray, 
                                   det_idx: int, team_confidence: float) -> Track:
        """Create new track with team information"""
        track = Track(
            id=self.next_track_id,
            detections=[detection],
            state=TrackState.TENTATIVE,
            confidence=detection.confidence,
            start_frame=self.frame_id,
            last_seen_frame=self.frame_id
        )
        
        # Add team information if available
        if len(team_assignments) > det_idx:
            track.team_id = int(team_assignments[det_idx])
            track.team_confidence = team_confidence
        
        self.active_tracks[self.next_track_id] = track
        
        # Initialize Kalman filter if enabled
        if self.use_kalman:
            self.kalman_filters[self.next_track_id] = KalmanTracker(detection.bbox)
        
        self.next_track_id += 1
        return track
    
    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """Update tracks with new detections"""
        self.frame_id += 1
        
        # Filter low confidence detections
        detections = [d for d in detections if d.confidence >= self.track_activation_threshold]
        
        if not detections:
            self._age_tracks()
            return list(self.active_tracks.values())
            
        # Extract appearance features if enabled
        if self.use_appearance and frame is not None:
            features = self.appearance_extractor.extract_batch(frame, detections)
        else:
            features = None
            
        # Associate detections with existing tracks
        matches, unmatched_detections, unmatched_tracks = self.associator.associate(
            detections, 
            list(self.active_tracks.values()) + list(self.lost_tracks.values()),
            features
        )
        
        # Update matched tracks
        for track_id, det_idx in matches:
            track = self._get_track(track_id)
            if track:
                self._update_track(track, detections[det_idx])
                
                # Move from lost to active if needed
                if track_id in self.lost_tracks:
                    self.active_tracks[track_id] = self.lost_tracks.pop(track_id)
                    
        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            if track_id in self.active_tracks:
                track = self.active_tracks[track_id]
                track.state = TrackState.LOST
                self.lost_tracks[track_id] = self.active_tracks.pop(track_id)
                
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            if len(self.active_tracks) + len(self.lost_tracks) < self.max_tracks:
                self._create_track(detections[det_idx])
                
        # Clean up old lost tracks
        self._cleanup_lost_tracks()
        
        # Periodic memory cleanup
        if self.frame_id % 50 == 0:
            self.memory_optimizer.cleanup()
            
        return self.get_active_tracks()
        
    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks"""
        # Only return confirmed tracks
        confirmed_tracks = []
        for track in self.active_tracks.values():
            if (track.state == TrackState.TRACKED and 
                len(track.detections) >= self.minimum_consecutive_frames):
                confirmed_tracks.append(track)
        return confirmed_tracks
        
    def reset(self):
        """Reset tracker state"""
        self.frame_id = 0
        self.next_track_id = 1
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.kalman_filters.clear()
        
    def _get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID from any state"""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        elif track_id in self.lost_tracks:
            return self.lost_tracks[track_id]
        return None
        
    def _create_track(self, detection: Detection):
        """Create new track from detection"""
        track = Track(
            id=self.next_track_id,
            detections=[detection],
            state=TrackState.TENTATIVE,
            confidence=detection.confidence,
            start_frame=self.frame_id,
            last_seen_frame=self.frame_id
        )
        
        self.active_tracks[self.next_track_id] = track
        
        # Initialize Kalman filter if enabled
        if self.use_kalman:
            self.kalman_filters[self.next_track_id] = KalmanTracker(detection.bbox)
            
        self.next_track_id += 1
        
    def _update_track(self, track: Track, detection: Detection):
        """Update existing track with new detection"""
        track.detections.append(detection)
        track.last_seen_frame = self.frame_id
        track.confidence = 0.7 * track.confidence + 0.3 * detection.confidence
        
        # Update state
        if track.state == TrackState.TENTATIVE:
            if len(track.detections) >= self.minimum_consecutive_frames:
                track.state = TrackState.TRACKED
        else:
            track.state = TrackState.TRACKED
            
        # Update Kalman filter
        if self.use_kalman and track.id in self.kalman_filters:
            self.kalman_filters[track.id].update(detection.bbox)
            
    def _age_tracks(self):
        """Age tracks when no detections available"""
        tracks_to_move = []
        
        for track_id, track in self.active_tracks.items():
            # Apply confidence decay
            time_lost = self.frame_id - track.last_seen_frame
            if time_lost > 0:
                decay_factor = np.exp(-0.1 * time_lost)
                track.confidence *= decay_factor
                
            # Move to lost if not seen recently
            if time_lost > 0:
                track.state = TrackState.LOST
                tracks_to_move.append(track_id)
                
        # Move tracks to lost state
        for track_id in tracks_to_move:
            self.lost_tracks[track_id] = self.active_tracks.pop(track_id)
            
    def _cleanup_lost_tracks(self):
        """Remove old lost tracks"""
        tracks_to_remove = []
        
        for track_id, track in self.lost_tracks.items():
            if self.frame_id - track.last_seen_frame > self.lost_track_buffer:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            track = self.lost_tracks.pop(track_id)
            track.state = TrackState.REMOVED
            self.removed_tracks[track_id] = track
            
            # Clean up Kalman filter
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]
