"""
Single frame processing pipeline
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from core import Detection, Track, PossessionInfo, PlayEvent
from core.constants import PLAYER_ID, BALL_ID, REF_ID, HOOP_ID, BACKBOARD_ID


class FrameProcessor:
    """Process individual frames with all analytics"""
    
    def __init__(self, 
                 detector,
                 mask_generator,
                 tracker,
                 team_classifier,
                 possession_tracker,
                 event_detector,
                 pose_estimator,
                 frame_annotator):
        """
        Initialize frame processor with components
        
        Args:
            detector: Object detector
            mask_generator: Mask generator
            tracker: Object tracker
            team_classifier: Team classifier
            possession_tracker: Possession tracker
            event_detector: Event detector
            pose_estimator: Pose estimator
            frame_annotator: Frame annotator
        """
        self.detector = detector
        self.mask_generator = mask_generator
        self.tracker = tracker
        self.team_classifier = team_classifier
        self.possession_tracker = possession_tracker
        self.event_detector = event_detector
        self.pose_estimator = pose_estimator
        self.frame_annotator = frame_annotator
        
        self.logger = logging.getLogger(__name__)
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame through entire pipeline
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            
        Returns:
            Tuple of (annotated_frame, analytics_data)
        """
        # Initialize results
        analytics_data = {
            'frame_idx': frame_idx,
            'detections': {},
            'tracks': [],
            'possession': None,
            'events': [],
            'actions': None
        }
        
        try:
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Generate masks
            if detections:
                masks = self.mask_generator.generate_masks(frame, detections)
                for det, mask in zip(detections, masks):
                    det.mask = mask
                    
            # Separate detections by class
            detections_by_class = self._separate_detections(detections)
            analytics_data['detections'] = detections_by_class
            
            # Update tracking for players
            player_detections = detections_by_class.get('players', [])
            if player_detections:
                tracks = self.tracker.update(player_detections, frame)
                analytics_data['tracks'] = tracks
                
                # Team classification
                if self.team_classifier and self.team_classifier.is_initialized():
                    team_assignments = self.team_classifier.classify(tracks, frame)
                    for track in tracks:
                        track.team_id = team_assignments.get(track.id)
                        
            # Possession tracking
            if self.possession_tracker:
                ball_track = self._get_ball_track(detections_by_class.get('ball', []))
                possession_info = self.possession_tracker.update_possession(
                    analytics_data['tracks'], ball_track, frame_idx
                )
                analytics_data['possession'] = possession_info
                
            # Event detection
            if self.event_detector and analytics_data['possession']:
                events = self.event_detector.detect_events(
                    detections_by_class,
                    analytics_data['possession'],
                    frame_idx
                )
                analytics_data['events'] = events
                
            # Pose estimation
            if self.pose_estimator and analytics_data['tracks']:
                crops = self._extract_crops(frame, analytics_data['tracks'])
                poses = self.pose_estimator.extract_poses(crops)
                analytics_data['poses'] = poses
                
            # Annotate frame
            annotated_frame = frame
            if self.frame_annotator:
                annotated_frame = self.frame_annotator.annotate(
                    frame,
                    detections,
                    analytics_data['tracks'],
                    analytics_data
                )
                
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            annotated_frame = frame
            
        return annotated_frame, analytics_data
        
    def _separate_detections(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Separate detections by class"""
        separated = {
            'players': [],
            'ball': [],
            'referees': [],
            'hoop': [],
            'backboard': [],
            'other': []
        }
        
        for det in detections:
            if det.class_id == PLAYER_ID:
                separated['players'].append(det)
            elif det.class_id == BALL_ID:
                separated['ball'].append(det)
            elif det.class_id == REF_ID:
                separated['referees'].append(det)
            elif det.class_id == HOOP_ID:
                separated['hoop'].append(det)
            elif det.class_id == BACKBOARD_ID:
                separated['backboard'].append(det)
            else:
                separated['other'].append(det)
                
        return separated
        
    def _get_ball_track(self, ball_detections: List[Detection]) -> Optional[Track]:
        """Create simple ball track from detections"""
        if not ball_detections:
            return None
            
        # Use first ball detection
        return Track(
            id=-1,  # Special ID for ball
            detections=[ball_detections[0]],
            state='tracked'
        )
        
    def _extract_crops(self, frame: np.ndarray, tracks: List[Track]) -> List[np.ndarray]:
        """Extract crops for tracks"""
        crops = []
        
        for track in tracks:
            if track.current_bbox is not None:
                x1, y1, x2, y2 = track.current_bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crops.append(frame[y1:y2, x1:x2])
                else:
                    crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
            else:
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
                
        return crops
