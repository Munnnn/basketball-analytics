"""
Enhanced Possession Tracker with Context Integration
Matches the pasted code's update_possession_with_context functionality
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .tracker import PossessionTracker
from .context import PossessionContext
from ..pose import PoseEstimator, ActionDetector
from core import Track, PossessionInfo


class EnhancedPossessionTracker(PossessionTracker):
    """
    CRITICAL FIX: Enhanced possession tracker with context awareness and pose integration
    This matches the pasted code's EnhancedPossessionTracker functionality
    """
    
    def __init__(self,
                 ball_proximity_threshold: float = 80.0,
                 possession_change_threshold: int = 8,
                 min_possession_duration: int = 5,
                 basketball_enhanced: bool = True,
                 context_tracking: bool = True):
        
        super().__init__(ball_proximity_threshold, possession_change_threshold, min_possession_duration)
        
        self.basketball_enhanced = basketball_enhanced
        self.context_tracking = context_tracking
        
        # Enhanced components
        if context_tracking:
            self.possession_context = PossessionContext(context_window=3)
            self.pose_estimator = PoseEstimator(use_openpose=True)
            self.action_detector = ActionDetector()
        else:
            self.possession_context = None
            self.pose_estimator = None
            self.action_detector = None
            
        # Frame history for movement analysis
        self.frame_history = deque(maxlen=10)
        
        # Enhanced tracking parameters
        self.confidence_threshold = 0.5
        
        print("✅ Enhanced possession tracker with context integration initialized")
    
    def update_possession_with_context(self, frame_data: Dict, frame_idx: int) -> Dict:
        """
        CRITICAL FIX: Enhanced possession tracking with context and pose analysis
        This exactly matches the pasted code's method signature and functionality
        """
        
        # Store frame data for movement analysis
        self.frame_history.append({
            'frame_idx': frame_idx,
            'player_positions': self._extract_player_positions(frame_data),
            'ball_position': self._extract_ball_position(frame_data),
            'timestamp': time.time()
        })
        
        player_detections = frame_data.get('player_detections')
        ball_detections = frame_data.get('ball_detections')
        player_team_mapping = frame_data.get('player_team_mapping', {})
        
        if player_detections is None or ball_detections is None:
            return self._no_possession_update_with_context(frame_idx)
        
        # Extract poses from player crops if available
        poses = []
        if self.pose_estimator and len(player_detections) > 0:
            try:
                crops = self._extract_player_crops(frame_data.get('frame'), player_detections)
                poses = self.pose_estimator.extract_poses_from_crops(crops)
            except Exception as e:
                print(f"Pose extraction failed: {e}")
                poses = [None] * len(player_detections)
        
        # Enhanced possession detection with pose data
        possession_result = self._get_player_in_possession_enhanced(
            player_detections, ball_detections, self.ball_proximity_threshold,
            player_team_mapping, poses
        )
        
        if possession_result is None:
            return self._no_possession_update_with_context(frame_idx)
        
        player_in_possession = possession_result['player_id']
        team_in_possession = possession_result['team_id']
        confidence = possession_result['confidence']
        
        print(f"   🎯 Frame {frame_idx}: Player {player_in_possession} (Team {team_in_possession}) in possession (conf: {confidence:.2f})")
        
        # Enhanced possession change logic with context
        if self._should_change_possession_with_context(
            player_in_possession, team_in_possession, confidence, frame_idx):
            
            self._end_current_possession_with_context(frame_idx, poses)
            self._start_new_possession_with_context(
                player_in_possession, team_in_possession, frame_idx, frame_data, poses
            )
            print(f"   ✨ NEW POSSESSION: Player {player_in_possession} (Team {team_in_possession})")
        else:
            self._update_current_possession_with_context(frame_data, frame_idx, poses)
        
        return self._get_possession_summary_with_context(frame_idx)
    
    def _extract_player_crops(self, frame: Optional[np.ndarray], player_detections) -> List[np.ndarray]:
        """Extract player crops for pose estimation"""
        if frame is None or len(player_detections) == 0:
            return []
        
        crops = []
        for bbox in player_detections.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
        
        return crops
    
    def _extract_player_positions(self, frame_
