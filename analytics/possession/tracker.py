"""
Basketball possession tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

from core import Track, Detection, PossessionInfo, BALL_ID, PLAYER_ID
from core.constants import (
    BALL_PROXIMITY_THRESHOLD, POSSESSION_CHANGE_THRESHOLD, 
    MIN_POSSESSION_DURATION
)


class PossessionTracker:
    """Track ball possession during basketball game"""
    
    def __init__(self,
                 ball_proximity_threshold: float = BALL_PROXIMITY_THRESHOLD,
                 possession_change_threshold: int = POSSESSION_CHANGE_THRESHOLD,
                 min_possession_duration: int = MIN_POSSESSION_DURATION):
        """
        Initialize possession tracker
        
        Args:
            ball_proximity_threshold: Max distance for possession
            possession_change_threshold: Frames needed to confirm change
            min_possession_duration: Minimum frames for valid possession
        """
        self.ball_proximity_threshold = ball_proximity_threshold
        self.possession_change_threshold = possession_change_threshold
        self.min_possession_duration = min_possession_duration
        
        # Current possession state
        self.current_possession = None
        self.possession_history = []
        
        # Possession change tracking
        self.possession_candidate = None
        self.candidate_frames = 0
        self.candidate_confidence = 0.0
        
        # Frame history for analysis
        self.frame_history = deque(maxlen=10)
        
    def update_possession(self,
                         player_tracks: List[Track],
                         ball_track: Optional[Track],
                         frame_idx: int) -> PossessionInfo:
        """
        Update possession for current frame
        
        Args:
            player_tracks: List of player tracks
            ball_track: Ball track (if detected)
            frame_idx: Current frame index
            
        Returns:
            Current possession information
        """
        # Store frame data
        self.frame_history.append({
            'frame_idx': frame_idx,
            'player_count': len(player_tracks),
            'ball_detected': ball_track is not None
        })
        
        # No ball detected
        if ball_track is None or not ball_track.current_position:
            return self._no_possession_update(frame_idx)
            
        # Find player in possession
        possession_result = self._find_player_in_possession(
            player_tracks, ball_track
        )
        
        if possession_result is None:
            return self._no_possession_update(frame_idx)
            
        player_id = possession_result['player_id']
        team_id = possession_result['team_id']
        confidence = possession_result['confidence']
        
        # Check for possession change
        if self._should_change_possession(player_id, team_id, confidence):
            self._end_current_possession(frame_idx)
            self._start_new_possession(player_id, team_id, frame_idx, ball_track)
            
        return PossessionInfo(
            frame_idx=frame_idx,
            player_id=player_id,
            team_id=team_id,
            ball_position=ball_track.current_position,
            confidence=confidence,
            duration=self._get_possession_duration(frame_idx),
            possession_change=self._is_possession_change(frame_idx)
        )
        
    def _find_player_in_possession(self,
                                  player_tracks: List[Track],
                                  ball_track: Track) -> Optional[Dict]:
        """Find which player has possession of the ball"""
        if not player_tracks or not ball_track.current_position:
            return None
            
        ball_pos = ball_track.current_position
        candidates = []
        
        for track in player_tracks:
            if track.current_position is None:
                continue
                
            # Calculate distance to ball
            distance = np.linalg.norm(track.current_position - ball_pos)
            
            if distance <= self.ball_proximity_threshold:
                # Calculate confidence based on distance
                confidence = 1.0 - (distance / self.ball_proximity_threshold)
                
                candidates.append({
                    'player_id': track.id,
                    'team_id': track.team_id,
                    'distance': distance,
                    'confidence': confidence
                })
                
        if not candidates:
            return None
            
        # Return closest player
        return min(candidates, key=lambda x: x['distance'])
        
    def _should_change_possession(self, player_id: int, team_id: int, 
                                 confidence: float) -> bool:
        """Determine if possession should change"""
        # No current possession
        if self.current_possession is None:
            return True
            
        current_player = self.current_possession.get('player_id')
        current_team = self.current_possession.get('team_id')
        
        # Same player - no change
        if player_id == current_player:
            self.possession_candidate = None
            self.candidate_frames = 0
            return False
            
        # Different player/team - track candidate
        if player_id != current_player or team_id != current_team:
            if (self.possession_candidate is None or 
                self.possession_candidate['player_id'] != player_id):
                self.possession_candidate = {
                    'player_id': player_id,
                    'team_id': team_id
                }
                self.candidate_frames = 1
                self.candidate_confidence = confidence
            else:
                self.candidate_frames += 1
                self.candidate_confidence = max(self.candidate_confidence, confidence)
                
            # Confirm change after threshold
            if self.candidate_frames >= self.possession_change_threshold:
                return True
                
        return False
        
    def _start_new_possession(self, player_id: int, team_id: int, 
                             frame_idx: int, ball_track: Track):
        """Start tracking new possession"""
        self.current_possession = {
            'player_id': player_id,
            'team_id': team_id,
            'start_frame': frame_idx,
            'ball_positions': [ball_track.current_position]
        }
        
        # Reset candidate
        self.possession_candidate = None
        self.candidate_frames = 0
        self.candidate_confidence = 0.0
        
    def _end_current_possession(self, frame_idx: int):
        """End current possession and store in history"""
        if self.current_possession is None:
            return
            
        duration = frame_idx - self.current_possession['start_frame']
        
        if duration >= self.min_possession_duration:
            self.current_possession['end_frame'] = frame_idx
            self.current_possession['duration'] = duration
            self.possession_history.append(self.current_possession.copy())
            
        self.current_possession = None
        
    def _no_possession_update(self, frame_idx: int) -> PossessionInfo:
        """Handle frame with no clear possession"""
        return PossessionInfo(
            frame_idx=frame_idx,
            player_id=None,
            team_id=None,
            ball_position=None,
            confidence=0.0,
            duration=self._get_possession_duration(frame_idx),
            possession_change=False
        )
        
    def _get_possession_duration(self, frame_idx: int) -> int:
        """Get duration of current possession"""
        if self.current_possession is None:
            return 0
        return frame_idx - self.current_possession['start_frame']
        
    def _is_possession_change(self, frame_idx: int) -> bool:
        """Check if possession changed in recent frames"""
        if len(self.possession_history) < 2:
            return False
            
        # Check if possession changed in last few frames
        for i in range(1, min(4, len(self.frame_history))):
            if frame_idx - i == self.current_possession.get('start_frame', -1):
                return True
        return False
        
    def get_possession_stats(self) -> Dict:
        """Get possession statistics"""
        team_possessions = {0: 0, 1: 0}
        total_duration = {0: 0, 1: 0}
        
        for possession in self.possession_history:
            team_id = possession.get('team_id')
            duration = possession.get('duration', 0)
            
            if team_id in [0, 1]:
                team_possessions[team_id] += 1
                total_duration[team_id] += duration
                
        return {
            'total_possessions': len(self.possession_history),
            'team_possessions': team_possessions,
            'avg_possession_duration': {
                0: total_duration[0] / max(team_possessions[0], 1),
                1: total_duration[1] / max(team_possessions[1], 1)
            }
        }
