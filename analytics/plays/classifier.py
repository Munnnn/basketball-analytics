"""
Basketball play type classification
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from core import Track, PlayClassification
from core.constants import PLAY_TYPES


class PlayTypeClassifier:
    """Classify basketball play types"""
    
    def __init__(self):
        """Initialize play classifier"""
        self.play_types = PLAY_TYPES
        
        # Play detection thresholds
        self.fast_break_speed_threshold = 180
        self.transition_speed_range = (100, 180)
        self.post_up_duration_threshold = 6
        self.pick_distance_threshold = 80
        self.isolation_spacing_threshold = 120
        
    def classify_play(self,
                     possession_data: Dict,
                     context: Optional[Dict] = None) -> PlayClassification:
        """
        Classify the type of basketball play
        
        Args:
            possession_data: Data about current possession
            context: Optional context information
            
        Returns:
            Play classification result
        """
        # Extract data
        player_positions = possession_data.get('player_positions', [])
        ball_position = possession_data.get('ball_position')
        velocities = possession_data.get('velocities', [])
        duration = possession_data.get('duration', 0)
        team_id = possession_data.get('team_id')
       
        #debug
        print(f"[PLAY DEBUG] Classifying play for team {team_id} | Duration: {duration}")
        print(f"  - Ball position: {ball_position}")
        print(f"  - Player positions: {player_positions}")
        print(f"  - Velocities: {velocities}")
        if duration < 3:
            print("  ⚠️ Possession duration too short to classify")

        # Get base classification
        play_type = self._get_base_classification(
            player_positions, ball_position, velocities, duration
        )
        print(f"  → Base play type detected: {play_type} ({self.play_types.get(play_type, 'Unknown')})")
        
        # Apply context modifications if available
        if context:
            play_type = self._apply_context_modifications(
                play_type, possession_data, context
            )
            
        # Get play name
        play_name = self.play_types.get(play_type, "Unknown")
        
        # Calculate confidence
        confidence = self._calculate_confidence(play_type, possession_data)
        
        return PlayClassification(
            play_type=play_type,
            play_name=play_name,
            confidence=confidence,
            start_frame=possession_data.get('start_frame', 0),
            end_frame=possession_data.get('end_frame', 0),
            team_id=team_id,
            key_players=possession_data.get('key_players', []),
            events=possession_data.get('events', [])
        )
        
    def _get_base_classification(self,
                               player_positions: List,
                               ball_position: Optional[np.ndarray],
                               velocities: List,
                               duration: int) -> int:
        """Get base play classification"""
        # Fast break detection
        if self._is_fast_break(velocities, duration):
            return 3
            
        # Transition detection
        if self._is_transition(velocities, duration):
            return 9
            
        # Post up detection
        if self._is_post_up(ball_position, duration):
            return 2
            
        # Pick and roll detection
        if self._is_pick_and_roll(player_positions, duration):
            return 1
            
        # Isolation detection
        if self._is_isolation(player_positions, duration):
            return 0
            
        # Default to half court set
        return 10
        
    def _is_fast_break(self, velocities: List, duration: int) -> bool:
        """Check if play is a fast break"""
        if duration > 8 or not velocities:
            return False
            
        # Count fast-moving players
        fast_players = 0
        for velocity in velocities:
            if velocity is not None:
                speed = np.linalg.norm(velocity)
                if speed > self.fast_break_speed_threshold:
                    fast_players += 1
                    
        return fast_players >= 2
        
    def _is_transition(self, velocities: List, duration: int) -> bool:
        """Check if play is a transition"""
        if duration < 3 or duration > 15 or not velocities:
            return False
            
        # Check average speed
        speeds = []
        for velocity in velocities:
            if velocity is not None:
                speeds.append(np.linalg.norm(velocity))
                
        if not speeds:
            return False
            
        avg_speed = np.mean(speeds)
        min_speed, max_speed = self.transition_speed_range
        
        return min_speed < avg_speed < max_speed
        
    def _is_post_up(self, ball_position: Optional[np.ndarray], 
                    duration: int) -> bool:
        """Check if play is a post up"""
        if ball_position is None or duration < self.post_up_duration_threshold:
            return False
            
        # Check if ball is in low post area (lower part of court)
        # Assuming court height of 720 pixels
        return ball_position[1] > 540  # Lower 25% of court
        
    def _is_pick_and_roll(self, player_positions: List, duration: int) -> bool:
        """Check if play is pick and roll"""
        if len(player_positions) < 2 or duration < 4:
            return False
            
        # Check for close proximity between players
        for i in range(len(player_positions)):
            for j in range(i + 1, len(player_positions)):
                if isinstance(player_positions[i], dict) and isinstance(player_positions[j], dict):
                    pos1 = list(player_positions[i].values())
                    pos2 = list(player_positions[j].values())
                    
                    if pos1 and pos2:
                        distance = np.linalg.norm(
                            np.array(pos1[0]) - np.array(pos2[0])
                        )
                        if distance < self.pick_distance_threshold:
                            return True
                            
        return False
        
    def _is_isolation(self, player_positions: List, duration: int) -> bool:
        """Check if play is isolation"""
        if len(player_positions) < 3 or duration < 4:
            return False
            
        # Check spacing between players
        positions = []
        for pos_dict in player_positions:
            if isinstance(pos_dict, dict):
                for pos in pos_dict.values():
                    positions.append(np.array(pos))
                    
        if len(positions) < 3:
            return False
            
        # Calculate average distance between players
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distances.append(np.linalg.norm(positions[i] - positions[j]))
                
        if not distances:
            return False
            
        avg_distance = np.mean(distances)
        return avg_distance > self.isolation_spacing_threshold
        
    def _apply_context_modifications(self,
                                   base_play_type: int,
                                   possession_data: Dict,
                                   context: Dict) -> int:
        """Apply context-based modifications to play classification"""
        momentum = context.get('team_momentum', {}).get(
            possession_data.get('team_id', 0), 0.0
        )
        
        # High momentum modifications
        if momentum > 0.3:
            if base_play_type == 10:  # Half court set
                tempo = context.get('tempo', 1.0)
                if tempo > 1.3:
                    return 9  # Upgrade to transition
                    
        # Low momentum modifications
        elif momentum < -0.3:
            if base_play_type == 3:  # Fast break
                return 9  # Downgrade to transition
                
        return base_play_type
        
    def _calculate_confidence(self, play_type: int, possession_data: Dict) -> float:
        """Calculate confidence in play classification"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on data quality
        if possession_data.get('duration', 0) >= 5:
            confidence += 0.1
            
        if len(possession_data.get('player_positions', [])) >= 4:
            confidence += 0.1
            
        if possession_data.get('velocities'):
            confidence += 0.1
            
        return min(confidence, 1.0)


class PlayClassifier(PlayTypeClassifier):
    """Alias for backward compatibility"""
    pass
