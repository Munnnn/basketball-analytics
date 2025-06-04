"""
Team statistics overlay
"""

import cv2
import numpy as np
from typing import Dict


class StatisticsOverlay:
    """Draw team statistics"""
    
    def __init__(self):
        """Initialize statistics overlay"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position = (10, 160)
        
    def draw(self, frame: np.ndarray, team_stats: Dict) -> np.ndarray:
        """Draw team statistics on frame"""
        if not team_stats:
            return frame
            
        x, y = self.position
        
        # Draw header
        cv2.putText(
            frame, "TEAM STATS:", (x, y),
            self.font, 0.5, (255, 255, 255), 2
        )
        
        y_offset = 25
        
        # Draw stats for each team
        for team_id in [0, 1]:
            if team_id not in team_stats:
                continue
                
            stats = team_stats[team_id]
            team_color = (0, 0, 255) if team_id == 0 else (255, 0, 0)
            
            # Format stats text
            possessions = stats.get('possessions', 0)
            shots = stats.get('shot_attempt', 0)
            
            text = f"T{team_id}: {possessions} poss"
            if shots > 0:
                text += f", {shots} shots"
                
            cv2.putText(
                frame, text, (x, y + y_offset),
                self.font, 0.4, team_color, 1
            )
            
            y_offset += 20
            
        return frame
