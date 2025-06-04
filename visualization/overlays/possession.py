"""
Possession indicator overlay
"""

import cv2
import numpy as np
from typing import Optional

from core import PossessionInfo


class PossessionOverlay:
    """Draw possession indicators"""
    
    def __init__(self):
        """Initialize possession overlay"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position = (10, 100)  # Default position
        
    def draw(self, frame: np.ndarray, possession_info: PossessionInfo) -> np.ndarray:
        """Draw possession indicator on frame"""
        if possession_info is None or possession_info.team_id is None:
            return frame
            
        # Get team color
        team_color = (0, 0, 255) if possession_info.team_id == 0 else (255, 0, 0)
        
        # Draw background
        x, y = self.position
        cv2.rectangle(frame, (x, y), (x + 290, y + 50), team_color, 3)
        cv2.rectangle(frame, (x + 3, y + 3), (x + 287, y + 47), (0, 0, 0), -1)
        
        # Create text
        text = f"POSSESSION: Team {possession_info.team_id}"
        if possession_info.player_id is not None:
            text += f" (Player #{possession_info.player_id})"
            
        # Draw text
        cv2.putText(
            frame, text, (x + 10, y + 30),
            self.font, 0.6, (255, 255, 255), 2
        )
        
        # Draw confidence bar
        if possession_info.confidence > 0:
            bar_width = int(270 * possession_info.confidence)
            cv2.rectangle(
                frame,
                (x + 10, y + 35),
                (x + 10 + bar_width, y + 42),
                team_color,
                -1
            )
            
        return frame
