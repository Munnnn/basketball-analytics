"""
Timeline and event overlay
"""

import cv2
import numpy as np
from typing import List

from core import PlayEvent


class TimelineOverlay:
    """Draw timeline and recent events"""
    
    def __init__(self, max_events: int = 3):
        """
        Initialize timeline overlay
        
        Args:
            max_events: Maximum recent events to show
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position = (10, 250)
        self.max_events = max_events
        
    def draw(self, frame: np.ndarray, events: List[PlayEvent]) -> np.ndarray:
        """Draw recent events on frame"""
        if not events:
            return frame
            
        x, y = self.position
        
        # Draw header
        cv2.putText(
            frame, "RECENT EVENTS:", (x, y),
            self.font, 0.5, (255, 255, 255), 2
        )
        
        # Show recent events
        recent_events = events[-self.max_events:]
        y_offset = 25
        
        for event in recent_events:
            # Format event text
            text = event.type
            if event.team_id is not None:
                text += f" (T{event.team_id})"
                
            # Get event color
            if event.type == 'shot_attempt':
                color = (0, 255, 0)  # Green
            elif event.type == 'potential_rebound':
                color = (0, 165, 255)  # Orange
            else:
                color = (255, 255, 0)  # Yellow
                
            cv2.putText(
                frame, text, (x, y + y_offset),
                self.font, 0.4, color, 1
            )
            
            y_offset += 15
            
        return frame
