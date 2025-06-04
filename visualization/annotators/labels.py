"""
Label generation and rendering
"""

import cv2
import numpy as np
from typing import List, Optional, Dict

from core import Detection, Track


class LabelAnnotator:
    """Generate and draw labels"""
    
    def __init__(self, 
                 font: int = cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale: float = 0.5,
                 thickness: int = 2):
        """
        Initialize label annotator
        
        Args:
            font: OpenCV font type
            font_scale: Font scale factor
            thickness: Text thickness
        """
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        
    def annotate(self, frame: np.ndarray,
                detections: List[Detection],
                tracks: List[Track]) -> np.ndarray:
        """Add labels to frame"""
        # Create labels for tracks
        track_labels = self._create_track_labels(tracks)
        
        # Draw labels
        for track in tracks:
            if track.current_bbox is not None and track.id in track_labels:
                self._draw_label(
                    frame, 
                    track.current_bbox,
                    track_labels[track.id],
                    self._get_label_color(track)
                )
                
        return frame
        
    def _create_track_labels(self, tracks: List[Track]) -> Dict[int, str]:
        """Create labels for tracks"""
        labels = {}
        
        for track in tracks:
            parts = []
            
            # Add prefix based on class
            parts.append(f"P#{track.id}")
            
            # Add team
            if track.team_id is not None:
                parts.append(f"T{track.team_id}")
                
            # Add confidence
            if track.confidence > 0:
                parts.append(f"({track.confidence:.2f})")
                
            labels[track.id] = " ".join(parts)
            
        return labels
        
    def _draw_label(self, frame: np.ndarray, bbox: np.ndarray,
                   label: str, color: Tuple[int, int, int]):
        """Draw label on frame"""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.thickness
        )
        
        # Position above bbox
        text_x = x1
        text_y = y1 - 5
        
        # Draw background
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (text_x, text_y - baseline),
            self.font,
            self.font_scale,
            (0, 0, 0),  # Black text
            self.thickness
        )
        
    def _get_label_color(self, track: Track) -> Tuple[int, int, int]:
        """Get label color based on team"""
        if track.team_id == 0:
            return (0, 0, 255)  # Red
        elif track.team_id == 1:
            return (255, 0, 0)  # Blue
        else:
            return (128, 128, 128)  # Gray
