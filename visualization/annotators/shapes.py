"""
Shape annotations (boxes, ellipses, triangles)
"""

import cv2
import numpy as np
from typing import List, Optional

from core import Detection, Track
from ..colors import TeamColorManager


class ShapeAnnotator:
    """Draw shape annotations on frames"""
    
    def __init__(self, use_ellipse: bool = True, thickness: int = 2):
        """
        Initialize shape annotator
        
        Args:
            use_ellipse: Use ellipses for players
            thickness: Line thickness
        """
        self.use_ellipse = use_ellipse
        self.thickness = thickness
        self.color_manager = TeamColorManager()
        
    def annotate(self, frame: np.ndarray, 
                detections: List[Detection],
                tracks: List[Track]) -> np.ndarray:
        """Add shape annotations to frame"""
        # Create track lookup
        track_lookup = {t.id: t for t in tracks}
        
        # Draw detections
        for detection in detections:
            # Find associated track
            track = None
            for t in tracks:
                if t.current_bbox is not None and np.allclose(t.current_bbox, detection.bbox):
                    track = t
                    break
                    
            # Get color
            if track:
                color = self.color_manager.get_track_color(track)
            else:
                color = self.color_manager.get_detection_color(detection)
                
            # Draw shape based on class
            if detection.class_id in [5, 6] and self.use_ellipse:  # Players and refs
                self._draw_ellipse(frame, detection.bbox, color)
            elif detection.class_id == 1:  # Ball
                self._draw_triangle(frame, detection.bbox, color)
            else:
                self._draw_box(frame, detection.bbox, color)
                
        return frame
        
    def _draw_box(self, frame: np.ndarray, bbox: np.ndarray, 
                  color: Tuple[int, int, int]):
        """Draw bounding box"""
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
    def _draw_ellipse(self, frame: np.ndarray, bbox: np.ndarray,
                     color: Tuple[int, int, int]):
        """Draw ellipse annotation"""
        x1, y1, x2, y2 = bbox.astype(int)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, self.thickness)
        
    def _draw_triangle(self, frame: np.ndarray, bbox: np.ndarray,
                      color: Tuple[int, int, int]):
        """Draw triangle annotation for ball"""
        x1, y1, x2, y2 = bbox.astype(int)
        cx = (x1 + x2) // 2
        
        # Triangle points above bbox
        pts = np.array([
            [cx, y1 - 10],
            [cx - 10, y1 - 25],
            [cx + 10, y1 - 25]
        ], np.int32)
        
        cv2.fillPoly(frame, [pts], color)
