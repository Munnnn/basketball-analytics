
"""
Mask overlay rendering
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from core import Detection, Track
from ..colors import TeamColorManager


class MaskAnnotator:
    """Render segmentation masks"""

    def __init__(self, opacity: float = 0.3):
        """
        Initialize mask annotator

        Args:
            opacity: Mask opacity (0-1)
        """
        self.opacity = opacity
        self.color_manager = TeamColorManager()

    def annotate(self, frame: np.ndarray,
                detections: List[Detection],
                tracks: List[Track]) -> np.ndarray:
        """Add mask overlays to frame"""
        overlay = frame.copy()

        # Create track lookup
        track_lookup = {}
        for track in tracks:
            if track.current_bbox is not None:
                # Use bbox as key (simplified)
                key = tuple(track.current_bbox.astype(int))
                track_lookup[key] = track

        # Apply masks
        for detection in detections:
            if detection.mask is None:
                continue

            # Find associated track
            bbox_key = tuple(detection.bbox.astype(int))
            track = track_lookup.get(bbox_key)

            # Get color
            if track:
                color = self.color_manager.get_track_color(track)
            else:
                color = self.color_manager.get_detection_color(detection)

            # Apply mask
            self._apply_mask(overlay, detection.mask, color)

        # Blend with original
        result = cv2.addWeighted(frame, 1 - self.opacity, overlay, self.opacity, 0)

        return result

    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray,
                   color: Tuple[int, int, int]):
        """Apply colored mask to frame"""
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Apply color to mask area
        frame[mask] = color
