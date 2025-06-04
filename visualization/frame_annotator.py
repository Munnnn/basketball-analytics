"""
Main frame annotation orchestrator
"""

import numpy as np
from typing import List, Dict, Optional

from core import Detection, Track
from .annotators import ShapeAnnotator, LabelAnnotator, MaskAnnotator
from .overlays import PossessionOverlay, StatisticsOverlay, TimelineOverlay
from .colors import TeamColorManager


class FrameAnnotator:
    """Orchestrate all frame annotations"""
    
    def __init__(self, use_ellipse: bool = True):
        """
        Initialize frame annotator
        
        Args:
            use_ellipse: Use ellipse for player annotations
        """
        self.use_ellipse = use_ellipse
        
        # Initialize components
        self.color_manager = TeamColorManager()
        self.shape_annotator = ShapeAnnotator(use_ellipse=use_ellipse)
        self.label_annotator = LabelAnnotator()
        self.mask_annotator = MaskAnnotator()
        
        # Overlays
        self.possession_overlay = PossessionOverlay()
        self.statistics_overlay = StatisticsOverlay()
        self.timeline_overlay = TimelineOverlay()
        
    def annotate(self,
                frame: np.ndarray,
                detections: List[Detection],
                tracks: List[Track],
                analytics_data: Dict) -> np.ndarray:
        """
        Annotate frame with all visualizations
        
        Args:
            frame: Input frame
            detections: All detections
            tracks: Active tracks
            analytics_data: Analytics data for overlays
            
        Returns:
            Annotated frame
        """
        # Start with copy
        annotated = frame.copy()
        
        # Apply masks if available
        if any(d.mask is not None for d in detections):
            annotated = self.mask_annotator.annotate(annotated, detections, tracks)
            
        # Draw shapes (boxes/ellipses)
        annotated = self.shape_annotator.annotate(annotated, detections, tracks)
        
        # Add labels
        annotated = self.label_annotator.annotate(annotated, detections, tracks)
        
        # Add overlays
        if analytics_data.get('possession'):
            annotated = self.possession_overlay.draw(
                annotated, analytics_data['possession']
            )
        if analytics_data.get('team_stats'):
            annotated = self.statistics_overlay.draw(
                annotated, analytics_data['team_stats']
            )
            
        if analytics_data.get('events'):
            annotated = self.timeline_overlay.draw(
                annotated, analytics_data['events']
            )
            
        return annotated
