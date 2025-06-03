"""
Track association algorithms
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from core import Detection, Track


class TrackAssociator:
    """Associates detections with existing tracks"""
    
    def __init__(self, minimum_matching_threshold: float = 0.2):
        """
        Initialize track associator
        
        Args:
            minimum_matching_threshold: Minimum IoU for valid association
        """
        self.minimum_matching_threshold = minimum_matching_threshold
        
    def associate(self, 
                  detections: List[Detection], 
                  tracks: List[Track],
                  features: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with tracks
        
        Args:
            detections: Current frame detections
            tracks: Existing tracks
            features: Optional appearance features
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
            where matches is list of (track_id, detection_idx) pairs
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), [t.id for t in tracks]
            
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(detections, tracks, features)
        
        # Solve assignment problem
        matches, unmatched_detections, unmatched_tracks = self._solve_assignment(
            cost_matrix, tracks
        )
        
        return matches, unmatched_detections, unmatched_tracks
        
    def _build_cost_matrix(self, 
                          detections: List[Detection], 
                          tracks: List[Track],
                          features: Optional[np.ndarray] = None) -> np.ndarray:
        """Build cost matrix for assignment"""
        n_tracks = len(tracks)
        n_detections = len(detections)
        cost_matrix = np.ones((n_tracks, n_detections))
        
        for i, track in enumerate(tracks):
            track_bbox = track.current_bbox
            if track_bbox is None:
                continue
                
            for j, detection in enumerate(detections):
                # IoU-based cost
                iou = self._compute_iou(track_bbox, detection.bbox)
                cost = 1.0 - iou
                
                # Add appearance cost if available
                if features is not None and hasattr(track, 'feature'):
                    feature_dist = np.linalg.norm(track.feature - features[j])
                    cost = 0.7 * cost + 0.3 * feature_dist
                    
                cost_matrix[i, j] = cost
                
        return cost_matrix
        
    def _solve_assignment(self, 
                         cost_matrix: np.ndarray, 
                         tracks: List[Track]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Solve assignment problem using Hungarian algorithm"""
        if cost_matrix.size == 0:
            return [], [], []
            
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(cost_matrix.shape[1]))
        
        # Process assignments
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1.0 - self.minimum_matching_threshold:
                matches.append((tracks[row].id, col))
                unmatched_detections.remove(col)
            else:
                unmatched_tracks.append(tracks[row].id)
                
        # Add tracks without assignments
        for i, track in enumerate(tracks):
            if i not in row_indices:
                unmatched_tracks.append(track.id)
                
        return matches, unmatched_detections, unmatched_tracks
        
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
