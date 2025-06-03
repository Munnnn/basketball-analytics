"""
Optical flow-based tracking
"""

import numpy as np
import cv2
from typing import Optional
import logging


class OpticalFlowTracker:
    """Optical flow-based mask propagation"""
    
    def __init__(self):
        """Initialize optical flow tracker"""
        self.prev_frame = None
        self.prev_points = None
        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def propagate_mask(self, current_frame: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Propagate mask to next frame using optical flow
        
        Args:
            current_frame: Current video frame
            mask: Binary mask to propagate
            
        Returns:
            Propagated mask or None if failed
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return mask
            
        try:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Extract feature points from mask
            if self.prev_points is None or len(self.prev_points) < 10:
                mask_uint8 = mask.astype(np.uint8) * 255
                corners = cv2.goodFeaturesToTrack(
                    mask_uint8,
                    maxCorners=50,
                    qualityLevel=0.01,
                    minDistance=10
                )
                
                if corners is not None and len(corners) > 0:
                    self.prev_points = corners
                else:
                    # Sample points from mask
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        step = max(1, len(y_coords) // 30)
                        sampled_indices = np.arange(0, len(y_coords), step)
                        self.prev_points = np.array([
                            [[x_coords[i], y_coords[i]]] for i in sampled_indices
                        ], dtype=np.float32)
                    else:
                        self.prev_frame = current_gray
                        return mask
                        
            if self.prev_points is None or len(self.prev_points) == 0:
                self.prev_frame = current_gray
                return mask
                
            # Calculate optical flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame,
                current_gray,
                self.prev_points,
                None,
                **self.flow_params
            )
            
            # Filter good points
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            if len(good_new) > 5:
                # Estimate transformation
                transform_matrix = cv2.estimateAffinePartial2D(good_old, good_new)[0]
                
                if transform_matrix is not None:
                    h, w = mask.shape
                    propagated_mask = cv2.warpAffine(
                        mask.astype(np.uint8),
                        transform_matrix,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    ).astype(bool)
                    
                    self.prev_points = good_new.reshape(-1, 1, 2)
                    self.prev_frame = current_gray
                    return propagated_mask
                    
        except Exception as e:
            logging.warning(f"Optical flow propagation failed: {e}")
            
        self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = None
        return mask
        
    def reset(self):
        """Reset optical flow state"""
        self.prev_frame = None
        self.prev_points = None
