"""
Kalman filter for motion prediction
"""

import numpy as np
from typing import Tuple

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("Warning: filterpy not installed. Kalman filtering disabled.")


class KalmanTracker:
    """Kalman filter for smooth motion estimation"""
    
    def __init__(self, initial_bbox: np.ndarray, dt: float = 1.0):
        """
        Initialize Kalman filter
        
        Args:
            initial_bbox: Initial bounding box [x1, y1, x2, y2]
            dt: Time step
        """
        if not FILTERPY_AVAILABLE:
            raise ImportError("filterpy required for Kalman filtering")
            
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.01, block_size=2)
        
        # Measurement noise
        self.kf.R *= 10
        
        # Initialize state
        x, y, w, h = self._bbox_to_xywh(initial_bbox)
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).T
        self.kf.P *= 100
        
    def predict(self) -> np.ndarray:
        """Predict next state"""
        self.kf.predict()
        x, y, w, h = self.kf.x[:4]
        return self._xywh_to_bbox(x, y, w, h)
        
    def update(self, bbox: np.ndarray):
        """Update with measurement"""
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.update(np.array([x, y, w, h]))
        
    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        x, y, w, h = self.kf.x[:4]
        return self._xywh_to_bbox(x, y, w, h)
        
    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate"""
        return self.kf.x[4:6]
        
    def _bbox_to_xywh(self, bbox: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert bbox to center format"""
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return x, y, w, h
        
    def _xywh_to_bbox(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        """Convert center format to bbox"""
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
