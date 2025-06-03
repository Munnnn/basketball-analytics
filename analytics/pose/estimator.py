"""
Pose estimation wrapper
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging

# Try to import pose estimation libraries
try:
    import openpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseEstimator:
    """Unified pose estimation interface"""
    
    def __init__(self, backend: str = 'auto'):
        """
        Initialize pose estimator
        
        Args:
            backend: 'openpose', 'mediapipe', or 'auto'
        """
        self.backend = backend
        self.initialized = False
        self.pose_detector = None
        
        if backend == 'auto':
            if OPENPOSE_AVAILABLE:
                self._init_openpose()
            elif MEDIAPIPE_AVAILABLE:
                self._init_mediapipe()
            else:
                logging.warning("No pose estimation backend available")
        elif backend == 'openpose' and OPENPOSE_AVAILABLE:
            self._init_openpose()
        elif backend == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            logging.warning(f"Backend {backend} not available")
            
    def _init_openpose(self):
        """Initialize OpenPose backend"""
        try:
            params = dict()
            params["model_folder"] = "./models/openpose/"
            params["face"] = False
            params["hand"] = False
            params["net_resolution"] = "368x368"
            params["output_resolution"] = "1280x720"
            params["num_gpu_start"] = 0
            
            self.openpose = op.WrapperPython()
            self.openpose.configure(params)
            self.openpose.start()
            
            self.backend = 'openpose'
            self.initialized = True
            logging.info("OpenPose initialized successfully")
            
        except Exception as e:
            logging.error(f"OpenPose initialization failed: {e}")
            self._init_mediapipe()  # Fallback
            
    def _init_mediapipe(self):
        """Initialize MediaPipe backend"""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.backend = 'mediapipe'
            self.initialized = True
            logging.info("MediaPipe pose estimation initialized")
            
        except Exception as e:
            logging.error(f"MediaPipe initialization failed: {e}")
            
    def extract_poses(self, crops: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Extract pose keypoints from player crops
        
        Args:
            crops: List of player crop images
            
        Returns:
            List of pose data dictionaries
        """
        if not self.initialized:
            return [None] * len(crops)
            
        poses = []
        for crop in crops:
            try:
                if self.backend == 'openpose':
                    pose_data = self._extract_openpose(crop)
                else:
                    pose_data = self._extract_mediapipe(crop)
                poses.append(pose_data)
            except Exception as e:
                logging.warning(f"Pose extraction failed: {e}")
                poses.append(None)
                
        return poses
        
    def _extract_openpose(self, crop: np.ndarray) -> Optional[Dict]:
        """Extract pose using OpenPose"""
        try:
            datum = op.Datum()
            datum.cvInputData = crop
            self.openpose.emplaceAndPop(op.VectorDatum([datum]))
            
            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                keypoints = datum.poseKeypoints[0]
                return self._process_openpose_keypoints(keypoints)
                
        except Exception as e:
            logging.warning(f"OpenPose extraction failed: {e}")
            
        return None
        
    def _extract_mediapipe(self, crop: np.ndarray) -> Optional[Dict]:
        """Extract pose using MediaPipe"""
        try:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_crop)
            
            if results.pose_landmarks:
                return self._process_mediapipe_keypoints(
                    results.pose_landmarks, crop.shape
                )
                
        except Exception as e:
            logging.warning(f"MediaPipe extraction failed: {e}")
            
        return None
        
    def _process_openpose_keypoints(self, keypoints: np.ndarray) -> Dict:
        """Process OpenPose keypoints"""
        keypoint_names = [
            'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
            'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'
        ]
        
        pose_data = {'keypoints': {}, 'confidence': 0.0}
        valid_points = 0
        total_confidence = 0.0
        
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                if conf > 0.1:
                    pose_data['keypoints'][name] = {
                        'x': float(x),
                        'y': float(y),
                        'confidence': float(conf)
                    }
                    valid_points += 1
                    total_confidence += conf
                    
        if valid_points > 0:
            pose_data['confidence'] = total_confidence / valid_points
            
        return pose_data
        
    def _process_mediapipe_keypoints(self, landmarks, shape: Tuple) -> Dict:
        """Process MediaPipe landmarks"""
        h, w = shape[:2]
        
        keypoint_map = {
            0: 'nose', 11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow', 15: 'left_wrist',
            16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee', 27: 'left_ankle',
            28: 'right_ankle'
        }
        
        pose_data = {'keypoints': {}, 'confidence': 0.0}
        valid_points = 0
        total_confidence = 0.0
        
        for idx, name in keypoint_map.items():
            landmark = landmarks.landmark[idx]
            if landmark.visibility > 0.5:
                pose_data['keypoints'][name] = {
                    'x': float(landmark.x * w),
                    'y': float(landmark.y * h),
                    'confidence': float(landmark.visibility)
                }
                valid_points += 1
                total_confidence += landmark.visibility
                
        if valid_points > 0:
            pose_data['confidence'] = total_confidence / valid_points
            
        return pose_data
