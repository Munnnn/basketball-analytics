"""
Basketball action detection from pose data
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class ActionDetector:
    """Detect basketball actions from pose keypoints"""
    
    def __init__(self):
        """Initialize action detector"""
        self.action_thresholds = {
            'screen': {
                'stance_width': 40,
                'arm_extension': 0.7
            },
            'shooting': {
                'elbow_angle': 90,
                'wrist_height': 0.8
            },
            'cutting': {
                'direction_change': 45,
                'min_speed': 5
            }
        }
        
    def detect_actions(self, 
                      poses: List[Optional[Dict]],
                      positions: List[np.ndarray],
                      frame_history: Optional[List] = None) -> Dict:
        """
        Detect basketball actions from poses
        
        Args:
            poses: List of pose data
            positions: List of player positions
            frame_history: Historical frame data
            
        Returns:
            Dictionary of detected actions
        """
        result = {
            'screens': [],
            'cuts': [],
            'shots': [],
            'picks': []
        }
        
        if not poses or len(poses) != len(positions):
            return result
            
        for i, (pose, position) in enumerate(zip(poses, positions)):
            if pose is None:
                continue
                
            # Calculate pose features
            features = self._calculate_pose_features(pose['keypoints'])
            
            # Detect individual actions
            if self._is_screening(features):
                result['screens'].append({
                    'player_index': i,
                    'position': position.tolist(),
                    'confidence': features.get('screen_confidence', 0.8)
                })
                
            if self._is_shooting(features):
                result['shots'].append({
                    'player_index': i,
                    'position': position.tolist(),
                    'confidence': features.get('shot_confidence', 0.8)
                })
                
            # Detect cuts using movement history
            if frame_history and self._is_cutting(i, frame_history):
                result['cuts'].append({
                    'player_index': i,
                    'position': position.tolist(),
                    'confidence': 0.7
                })
                
        # Detect picks (screens with nearby players)
        result['picks'] = self._detect_picks(result['screens'], positions)
        
        return result
        
    def _calculate_pose_features(self, keypoints: Dict) -> Dict:
        """Calculate features from pose keypoints"""
        features = {
            'body_angle': 0.0,
            'arm_extension': 0.0,
            'stance_width': 0.0,
            'elbow_angle': 0.0,
            'wrist_height_ratio': 0.0
        }
        
        try:
            # Body angle
            if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
                left = keypoints['left_shoulder']
                right = keypoints['right_shoulder']
                
                if left['confidence'] > 0.5 and right['confidence'] > 0.5:
                    dx = right['x'] - left['x']
                    dy = right['y'] - left['y']
                    features['body_angle'] = float(np.arctan2(dy, dx))
                    
            # Stance width
            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                left_hip = keypoints['left_hip']
                right_hip = keypoints['right_hip']
                
                if left_hip['confidence'] > 0.5 and right_hip['confidence'] > 0.5:
                    features['stance_width'] = float(np.sqrt(
                        (right_hip['x'] - left_hip['x'])**2 +
                        (right_hip['y'] - left_hip['y'])**2
                    ))
                    
            # Arm extension
            arm_extensions = []
            for side in ['left', 'right']:
                shoulder_key = f'{side}_shoulder'
                elbow_key = f'{side}_elbow'
                wrist_key = f'{side}_wrist'
                
                if all(key in keypoints for key in [shoulder_key, elbow_key, wrist_key]):
                    shoulder = keypoints[shoulder_key]
                    elbow = keypoints[elbow_key]
                    wrist = keypoints[wrist_key]
                    
                    if all(p['confidence'] > 0.5 for p in [shoulder, elbow, wrist]):
                        # Calculate extension ratio
                        shoulder_to_wrist = np.sqrt(
                            (wrist['x'] - shoulder['x'])**2 +
                            (wrist['y'] - shoulder['y'])**2
                        )
                        shoulder_to_elbow = np.sqrt(
                            (elbow['x'] - shoulder['x'])**2 +
                            (elbow['y'] - shoulder['y'])**2
                        )
                        elbow_to_wrist = np.sqrt(
                            (wrist['x'] - elbow['x'])**2 +
                            (wrist['y'] - elbow['y'])**2
                        )
                        
                        max_extension = shoulder_to_elbow + elbow_to_wrist
                        if max_extension > 0:
                            extension_ratio = shoulder_to_wrist / max_extension
                            arm_extensions.append(extension_ratio)
                            
                        # Calculate elbow angle for shooting detection
                        if side == 'right':
                            features['elbow_angle'] = self._calculate_angle(
                                shoulder, elbow, wrist
                            )
                            
            if arm_extensions:
                features['arm_extension'] = float(np.mean(arm_extensions))
                
            # Wrist height ratio (for shooting)
            if 'right_wrist' in keypoints and 'right_shoulder' in keypoints:
                wrist = keypoints['right_wrist']
                shoulder = keypoints['right_shoulder']
                
                if wrist['confidence'] > 0.5 and shoulder['confidence'] > 0.5:
                    # Lower y means higher position
                    if shoulder['y'] > 0:
                        features['wrist_height_ratio'] = 1.0 - (wrist['y'] / shoulder['y'])
                        
        except Exception as e:
            logging.warning(f"Error calculating pose features: {e}")
            
        return features
        
    def _is_screening(self, features: Dict) -> bool:
        """Check if player is setting a screen"""
        thresholds = self.action_thresholds['screen']
        
        is_wide_stance = features['stance_width'] > thresholds['stance_width']
        is_arms_extended = features['arm_extension'] > thresholds['arm_extension']
        
        if is_wide_stance and is_arms_extended:
            features['screen_confidence'] = min(
                features['arm_extension'],
                features['stance_width'] / 50.0
            )
            return True
            
        return False
        
    def _is_shooting(self, features: Dict) -> bool:
        """Check if player is in shooting motion"""
        thresholds = self.action_thresholds['shooting']
        
        # Check elbow angle (around 90 degrees for shooting)
        good_elbow = abs(features['elbow_angle'] - 90) < 30
        
        # Check wrist above shoulder
        wrist_high = features['wrist_height_ratio'] > thresholds['wrist_height']
        
        if good_elbow and wrist_high:
            features['shot_confidence'] = min(
                1.0 - abs(features['elbow_angle'] - 90) / 90,
                features['wrist_height_ratio']
            )
            return True
            
        return False
        
    def _is_cutting(self, player_index: int, frame_history: List) -> bool:
        """Check if player is making a cut"""
        if len(frame_history) < 3:
            return False
            
        try:
            # Get recent positions
            recent_positions = []
            for frame_data in frame_history[-5:]:
                if 'player_positions' in frame_data:
                    positions = frame_data['player_positions']
                    if player_index < len(positions):
                        recent_positions.append(positions[player_index])
                        
            if len(recent_positions) < 3:
                return False
                
            # Calculate movement vectors
            movements = []
            for i in range(1, len(recent_positions)):
                movement = np.array(recent_positions[i]) - np.array(recent_positions[i-1])
                movements.append(movement)
                
            # Detect sharp direction changes
            for i in range(1, len(movements)):
                prev_movement = movements[i-1]
                curr_movement = movements[i]
                
                prev_norm = np.linalg.norm(prev_movement)
                curr_norm = np.linalg.norm(curr_movement)
                
                if prev_norm > 5 and curr_norm > 5:  # Minimum movement
                    # Calculate angle between movements
                    dot_product = np.dot(prev_movement, curr_movement)
                    angle = np.arccos(np.clip(
                        dot_product / (prev_norm * curr_norm), -1, 1
                    ))
                    
                    if angle > np.radians(self.action_thresholds['cutting']['direction_change']):
                        return True
                        
        except Exception as e:
            logging.warning(f"Error detecting cut: {e}")
            
        return False
        
    def _detect_picks(self, screens: List[Dict], positions: List[np.ndarray]) -> List[Dict]:
        """Detect pick plays (screens with cutters)"""
        picks = []
        
        for screen in screens:
            screener_idx = screen['player_index']
            screener_pos = np.array(screen['position'])
            
            # Find nearby players
            for j, other_pos in enumerate(positions):
                if j != screener_idx:
                    distance = np.linalg.norm(screener_pos - other_pos)
                    
                    if distance < 80:  # Close proximity
                        picks.append({
                            'screener_index': screener_idx,
                            'cutter_index': j,
                            'position': screen['position'],
                            'confidence': screen['confidence'] * 0.9
                        })
                        
        return picks
        
    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
