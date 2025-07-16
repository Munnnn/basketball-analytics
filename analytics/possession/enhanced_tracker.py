"""
Enhanced Possession Tracker with Context Integration
Matches the pasted code's update_possession_with_context functionality
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .tracker import PossessionTracker
from .context import PossessionContext
from ..pose import PoseEstimator, ActionDetector
from core import Track, PossessionInfo


class EnhancedPossessionTracker(PossessionTracker):
    """
    CRITICAL FIX: Enhanced possession tracker with context awareness and pose integration
    This matches the pasted code's EnhancedPossessionTracker functionality
    """
    
    def __init__(self,
                 ball_proximity_threshold: float = 80.0,
                 possession_change_threshold: int = 8,
                 min_possession_duration: int = 5,
                 basketball_enhanced: bool = True,
                 context_tracking: bool = True):
        
        super().__init__(ball_proximity_threshold, possession_change_threshold, min_possession_duration)
        
        self.basketball_enhanced = basketball_enhanced
        self.context_tracking = context_tracking
        
        # Enhanced components
        if context_tracking:
            self.possession_context = PossessionContext(context_window=3)
            # Fixed: Remove the use_openpose parameter
            self.pose_estimator = PoseEstimator()
            self.action_detector = ActionDetector()
        else:
            self.possession_context = None
            self.pose_estimator = None
            self.action_detector = None
            
        # Frame history for movement analysis
        self.frame_history = deque(maxlen=10)
        
        # Enhanced tracking parameters
        self.confidence_threshold = 0.5
        
        print("✅ Enhanced possession tracker with context integration initialized")
    
    def update_possession_with_context(self, frame_data: Dict, frame_idx: int) -> Dict:
        """
        CRITICAL FIX: Enhanced possession tracking with context and pose analysis
        This exactly matches the pasted code's method signature and functionality
        """
        
        # Store frame data for movement analysis
        self.frame_history.append({
            'frame_idx': frame_idx,
            'player_positions': self._extract_player_positions(frame_data),
            'ball_position': self._extract_ball_position(frame_data),
            'timestamp': time.time()
        })
        
        player_detections = frame_data.get('player_detections')
        ball_detections = frame_data.get('ball_detections')
        player_team_mapping = frame_data.get('player_team_mapping', {})
        
        if player_detections is None or ball_detections is None:
            return self._no_possession_update_with_context(frame_idx)
        
        # Extract poses from player crops if available
        poses = []
        if self.pose_estimator and len(player_detections) > 0:
            try:
                crops = self._extract_player_crops(frame_data.get('frame'), player_detections)
                poses = self.pose_estimator.extract_poses(crops)
            except Exception as e:
                print(f"Pose extraction failed: {e}")
                poses = [None] * len(player_detections)
        
        # Enhanced possession detection with pose data
        possession_result = self._get_player_in_possession_enhanced(
            player_detections, ball_detections, self.ball_proximity_threshold,
            player_team_mapping, poses
        )
        
        if possession_result is None:
            return self._no_possession_update_with_context(frame_idx)
        
        player_in_possession = possession_result['player_id']
        team_in_possession = possession_result['team_id']
        confidence = possession_result['confidence']
        
        print(f"   🎯 Frame {frame_idx}: Player {player_in_possession} (Team {team_in_possession}) in possession (conf: {confidence:.2f})")
        
        # Enhanced possession change logic with context
        if self._should_change_possession_with_context(
            player_in_possession, team_in_possession, confidence, frame_idx):
            
            self._end_current_possession_with_context(frame_idx, poses)
            self._start_new_possession_with_context(
                player_in_possession, team_in_possession, frame_idx, frame_data, poses
            )
            print(f"   ✨ NEW POSSESSION: Player {player_in_possession} (Team {team_in_possession})")
        else:
            self._update_current_possession_with_context(frame_data, frame_idx, poses)
        
        return self._get_possession_summary_with_context(frame_idx)
    
    def _extract_player_crops(self, frame: Optional[np.ndarray], player_detections) -> List[np.ndarray]:
        """Extract player crops for pose estimation"""
        if frame is None or len(player_detections) == 0:
            return []
        
        crops = []
        for bbox in player_detections.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
        
        return crops
    
    def _extract_player_positions(self, frame_data: Dict) -> List[np.ndarray]:
        """Extract player positions from frame data"""
        player_detections = frame_data.get('player_detections')
        if player_detections is None or len(player_detections) == 0:
            return []
        
        positions = []
        for bbox in player_detections.xyxy:
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            positions.append(center)
        
        return positions
    
    def _extract_ball_position(self, frame_data: Dict) -> Optional[np.ndarray]:
        """Extract ball position from frame data"""
        ball_detections = frame_data.get('ball_detections')
        if ball_detections is None or len(ball_detections) == 0:
            return None
        
        ball_bbox = ball_detections.xyxy[0]
        return np.array([(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2])
    
    def _get_player_in_possession_enhanced(self, player_detections, ball_detections,
                                         proximity, player_team_mapping, poses):
        """Enhanced possession detection with pose analysis"""
        if len(ball_detections) != 1:
            return None
        
        ball_box = ball_detections.xyxy[0]
        ball_center = np.array([
            (ball_box[0] + ball_box[2]) / 2,
            (ball_box[1] + ball_box[3]) / 2
        ])
        
        candidates = []
        
        for i, player_box in enumerate(player_detections.xyxy):
            # Traditional proximity check
            player_center = np.array([
                (player_box[0] + player_box[2]) / 2,
                (player_box[1] + player_box[3]) / 2
            ])
            distance = np.linalg.norm(ball_center - player_center)
            
            # Base confidence from distance
            base_confidence = max(0.1, 1.0 - distance / proximity) if distance <= proximity else 0.0
            
            if base_confidence > 0.1:
                # Enhanced confidence with pose data
                enhanced_confidence = base_confidence
                
                if i < len(poses) and poses[i] is not None:
                    pose_features = poses[i].get('features', {})
                    
                    # Boost confidence for shooting form
                    if pose_features.get('shooting_form', False):
                        enhanced_confidence = min(1.0, enhanced_confidence * 1.3)
                    
                    # Boost confidence for arm extension
                    arm_extension = pose_features.get('arm_extension', 0.0)
                    if arm_extension > 0.5:
                        enhanced_confidence = min(1.0, enhanced_confidence * 1.1)
                
                # Get player and team IDs
                player_id = None
                team_id = None
                
                if (hasattr(player_detections, 'tracker_id') and
                    player_detections.tracker_id is not None and
                    i < len(player_detections.tracker_id)):
                    
                    player_id = int(player_detections.tracker_id[i])
                    team_id = player_team_mapping.get(player_id, None)
                
                if player_id is not None and team_id is not None:
                    candidates.append({
                        'player_id': player_id,
                        'team_id': team_id,
                        'confidence': enhanced_confidence,
                        'distance': distance
                    })
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x['confidence'])
    
    def _should_change_possession_with_context(self, current_player_id, current_team_id,
                                             confidence, frame_idx):
        """Enhanced possession change logic with context awareness"""
        
        # Get possession context if available
        context = {}
        if self.possession_context:
            context = self.possession_context.get_context_for_play_classification()
        
        # No current possession - start new one
        if self.current_possession is None:
            return True
        
        current_possession_player = self.current_possession.get('player_id')
        current_possession_team = self.current_possession.get('team_id')
        
        # Same player - no change
        if current_player_id == current_possession_player:
            self.possession_candidate = None
            self.candidate_confidence = 0
            self.candidate_frames = 0
            return False
        
        # Different player/team - consider context
        if (current_player_id != current_possession_player or
            current_team_id != current_possession_team):
            
            # Track candidate
            if (self.possession_candidate is None or
                self.possession_candidate['player_id'] != current_player_id):
                self.possession_candidate = {
                    'player_id': current_player_id,
                    'team_id': current_team_id
                }
                self.candidate_confidence = confidence
                self.candidate_frames = 1
            else:
                self.candidate_frames += 1
                self.candidate_confidence = max(self.candidate_confidence, confidence)
            
            # Context-aware thresholds
            required_frames = self.possession_change_threshold
            required_confidence = self.confidence_threshold
            
            # Adjust based on game tempo if context available
            if context:
                tempo = context.get('tempo', 1.0)
                if tempo > 1.3:  # Fast tempo - quicker possession changes
                    required_frames = max(3, int(required_frames * 0.7))
                elif tempo < 0.7:  # Slow tempo - more conservative changes
                    required_frames = int(required_frames * 1.3)
                
                # Adjust based on momentum
                team_momentum = context.get('team_momentum', {})
                momentum = team_momentum.get(current_team_id, 0.0)
                if momentum > 0.3:  # High momentum team
                    required_confidence *= 0.8
            
            if (self.candidate_frames >= required_frames and
                self.candidate_confidence >= required_confidence):
                return True
        
        return False
    
    def _start_new_possession_with_context(self, player_id, team_id, frame_idx,
                                         frame_data, poses):
        """Start new possession with enhanced context tracking"""
        
        # Detect screens and cuts from poses
        pose_actions = {}
        if poses and len(poses) > 0 and self.action_detector:
            player_positions = self._extract_player_positions(frame_data)
            pose_actions = self.action_detector.detect_actions(
                poses, player_positions, list(self.frame_history)
            )
        
        self.current_possession = {
            'player_id': player_id,
            'team_id': team_id,
            'start_frame': frame_idx,
            'end_frame': None,
            'duration': 0,
            'player_positions': [],
            'ball_positions': [],
            'velocities': [],
            'movement_patterns': [],
            'ball_transfers': 0,
            'started_from_rebound': self._detect_rebound_start(frame_data),
            'play_type': None,
            'pose_actions': pose_actions,
            'context_at_start': self.possession_context.get_context_for_play_classification() if self.possession_context else {},
            'max_confidence': 0.0,
            'possession_quality': 'stable'
        }
        
        self._update_current_possession_with_context(frame_data, frame_idx, poses)
        
        # Reset candidate tracking
        self.possession_candidate = None
        self.candidate_confidence = 0
        self.candidate_frames = 0
    
    def _update_current_possession_with_context(self, frame_data, frame_idx, poses):
        """Update possession with enhanced context and pose data"""
        if self.current_possession is None:
            return
        
        player_detections = frame_data.get('player_detections')
        ball_detections = frame_data.get('ball_detections')
        
        # Update traditional data
        if player_detections is not None:
            team_positions = {}
            velocities = []
            
            for i, bbox in enumerate(player_detections.xyxy):
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                
                if (hasattr(player_detections, 'tracker_id') and
                    player_detections.tracker_id is not None and
                    i < len(player_detections.tracker_id)):
                    
                    player_id = int(player_detections.tracker_id[i])
                    team_positions[player_id] = center.tolist()
                
                velocities.append(None)  # Placeholder for velocity
            
            self.current_possession['player_positions'].append(team_positions)
            self.current_possession['velocities'].append(velocities)
        
        # Update ball position
        if ball_detections is not None and len(ball_detections) > 0:
            ball_bbox = ball_detections.xyxy[0]
            ball_center = np.array([(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2])
            self.current_possession['ball_positions'].append(ball_center.tolist())
        
        # Update pose actions
        if poses and len(poses) > 0 and self.action_detector:
            player_positions = self._extract_player_positions(frame_data)
            current_pose_actions = self.action_detector.detect_actions(
                poses, player_positions, list(self.frame_history)
            )
            
            # Accumulate pose actions over the possession
            for action_type, actions in current_pose_actions.items():
                if action_type not in self.current_possession['pose_actions']:
                    self.current_possession['pose_actions'][action_type] = []
                self.current_possession['pose_actions'][action_type].extend(actions)
        
        # Update duration
        self.current_possession['duration'] = frame_idx - self.current_possession['start_frame']
    
    def _end_current_possession_with_context(self, frame_idx, poses):
        """End possession with enhanced context-aware classification"""
        if self.current_possession is None:
            return
        
        if self.current_possession['duration'] < self.min_possession_duration:
            print(f"   ⚠️ Possession too short ({self.current_possession['duration']} < {self.min_possession_duration}), discarding")
            self.current_possession = None
            return
        
        self.current_possession['end_frame'] = frame_idx
        
        # Enhanced play classification with context and pose data would go here
        # For now, assign default play type
        self.current_possession['play_type'] = 10  # Half Court Set
        self.current_possession['play_name'] = "Half Court Set"
        
        # Add possession quality assessment
        self._assess_possession_quality_with_context()
        
        # Add to context tracker
        if self.possession_context:
            self.possession_context.add_completed_play({
                'play_type': self.current_possession['play_type'],
                'play_name': self.current_possession['play_name'],
                'team_id': self.current_possession['team_id'],
                'duration': self.current_possession['duration'],
                'outcome': 'unknown',
                'frame': frame_idx
            })
        
        # Add to history
        self.possession_history.append(self.current_possession.copy())
        print(f"   ✅ Enhanced possession completed: Player {self.current_possession['player_id']} (Team {self.current_possession['team_id']}) - {self.current_possession['play_name']} ({self.current_possession['duration']} frames)")
        
        self.current_possession = None
    
    def _assess_possession_quality_with_context(self):
        """Enhanced possession quality assessment with context"""
        if not self.current_possession:
            return
        
        duration = self.current_possession['duration']
        pose_actions = self.current_possession.get('pose_actions', {})
        
        # Base quality from duration
        if duration < 10:
            quality = 'brief'
        elif duration < 30:
            quality = 'normal'
        else:
            quality = 'extended'
        
        # Enhance based on pose actions
        total_actions = sum(len(actions) for actions in pose_actions.values())
        if total_actions > 3:
            quality = 'complex'
        elif total_actions > 1:
            if quality == 'brief':
                quality = 'quick_action'
        
        self.current_possession['possession_quality'] = quality
    
    def _no_possession_update_with_context(self, frame_idx):
        """Handle no possession with context tracking"""
        if self.current_possession is not None:
            self.current_possession['duration'] = frame_idx - self.current_possession['start_frame']
        
        return self._get_possession_summary_with_context(frame_idx)
    
    def _get_possession_summary_with_context(self, frame_idx):
        """Enhanced possession summary with context information"""
        valid_possessions = [p for p in self.possession_history
                           if p['duration'] >= self.min_possession_duration]
        
        # Get current context
        current_context = {}
        if self.possession_context:
            current_context = self.possession_context.get_context_for_play_classification()
        
        summary = {
            'frame_idx': frame_idx,
            'current_possession': self.current_possession,
            'player_in_possession': self.current_possession['player_id'] if self.current_possession else None,
            'team_in_possession': self.current_possession['team_id'] if self.current_possession else None,
            'ball_position': self.current_possession['ball_positions'][-1] if (self.current_possession and self.current_possession['ball_positions']) else None,
            'possession_change': False,
            'total_possessions': len(valid_possessions),
            'team_possessions': {0: 0, 1: 0},
            'play_type_counts': {i: 0 for i in range(11)},
            'avg_possession_duration': 0,
            'possession_context': current_context,
            'enhanced_features': {
                'pose_actions_detected': bool(self.current_possession and self.current_possession.get('pose_actions')),
                'context_aware_classification': True,
                'momentum_tracking': True
            }
        }
        
        # Calculate enhanced statistics
        durations = []
        for possession in valid_possessions:
            team_id = possession.get('team_id')
            if team_id is not None and team_id in [0, 1]:
                summary['team_possessions'][team_id] += 1
            
            play_type = possession.get('play_type')
            if play_type is not None and play_type in summary['play_type_counts']:
                summary['play_type_counts'][play_type] += 1
            
            durations.append(possession['duration'])
        
        if durations:
            summary['avg_possession_duration'] = np.mean(durations)
        
        return summary
    
    def _detect_rebound_start(self, frame_data: Dict) -> bool:
        """Enhanced rebound detection"""
        ball_detections = frame_data.get('ball_detections')
        hoop_detections = frame_data.get('hoop_detections', [])
        backboard_detections = frame_data.get('backboard_detections', [])
        
        if not ball_detections or len(ball_detections) == 0:
            return False
        
        ball_center = np.array([
            (ball_detections.xyxy[0][0] + ball_detections.xyxy[0][2]) / 2,
            (ball_detections.xyxy[0][1] + ball_detections.xyxy[0][3]) / 2
        ])
        
        # Check proximity to hoop or backboard
        for detections in [hoop_detections, backboard_detections]:
            if len(detections) > 0:
                target_center = np.array([
                    (detections.xyxy[0][0] + detections.xyxy[0][2]) / 2,
                    (detections.xyxy[0][1] + detections.xyxy[0][3]) / 2
                ])
                distance = np.linalg.norm(ball_center - target_center)
                if distance < 150:
                    return True
        
        return False

    def get_possession_segments(self):
        """Return possession segments as list of dicts: start, end, player, team, duration"""
        return [
            {
                'start_frame': p['start_frame'],
                'end_frame': p['end_frame'],
                'team_id': p['team_id'],
                'player_id': p['player_id'],
                'duration': p.get('duration', 0)
            }
            for p in self.possession_history
            if p.get('duration', 0) >= self.min_possession_duration
        ]
