"""
Single frame processing pipeline with basketball intelligence
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import logging

from core import Detection, Track, PossessionInfo, PlayEvent, BasketballAction
from core.constants import PLAYER_ID, BALL_ID, REF_ID, HOOP_ID, BACKBOARD_ID
from team_identification import AdvancedTeamClassificationManager
from analytics.possession import EnhancedPossessionTracker

class FrameProcessor:
    """Process individual frames with basketball analytics"""
    
    def __init__(self, 
                 detector,
                 mask_generator,
                 tracker,
                 team_classifier,
                 possession_tracker,
                 event_detector,
                 pose_estimator,
                 frame_annotator,
                 basketball_enhanced: bool = True):
        """
        Initialize frame processor with basketball components
        
        Args:
            detector: Object detector
            mask_generator: Mask generator
            tracker: Enhanced basketball tracker
            team_classifier: Team classifier with 5v5 balancing
            possession_tracker: Basketball possession tracker
            event_detector: Basketball event detector
            pose_estimator: Basketball pose estimator
            frame_annotator: Basketball-aware frame annotator
            basketball_enhanced: Enable basketball-specific processing
        """
        self.detector = detector
        self.mask_generator = mask_generator
        self.tracker = tracker
        self.team_classifier = tracker.team_classification_manager if hasattr(tracker, 'team_classification_manager') else team_classifier
        self.possession_tracker = tracker.enhanced_possession_tracker if hasattr(tracker, 'enhanced_possession_tracker') else possession_tracker
        self.event_detector = event_detector
        self.pose_estimator = pose_estimator
        self.frame_annotator = frame_annotator
        self.basketball_enhanced = basketball_enhanced
        
        self.logger = logging.getLogger(__name__)
        
        # Basketball-specific state
        self.basketball_stats = {
            'frames_processed': 0,
            'team_classifications': 0,
            'possession_changes': 0,
            'basketball_events': 0,
            'pose_actions_detected': 0
        }
        
    def process_basketball_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame with basketball intelligence
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            
        Returns:
            Tuple of (annotated_frame, basketball_analytics_data)
        """
        start_time = time.time()
        
        # Initialize basketball analytics
        analytics_data = {
            'frame_idx': frame_idx,
            'detections': {},
            'tracks': [],
            'team_assignments': {},
            'possession': None,
            'events': [],
            'basketball_actions': [],
            'team_stats': {},
            'basketball_metadata': {
                'processing_time': 0.0,
                'basketball_enhanced': self.basketball_enhanced
            }
        }
        
        try:
            # Step 1: Basketball-optimized object detection
            detections = self.detector.detect(frame)
            if not detections:
                return frame, analytics_data
            
            # Step 2: Generate basketball-optimized masks
            if detections:
                if hasattr(self.mask_generator, 'generate_basketball_masks'):
                    masks = self.mask_generator.generate_basketball_masks(frame, detections)
                else:
                    masks = self.mask_generator.generate_masks(frame, detections)
                    
                for det, mask in zip(detections, masks):
                    det.mask = mask
                    
            # Step 3: Separate basketball detections
            detections_by_class = self._separate_basketball_detections(detections)
            analytics_data['detections'] = detections_by_class
            
            # Step 4: Basketball player tracking with team classification
            player_detections = detections_by_class.get('players', [])
            if player_detections:
                # Enhanced basketball tracking
                if hasattr(self.tracker, 'update_basketball'):
                    tracks = self.tracker.update_basketball(player_detections, frame, frame_idx)
                else:
                    tracks = self.tracker.update(player_detections, frame)
                    
                analytics_data['tracks'] = tracks
                
                # Basketball team classification with 5v5 balancing
                if self.team_classifier and tracks:
                    if not self.team_classifier.is_initialized():
                        # Collect crops for team classifier training
                        crops = self._extract_basketball_crops(frame, tracks)
                        if len(crops) >= 6:  # Minimum for basketball team training
                            self.team_classifier.fit(crops)
                            
                    if self.team_classifier.is_initialized():
                        if hasattr(self.team_classifier, 'classify_basketball_teams'):
                            team_assignments = self.team_classifier.classify_basketball_teams(
                                tracks, frame, enforce_5v5=True
                            )
                        else:
                            team_assignments = self.team_classifier.classify(tracks, frame)
                            
                        # Apply team assignments to tracks
                        for track in tracks:
                            if track.id in team_assignments:
                                track.team_id = team_assignments[track.id]
                                
                        analytics_data['team_assignments'] = team_assignments
                        self.basketball_stats['team_classifications'] += 1
                        
                        # Generate team statistics
                        analytics_data['team_stats'] = self._generate_basketball_team_stats(tracks)
                        
            # Step 5: Basketball possession tracking with context
            if self.possession_tracker and analytics_data['tracks']:
                ball_track = self._get_basketball_ball_track(detections_by_class.get('ball', []))
                
                if hasattr(self.possession_tracker, 'update_basketball_possession'):
                    possession_info = self.possession_tracker.update_basketball_possession(
                        analytics_data['tracks'], ball_track, frame_idx, context_enabled=True
                    )
                else:
                    possession_info = self.possession_tracker.update_possession(
                        analytics_data['tracks'], ball_track, frame_idx
                    )
                    
                analytics_data['possession'] = possession_info
                
                if possession_info and possession_info.possession_change:
                    self.basketball_stats['possession_changes'] += 1

                    play_result = self.possession_tracker.play_classifier.classify_play(
                        possession_data=possession_info.to_dict(),
                        context={}
                    )
                    possession_info.play = play_result
                    
            # Step 6: Basketball event detection
            if self.event_detector and analytics_data['possession']:
                if hasattr(self.event_detector, 'detect_basketball_events'):
                    events = self.event_detector.detect_basketball_events(
                        detections_by_class,
                        analytics_data['possession'],
                        frame_idx
                    )
                else:
                    events = self.event_detector.detect_events(
                        detections_by_class,
                        analytics_data['possession'],
                        frame_idx
                    )
                    
                analytics_data['events'] = events
                if events:
                    self.basketball_stats['basketball_events'] += len(events)
                    
            # Step 7: Basketball pose estimation and action detection
            if self.pose_estimator and analytics_data['tracks']:
                crops = self._extract_basketball_crops(frame, analytics_data['tracks'])
                
                if hasattr(self.pose_estimator, 'extract_basketball_poses'):
                    poses = self.pose_estimator.extract_basketball_poses(crops)
                else:
                    poses = self.pose_estimator.extract_poses(crops)
                    
                analytics_data['poses'] = poses

                # Attach pose to each track
                for track, pose in zip(analytics_data['tracks'], poses):
                    track.pose = pose  # Store as flat list of keypoints

                # Detect basketball actions (screens, cuts, etc.)
                if poses and hasattr(self.pose_estimator, 'detect_basketball_actions'):
                    positions = [track.current_position for track in analytics_data['tracks']]
                    basketball_actions = self.pose_estimator.detect_basketball_actions(
                        poses, positions, frame_idx
                    )
                    analytics_data['basketball_actions'] = basketball_actions
                    
                    if basketball_actions:
                        self.basketball_stats['pose_actions_detected'] += len(basketball_actions)
                        
            # Step 8: Basketball-aware frame annotation
            annotated_frame = frame
            if self.frame_annotator:
                # Prepare basketball context for visualization
                basketball_context = {
                    'possession': analytics_data['possession'],
                    'team_stats': analytics_data['team_stats'],
                    'events': analytics_data['events'],
                    'basketball_actions': analytics_data.get('basketball_actions', []),
                    'frame_stats': self._generate_frame_stats(analytics_data)
                }
                
                if hasattr(self.frame_annotator, 'annotate_basketball_frame'):
                    annotated_frame = self.frame_annotator.annotate_basketball_frame(
                        frame,
                        detections,
                        analytics_data['tracks'],
                        analytics_data,
                        basketball_context
                    )
                else:
                    annotated_frame = self.frame_annotator.annotate(
                        frame,
                        detections,
                        analytics_data['tracks'],
                        analytics_data
                    )
                    
        except Exception as e:
            self.logger.error(f"Basketball frame processing error at frame {frame_idx}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            annotated_frame = frame
            
        # Update processing metadata
        processing_time = time.time() - start_time
        analytics_data['basketball_metadata']['processing_time'] = processing_time
        self.basketball_stats['frames_processed'] += 1
        
        return annotated_frame, analytics_data
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """Legacy method - calls basketball frame processing"""
        return self.process_basketball_frame(frame, frame_idx)
        
    def _separate_basketball_detections(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Separate detections by basketball class"""
        separated = {
            'players': [],
            'ball': [],
            'referees': [],
            'hoop': [],
            'backboard': [],
            'court_elements': [],
            'other': []
        }
        
        for det in detections:
            if det.class_id == PLAYER_ID:
                separated['players'].append(det)
            elif det.class_id == BALL_ID:
                separated['ball'].append(det)
            elif det.class_id == REF_ID:
                separated['referees'].append(det)
            elif det.class_id == HOOP_ID:
                separated['hoop'].append(det)
            elif det.class_id == BACKBOARD_ID:
                separated['backboard'].append(det)
            elif det.class_id in [2, 4, 7, 8]:  # Court elements (gameclock, period, scoreboard, team_points)
                separated['court_elements'].append(det)
            else:
                separated['other'].append(det)
                
        return separated
        
    def _get_basketball_ball_track(self, ball_detections: List[Detection]) -> Optional[Track]:
        """Create basketball ball track from detections"""
        if not ball_detections:
            return None
            
        # Use most confident ball detection
        best_ball = max(ball_detections, key=lambda x: x.confidence)
        
        return Track(
            id=-1,  # Special ID for ball
            detections=[best_ball],
            state='tracked',
            confidence=best_ball.confidence
        )
        
    def _extract_basketball_crops(self, frame: np.ndarray, tracks: List[Track]) -> List[np.ndarray]:
        """Extract basketball-optimized crops (jersey region focus)"""
        crops = []
        
        for track in tracks:
            if track.current_bbox is not None:
                x1, y1, x2, y2 = track.current_bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    
                    # Extract jersey region (upper 60% for better team classification)
                    h = crop.shape[0]
                    jersey_height = int(h * 0.6)
                    if jersey_height > 0:
                        jersey_crop = crop[:jersey_height, :]
                        crops.append(jersey_crop)
                    else:
                        crops.append(crop)
                else:
                    crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
            else:
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
                
        return crops

    def _extract_basketball_crops(self, frame: np.ndarray, tracks: List[Track]) -> List[np.ndarray]:
        """Extract jersey crops using pose keypoints if available"""
        crops = []
    
        for track in tracks:
            # Try using pose first (more accurate)
            pose = getattr(track, "pose", None)
            if pose and isinstance(pose, list) and len(pose) >= 26:  # 13 keypoints min
                crop = self._extract_jersey_crop_from_pose(frame, pose)
                if crop is not None:
                    crops.append(crop)
                    continue
    
            # Fallback to bbox (if no pose or failure)
            if track.current_bbox is not None:
                x1, y1, x2, y2 = track.current_bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    fallback_crop = frame[y1:y2, x1:x2]
                    h = fallback_crop.shape[0]
                    jersey_crop = fallback_crop[:int(h * 0.6), :] if h > 0 else fallback_crop
                    crops.append(jersey_crop)
                    continue
    
            # Final fallback if bbox is invalid
            crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
    
        return crops

    def _extract_jersey_crop_from_pose(self, frame: np.ndarray, pose: List[float]) -> Optional[np.ndarray]:
        """
        Extract torso region using pose keypoints.
        pose: flat list [x0, y0, x1, y1, ..., xk, yk]
        """
        try:
            # OpenPose/MMPose format assumes:
            # 5 = left_shoulder, 6 = right_shoulder
            # 11 = left_hip, 12 = right_hip
            ls = pose[5*2:5*2+2]
            rs = pose[6*2:6*2+2]
            lh = pose[11*2:11*2+2]
            rh = pose[12*2:12*2+2]
    
            x_coords = [ls[0], rs[0], lh[0], rh[0]]
            y_coords = [ls[1], rs[1], lh[1], rh[1]]
    
            x_min = int(max(0, min(x_coords)))
            x_max = int(min(frame.shape[1], max(x_coords)))
            y_min = int(max(0, min(y_coords)))
            y_max = int(min(frame.shape[0], max(y_coords)))
    
            if x_max > x_min and y_max > y_min:
                return frame[y_min:y_max, x_min:x_max]
    
        except Exception as e:
            self.logger.warning(f"Pose jersey crop failed: {e}")
    
        return None

    
    def _generate_basketball_team_stats(self, tracks: List[Track]) -> Dict:
        """Generate basketball team statistics"""
        team_stats = {
            'team_distribution': {0: 0, 1: 0, None: 0},
            'total_players': len(tracks),
            'team_balance_score': 0.0,
            'is_basketball_balanced': False
        }
        
        for track in tracks:
            team_id = track.team_id
            if team_id in [0, 1]:
                team_stats['team_distribution'][team_id] += 1
            else:
                team_stats['team_distribution'][None] += 1
                
        # Basketball balance analysis
        team_0_count = team_stats['team_distribution'][0]
        team_1_count = team_stats['team_distribution'][1]
        
        if team_0_count + team_1_count > 0:
            balance_diff = abs(team_0_count - team_1_count)
            team_stats['team_balance_score'] = 1.0 - (balance_diff / (team_0_count + team_1_count))
            
        # Basketball 5v5 validation
        team_stats['is_basketball_balanced'] = (4 <= team_0_count <= 6 and 4 <= team_1_count <= 6)
        
        return team_stats
        
    def _generate_frame_stats(self, analytics_data: Dict) -> Dict:
        """Generate frame-level statistics"""
        tracks = analytics_data.get('tracks', [])
        team_assignments = analytics_data.get('team_assignments', {})
        
        return {
            'frame_idx': analytics_data['frame_idx'],
            'total_detections': sum(len(dets) for dets in analytics_data['detections'].values()),
            'tracked_players': len(tracks),
            'team_assignments_count': len(team_assignments),
            'player_ids': [track.id for track in tracks],
            'team_distribution': analytics_data.get('team_stats', {}).get('team_distribution', {}),
            'avg_confidence': np.mean([track.confidence for track in tracks]) if tracks else 0.0,
            'possession_active': analytics_data.get('possession') is not None,
            'events_detected': len(analytics_data.get('events', [])),
            'basketball_actions_detected': len(analytics_data.get('basketball_actions', []))
        }
        
    def get_basketball_statistics(self) -> Dict:
        """Get basketball processing statistics"""
        return self.basketball_stats.copy()
        
    def reset_basketball_statistics(self):
        """Reset basketball processing statistics"""
        self.basketball_stats = {
            'frames_processed': 0,
            'team_classifications': 0,
            'possession_changes': 0,
            'basketball_events': 0,
            'pose_actions_detected': 0
        }
