"""
Main video processing pipeline with basketball-specific analysis
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator
import numpy as np
import cv2

from core import AnalysisResult, Detection, Track, PlayEvent
from detection import YoloDetector, MaskGenerator, BatchProcessor
from tracking import EnhancedTracker
from team_identification import UnifiedTeamClassifier
from analytics import PossessionTracker, PlayClassifier, EventDetector, TimelineGenerator
from analytics.pose import PoseEstimator, ActionDetector
from visualization import FrameAnnotator
from io import VideoReader, VideoWriter, StreamingWriter
from utils import MemoryMonitor, cleanup_resources

# Basketball-specific constants from original code
PLAYER_ID = 5
REF_ID = 6
BALL_ID = 1
BACKBOARD_ID = 0
HOOP_ID = 3


@dataclass
class ProcessorConfig:
    """Configuration for video processor"""
    # Model paths
    yolo_model_path: str
    sam_checkpoint_path: Optional[str] = None

    # Processing parameters
    batch_size: int = 20
    start_frame: int = 0
    end_frame: Optional[int] = None

    # Detection parameters
    detection_confidence: float = 0.2
    nms_threshold: float = 0.5

    # Tracking parameters
    track_activation_threshold: float = 0.4
    lost_track_buffer: int = 90
    min_matching_threshold: float = 0.2
    min_consecutive_frames: int = 3
    max_tracks: int = 15

    # Basketball team classification - ENHANCED FROM ORIGINAL
    use_ml_classification: bool = True
    use_color_fallback: bool = True
    enforce_basketball_rules: bool = True
    basketball_5v5_balancing: bool = True  # NEW: From original enhanced tracker

    # Analytics
    enable_possession_tracking: bool = True
    enable_play_classification: bool = True
    enable_pose_estimation: bool = True
    enable_event_detection: bool = True
    enable_context_tracking: bool = True  # NEW: From enhanced play analysis

    # Visualization
    enable_visualization: bool = True
    use_ellipse_annotation: bool = True
    team_aware_colors: bool = True  # NEW: From visualization module

    # Output
    output_video_path: Optional[str] = None
    save_analytics: bool = True
    analytics_output_path: Optional[str] = None
    streaming_output: bool = False

    # Performance
    use_gpu: bool = True
    memory_optimization: bool = True


class VideoProcessor:
    """Main video processing pipeline with basketball intelligence"""

    def __init__(self, config: ProcessorConfig):
        """Initialize video processor with basketball-specific components"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._initialize_components()

        # Memory monitoring
        self.memory_monitor = MemoryMonitor() if config.memory_optimization else None

        # Basketball-specific state tracking
        self.basketball_stats = {
            'team_corrections_applied': 0,
            'possessions_tracked': 0,
            'play_classifications': 0,
            'frames_with_teams': 0
        }

    def _initialize_components(self):
        """Initialize all processing components with basketball enhancements"""
        device = 'cuda' if self.config.use_gpu else 'cpu'

        # Detection with basketball-optimized parameters
        self.detector = YoloDetector(
            model_path=self.config.yolo_model_path,
            device=device,
            confidence=self.config.detection_confidence,
            nms_threshold=self.config.nms_threshold
        )

        # Enhanced mask generation for basketball (from original EdgeTAM integration)
        if self.config.sam_checkpoint_path:
            self.mask_generator = MaskGenerator(
                checkpoint_path=self.config.sam_checkpoint_path,
                basketball_optimized=True  # NEW: Basketball-specific optimization
            )
        else:
            self.mask_generator = MaskGenerator(basketball_fallback=True)

        # Batch processor with memory optimization
        self.batch_processor = BatchProcessor(
            batch_size=self.config.batch_size,
            memory_efficient=self.config.memory_optimization
        )

        # Enhanced basketball tracking (from original EnhancedEdgeTAMTracker)
        self.tracker = EnhancedTracker(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.min_matching_threshold,
            minimum_consecutive_frames=self.config.min_consecutive_frames,
            max_tracks=self.config.max_tracks,
            use_basketball_logic=True,  # NEW: Enable basketball-specific tracking
            enable_team_identification=True
        )

        # Unified team classification with basketball intelligence
        self.team_classifier = UnifiedTeamClassifier(
            use_ml=self.config.use_ml_classification,
            use_color_fallback=self.config.use_color_fallback,
            enforce_basketball_rules=self.config.enforce_basketball_rules,
            basketball_5v5_balancing=self.config.basketball_5v5_balancing,
            device=device
        )

        # Enhanced basketball analytics
        if self.config.enable_possession_tracking:
            self.possession_tracker = PossessionTracker(
                basketball_enhanced=True,  # NEW: Basketball-specific possession logic
                context_tracking=self.config.enable_context_tracking
            )
        else:
            self.possession_tracker = None

        if self.config.enable_play_classification:
            self.play_classifier = PlayClassifier(
                basketball_play_types=True,  # NEW: 11 basketball play types
                context_aware=self.config.enable_context_tracking
            )
        else:
            self.play_classifier = None

        if self.config.enable_event_detection:
            self.event_detector = EventDetector(
                basketball_events=True  # NEW: Basketball-specific events
            )
        else:
            self.event_detector = None

        if self.config.enable_pose_estimation:
            self.pose_estimator = PoseEstimator(
                basketball_actions=True  # NEW: Basketball pose actions
            )
            self.action_detector = ActionDetector(
                detect_screens=True,
                detect_cuts=True
            )
        else:
            self.pose_estimator = None
            self.action_detector = None

        # Timeline generator with basketball context
        self.timeline_generator = TimelineGenerator(
            basketball_timeline=True
        )

        # Basketball-aware visualization
        if self.config.enable_visualization:
            self.frame_annotator = FrameAnnotator(
                use_ellipse=self.config.use_ellipse_annotation,
                team_aware_colors=self.config.team_aware_colors,
                basketball_overlays=True  # NEW: Basketball-specific overlays
            )
        else:
            self.frame_annotator = None

    def process_video(self, video_path: str,
                     progress_callback: Optional[callable] = None) -> AnalysisResult:
        """Process video with basketball analysis"""
        self.logger.info(f"Starting basketball video analysis: {video_path}")

        # Start memory monitoring
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()

        start_time = time.time()

        # Initialize video reader
        video_reader = VideoReader(video_path)
        video_info = video_reader.get_info()

        # Determine frame range
        start_frame = self.config.start_frame
        end_frame = self.config.end_frame or video_info['total_frames']
        total_frames = end_frame - start_frame

        # Initialize video writer with basketball annotations
        video_writer = None
        if self.config.output_video_path and self.config.enable_visualization:
            video_writer = VideoWriter(
                self.config.output_video_path,
                fps=video_info['fps'],
                width=video_info['width'],
                height=video_info['height'],
                basketball_codec=True  # NEW: Optimized for basketball videos
            )

        # Initialize streaming writer
        streaming_writer = None
        if self.config.streaming_output:
            streaming_writer = StreamingWriter(
                self.config.analytics_output_path or "basketball_analytics.pkl"
            )

        # Process video
        try:
            results = self._process_basketball_frames(
                video_reader,
                video_writer,
                streaming_writer,
                start_frame,
                end_frame,
                progress_callback
            )

            # Create basketball analysis result
            processing_time = time.time() - start_time
            analysis_result = self._create_basketball_analysis_result(
                video_path, video_info, results, processing_time
            )

            # Save basketball analytics
            if self.config.save_analytics:
                self._save_basketball_analytics(analysis_result)

            return analysis_result

        finally:
            # Cleanup
            video_reader.close()
            if video_writer:
                video_writer.close()
            if streaming_writer:
                streaming_writer.close()
            if self.memory_monitor:
                memory_stats = self.memory_monitor.get_summary()
                self.logger.info(f"Basketball analysis memory usage: {memory_stats}")
            cleanup_resources()

    def _process_basketball_frames(self,
                                  video_reader: VideoReader,
                                  video_writer: Optional[VideoWriter],
                                  streaming_writer: Optional[StreamingWriter],
                                  start_frame: int,
                                  end_frame: int,
                                  progress_callback: Optional[callable]) -> Dict:
        """Process frames with basketball-specific analysis"""
        results = {
            'tracks': [],
            'teams': [],
            'possessions': [],
            'plays': [],
            'events': [],
            'frames_processed': 0,
            'basketball_stats': self.basketball_stats.copy()
        }

        # Process in batches for memory efficiency
        frame_generator = video_reader.read_frames(start_frame, end_frame)

        for batch_data in self.batch_processor.create_batches(
            frame_generator, self.config.batch_size
        ):
            frames, frame_indices = zip(*batch_data)

            # Basketball-optimized YOLO detection
            batch_detections = self.detector.detect_batch(
                list(frames), 
                basketball_optimized=True
            )

            # Process each frame with basketball logic
            for frame, frame_idx, detections in zip(frames, frame_indices, batch_detections):
                # Generate basketball-optimized masks
                masks = self.mask_generator.generate_basketball_masks(frame, detections)
                for det, mask in zip(detections, masks):
                    det.mask = mask

                # Separate basketball-specific detections
                player_detections = [d for d in detections if d.class_id == PLAYER_ID]
                ball_detections = [d for d in detections if d.class_id == BALL_ID]
                ref_detections = [d for d in detections if d.class_id == REF_ID]
                hoop_detections = [d for d in detections if d.class_id == HOOP_ID]
                backboard_detections = [d for d in detections if d.class_id == BACKBOARD_ID]

                # Basketball-enhanced tracking
                tracks = self.tracker.update_basketball(player_detections, frame, frame_idx)

                # Basketball team classification with 5v5 balancing
                if tracks and not self.team_classifier.is_initialized():
                    crops = [self._extract_basketball_crop(frame, t.current_bbox) for t in tracks]
                    self.team_classifier.fit([c for c in crops if c is not None])

                if self.team_classifier.is_initialized():
                    team_assignments = self.team_classifier.classify_basketball_teams(
                        tracks, frame, enforce_5v5=self.config.basketball_5v5_balancing
                    )
                    
                    for track in tracks:
                        if track.id in team_assignments:
                            track.team_id = team_assignments[track.id]
                            
                    # Track team classification success
                    if team_assignments:
                        self.basketball_stats['frames_with_teams'] += 1

                # Basketball analytics update
                frame_results = self._update_basketball_analytics(
                    frame, frame_idx, tracks, ball_detections, 
                    hoop_detections, backboard_detections, detections
                )

                # Basketball-aware visualization
                annotated_frame = frame
                if self.config.enable_visualization and self.frame_annotator:
                    basketball_context = {
                        'possession': frame_results.get('possession'),
                        'team_stats': self.basketball_stats,
                        'play_events': frame_results.get('events', [])
                    }
                    
                    annotated_frame = self.frame_annotator.annotate_basketball_frame(
                        frame, detections, tracks, frame_results, basketball_context
                    )

                # Write output
                if video_writer:
                    video_writer.write_frame(annotated_frame)

                # Stream basketball results
                if streaming_writer:
                    basketball_frame_data = {
                        'frame_idx': frame_idx,
                        'tracks': tracks,
                        'team_assignments': {t.id: t.team_id for t in tracks if t.team_id is not None},
                        'possession': frame_results.get('possession'),
                        'basketball_analytics': frame_results
                    }
                    streaming_writer.write(basketball_frame_data)

                results['frames_processed'] += 1

                # Progress callback
                if progress_callback:
                    progress = results['frames_processed'] / (end_frame - start_frame)
                    progress_callback(progress)

            # Basketball-specific periodic cleanup
            if results['frames_processed'] % 100 == 0:
                cleanup_resources()
                self._log_basketball_progress(results['frames_processed'])

        # Finalize basketball results
        results['tracks'] = self.tracker.get_all_tracks()
        results['teams'] = self._extract_basketball_teams(results['tracks'])
        
        # Get possession and play summaries
        if self.possession_tracker:
            results['possessions'] = self.possession_tracker.get_all_possessions()
            self.basketball_stats['possessions_tracked'] = len(results['possessions'])
            
        if self.play_classifier:
            results['plays'] = self.play_classifier.get_classified_plays()
            self.basketball_stats['play_classifications'] = len(results['plays'])

        results['basketball_stats'] = self.basketball_stats

        return results

    def _update_basketball_analytics(self, frame: np.ndarray, frame_idx: int,
                                   tracks: List[Track], ball_detections: List[Detection],
                                   hoop_detections: List[Detection], 
                                   backboard_detections: List[Detection],
                                   all_detections: List[Detection]) -> Dict:
        """Update basketball-specific analytics"""
        frame_results = {}

        # Find basketball ball track
        ball_track = None
        if ball_detections:
            ball_track = Track(
                id=-1,  # Special ID for ball
                detections=ball_detections[:1],
                state='tracked'
            )

        # Enhanced basketball possession tracking
        if self.possession_tracker:
            possession_info = self.possession_tracker.update_basketball_possession(
                tracks, ball_track, frame_idx, 
                context_enabled=self.config.enable_context_tracking
            )
            frame_results['possession'] = possession_info
            self.timeline_generator.add_basketball_possession_change(possession_info)

        # Basketball pose estimation and action detection
        if self.pose_estimator and tracks:
            crops = [self._extract_basketball_crop(frame, t.current_bbox) for t in tracks]
            poses = self.pose_estimator.extract_basketball_poses(crops)

            if self.action_detector:
                positions = [t.current_position for t in tracks]
                basketball_actions = self.action_detector.detect_basketball_actions(
                    poses, positions, detect_screens=True, detect_cuts=True
                )
                frame_results['basketball_actions'] = basketball_actions

        # Basketball event detection (shots, rebounds, etc.)
        if self.event_detector:
            basketball_detection_dict = {
                'ball': ball_detections,
                'hoop': hoop_detections,
                'backboard': backboard_detections,
                'players': [d for d in all_detections if d.class_id == PLAYER_ID]
            }

            basketball_events = self.event_detector.detect_basketball_events(
                basketball_detection_dict,
                frame_results.get('possession'),
                frame_idx
            )

            for event in basketball_events:
                self.timeline_generator.add_basketball_event(event)
            frame_results['events'] = basketball_events

        # Basketball play classification with context
        if self.play_classifier and self.possession_tracker:
            if hasattr(self.possession_tracker, 'last_completed_possession'):
                possession_data = self.possession_tracker.last_completed_possession
                basketball_context = self.possession_tracker.get_basketball_context()

                basketball_play = self.play_classifier.classify_basketball_play(
                    possession_data, basketball_context,
                    pose_data=frame_results.get('basketball_actions')
                )
                
                if basketball_play:
                    self.timeline_generator.add_basketball_play_classification(basketball_play)
                    frame_results['play'] = basketball_play

        return frame_results

    def _extract_basketball_crop(self, frame: np.ndarray, bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extract basketball-optimized crop (jersey region focus)"""
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            
            # Extract jersey region (upper portion) for better team classification
            h = crop.shape[0]
            jersey_height = int(h * 0.6)  # Focus on upper 60% for jersey
            if jersey_height > 0:
                return crop[:jersey_height, :]
            return crop
        return None

    def _extract_basketball_teams(self, tracks: List[Track]) -> List[Dict]:
        """Extract basketball team information with 5v5 validation"""
        teams = {0: {'id': 0, 'players': [], 'name': 'Team 0'}, 
                 1: {'id': 1, 'players': [], 'name': 'Team 1'}}

        for track in tracks:
            if track.team_id in teams:
                teams[track.team_id]['players'].append(track.id)

        # Basketball validation: warn if teams are heavily imbalanced
        team0_count = len(teams[0]['players'])
        team1_count = len(teams[1]['players'])
        
        if abs(team0_count - team1_count) > 3:
            self.logger.warning(f"Basketball team imbalance detected: T0={team0_count}, T1={team1_count}")

        return list(teams.values())

    def _create_basketball_analysis_result(self, video_path: str,
                                         video_info: Dict,
                                         results: Dict,
                                         processing_time: float) -> AnalysisResult:
        """Create basketball-enhanced analysis result"""
        timeline_data = self.timeline_generator.generate_basketball_timeline()

        return AnalysisResult(
            video_path=video_path,
            fps=video_info['fps'],
            total_frames=video_info['total_frames'],
            processed_frames=results['frames_processed'],
            tracks=results['tracks'],
            teams=results['teams'],
            possessions=timeline_data['summary'].get('possessions', []),
            plays=timeline_data['summary'].get('plays', []),
            events=timeline_data['events'],
            team_stats=timeline_data['summary'].get('team_statistics', {}),
            processing_time=processing_time,
            pose_estimation_enabled=self.config.enable_pose_estimation,
            context_tracking_enabled=self.config.enable_context_tracking,
            # Basketball-specific metadata
            basketball_enhanced=True,
            team_classification_stats=self.team_classifier.get_statistics() if self.team_classifier else {},
            basketball_analysis_stats=self.basketball_stats
        )

    def _log_basketball_progress(self, frames_processed: int):
        """Log basketball analysis progress"""
        self.logger.info(f"Basketball analysis progress: {frames_processed} frames processed")
        self.logger.info(f"Teams identified in {self.basketball_stats['frames_with_teams']} frames")
        if self.basketball_stats['possessions_tracked'] > 0:
            self.logger.info(f"Possessions tracked: {self.basketball_stats['possessions_tracked']}")

    def _save_basketball_analytics(self, analysis_result: AnalysisResult):
        """Save basketball analytics to file"""
        output_path = self.config.analytics_output_path or "basketball_analytics.json"

        import json
        with open(output_path, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)

        self.logger.info(f"Basketball analytics saved to: {output_path}")
