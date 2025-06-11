"""
Main video processing pipeline synchronized with FrameProcessor and BatchOptimizer
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
from pipeline.frame_processor import FrameProcessor
from pipeline.batch_optimizer import BatchOptimizer

# Basketball-specific constants from original code
PLAYER_ID = 5
REF_ID = 6
BALL_ID = 1
BACKBOARD_ID = 0
HOOP_ID = 3


@dataclass
class ProcessorConfig:
    """Configuration for video processor - SYNCHRONIZED"""
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
    adaptive_batching: bool = True  # NEW: Use adaptive batch optimization


class VideoProcessor:
    """Main video processing pipeline with synchronized components"""

    def __init__(self, config: ProcessorConfig):
        """Initialize video processor with synchronized basketball components"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components in dependency order
        self._initialize_components()

        # Initialize synchronized processors
        self._initialize_synchronized_processors()

        # Memory monitoring
        self.memory_monitor = MemoryMonitor() if config.memory_optimization else None

        # Basketball-specific state tracking
        self.basketball_stats = {
            'team_corrections_applied': 0,
            'possessions_tracked': 0,
            'play_classifications': 0,
            'frames_with_teams': 0,
            'batch_optimizations': 0
        }

    def _initialize_components(self):
        """Initialize all processing components"""
        device = 'cuda' if self.config.use_gpu else 'cpu'

        # Detection with basketball-optimized parameters
        self.detector = YoloDetector(
            model_path=self.config.yolo_model_path,
            device=device,
            confidence=self.config.detection_confidence,
            nms_threshold=self.config.nms_threshold
        )

        # Enhanced mask generation for basketball
        if self.config.sam_checkpoint_path:
            self.mask_generator = MaskGenerator(
                checkpoint_path=self.config.sam_checkpoint_path,
                basketball_optimized=True
            )
        else:
            self.mask_generator = MaskGenerator(basketball_fallback=True)

        # Enhanced basketball tracking
        self.tracker = EnhancedTracker(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.min_matching_threshold,
            minimum_consecutive_frames=self.config.min_consecutive_frames,
            max_tracks=self.config.max_tracks,
            use_basketball_logic=True,
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
                basketball_enhanced=True,
                context_tracking=self.config.enable_context_tracking
            )
        else:
            self.possession_tracker = None

        if self.config.enable_play_classification:
            self.play_classifier = PlayClassifier(
                basketball_play_types=True,
                context_aware=self.config.enable_context_tracking
            )
        else:
            self.play_classifier = None

        if self.config.enable_event_detection:
            self.event_detector = EventDetector(basketball_events=True)
        else:
            self.event_detector = None

        if self.config.enable_pose_estimation:
            self.pose_estimator = PoseEstimator(basketball_actions=True)
            self.action_detector = ActionDetector(
                detect_screens=True, detect_cuts=True
            )
        else:
            self.pose_estimator = None
            self.action_detector = None

        # Timeline generator with basketball context
        self.timeline_generator = TimelineGenerator(basketball_timeline=True)

        # Basketball-aware visualization
        if self.config.enable_visualization:
            self.frame_annotator = FrameAnnotator(
                use_ellipse=self.config.use_ellipse_annotation,
                team_aware_colors=self.config.team_aware_colors,
                basketball_overlays=True
            )
        else:
            self.frame_annotator = None

    def _initialize_synchronized_processors(self):
        """Initialize synchronized FrameProcessor and BatchOptimizer"""
        # Initialize FrameProcessor with all components
        self.frame_processor = FrameProcessor(
            detector=self.detector,
            mask_generator=self.mask_generator,
            tracker=self.tracker,
            team_classifier=self.team_classifier,
            possession_tracker=self.possession_tracker,
            event_detector=self.event_detector,
            pose_estimator=self.pose_estimator,
            frame_annotator=self.frame_annotator,
            basketball_enhanced=True
        )

        # Initialize BatchOptimizer with basketball optimization
        self.batch_optimizer = BatchOptimizer(
            target_memory_usage=0.8,
            min_batch_size=1,
            max_batch_size=self.config.batch_size,
            basketball_optimized=True
        )

    def process_video(self, video_path: str,
                     progress_callback: Optional[callable] = None) -> AnalysisResult:
        """Process video with synchronized basketball analysis"""
        self.logger.info(f"Starting synchronized basketball video analysis: {video_path}")

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

        # Optimize batch size for basketball processing
        optimal_batch_size = self.batch_optimizer.optimize_basketball_batch_size(
            frame_size=(video_info['height'], video_info['width'], 3),
            enable_pose_estimation=self.config.enable_pose_estimation,
            enable_team_classification=self.config.use_ml_classification
        )

        self.logger.info(f"Using basketball-optimized batch size: {optimal_batch_size}")

        # Initialize video writer
        video_writer = None
        if self.config.output_video_path and self.config.enable_visualization:
            video_writer = VideoWriter(
                self.config.output_video_path,
                fps=video_info['fps'],
                width=video_info['width'],
                height=video_info['height'],
                basketball_codec=True
            )

        # Initialize streaming writer
        streaming_writer = None
        if self.config.streaming_output:
            streaming_writer = StreamingWriter(
                self.config.analytics_output_path or "basketball_analytics.pkl"
            )

        # Process video with synchronized components
        try:
            results = self._process_synchronized_frames(
                video_reader,
                video_writer,
                streaming_writer,
                start_frame,
                end_frame,
                optimal_batch_size,
                progress_callback
            )

            # Create basketball analysis result
            processing_time = time.time() - start_time
            analysis_result = self._create_synchronized_analysis_result(
                video_path, video_info, results, processing_time
            )

            # Save basketball analytics
            if self.config.save_analytics:
                self._save_synchronized_analytics(analysis_result)

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
                self.logger.info(f"Synchronized basketball analysis memory: {memory_stats}")
            cleanup_resources()

    def _process_synchronized_frames(self,
                                   video_reader: VideoReader,
                                   video_writer: Optional[VideoWriter],
                                   streaming_writer: Optional[StreamingWriter],
                                   start_frame: int,
                                   end_frame: int,
                                   optimal_batch_size: int,
                                   progress_callback: Optional[callable]) -> Dict:
        """Process frames using synchronized FrameProcessor and BatchOptimizer"""
        results = {
            'tracks': [],
            'teams': [],
            'possessions': [],
            'plays': [],
            'events': [],
            'frames_processed': 0,
            'basketball_stats': self.basketball_stats.copy(),
            'frame_processor_stats': {},
            'batch_optimizer_stats': {}
        }

        # Read all frames first for adaptive batching
        frame_generator = video_reader.read_frames(start_frame, end_frame)
        frame_data = list(frame_generator)  # [(frame, frame_idx), ...]

        if not frame_data:
            self.logger.warning("No frames to process")
            return results

        # Process using adaptive batch optimization
        if self.config.adaptive_batching:
            batch_generator = self.batch_optimizer.adaptive_basketball_batch_generator(
                frame_data,
                initial_batch_size=optimal_batch_size,
                player_count_hint=8  # Estimate ~8 players for basketball
            )
        else:
            # Fixed batch size
            def fixed_batch_generator():
                for i in range(0, len(frame_data), optimal_batch_size):
                    yield frame_data[i:i + optimal_batch_size]
            batch_generator = fixed_batch_generator()

        # Process batches
        batch_count = 0
        for batch in batch_generator:
            batch_count += 1
            batch_start_time = time.time()

            # Process each frame in batch using FrameProcessor
            for frame, frame_idx in batch:
                # Use synchronized FrameProcessor
                annotated_frame, frame_analytics = self.frame_processor.process_basketball_frame(
                    frame, frame_idx
                )

                # Store frame results
                self._store_frame_results(frame_analytics, results)

                # Write output
                if video_writer:
                    video_writer.write_frame(annotated_frame)

                # Stream results
                if streaming_writer:
                    synchronized_frame_data = {
                        'frame_idx': frame_idx,
                        'analytics': frame_analytics,
                        'synchronized': True,
                        'basketball_enhanced': True
                    }
                    streaming_writer.write(synchronized_frame_data)

                results['frames_processed'] += 1

                # Progress callback
                if progress_callback:
                    progress = results['frames_processed'] / len(frame_data)
                    progress_callback(progress)

            # Log batch processing
            batch_time = time.time() - batch_start_time
            self.logger.debug(f"Basketball batch {batch_count} processed in {batch_time:.2f}s")

            # Periodic cleanup and statistics update
            if batch_count % 5 == 0:
                cleanup_resources()
                self._update_synchronized_statistics(results)

        # Finalize results with synchronized data
        results = self._finalize_synchronized_results(results)

        # Get final statistics
        results['frame_processor_stats'] = self.frame_processor.get_basketball_statistics()
        results['batch_optimizer_stats'] = self.batch_optimizer.get_basketball_performance_summary()

        return results

    def _store_frame_results(self, frame_analytics: Dict, results: Dict):
        """Store frame analytics in results"""
        # Store tracks
        tracks = frame_analytics.get('tracks', [])
        if tracks:
            results['tracks'].extend(tracks)

        # Store possession changes
        possession = frame_analytics.get('possession')
        if possession and possession.possession_change:
            self.basketball_stats['possessions_tracked'] += 1

        # Store events
        events = frame_analytics.get('events', [])
        if events:
            results['events'].extend(events)

        # Update team statistics
        team_assignments = frame_analytics.get('team_assignments', {})
        if team_assignments:
            self.basketball_stats['frames_with_teams'] += 1

    def _update_synchronized_statistics(self, results: Dict):
        """Update synchronized processing statistics"""
        self.logger.info(f"Synchronized processing: {results['frames_processed']} frames, "
                        f"Teams in {self.basketball_stats['frames_with_teams']} frames")

    def _finalize_synchronized_results(self, results: Dict) -> Dict:
        """Finalize synchronized results"""
        # Get all tracks from tracker
        all_tracks = self.tracker.get_all_tracks() if hasattr(self.tracker, 'get_all_tracks') else []
        results['tracks'] = all_tracks

        # Extract teams
        results['teams'] = self._extract_synchronized_teams(all_tracks)

        # Get possession and play summaries
        if self.possession_tracker:
            if hasattr(self.possession_tracker, 'get_all_possessions'):
                results['possessions'] = self.possession_tracker.get_all_possessions()
            else:
                results['possessions'] = []

        if self.play_classifier:
            if hasattr(self.play_classifier, 'get_classified_plays'):
                results['plays'] = self.play_classifier.get_classified_plays()
            else:
                results['plays'] = []

        results['basketball_stats'] = self.basketball_stats

        return results

    def _extract_synchronized_teams(self, tracks: List[Track]) -> List[Dict]:
        """Extract basketball team information with synchronized validation"""
        teams = {0: {'id': 0, 'players': [], 'name': 'Team 0'}, 
                 1: {'id': 1, 'players': [], 'name': 'Team 1'}}

        for track in tracks:
            if track.team_id in teams:
                teams[track.team_id]['players'].append(track.id)

        # Synchronized basketball validation
        team0_count = len(teams[0]['players'])
        team1_count = len(teams[1]['players'])
        
        if abs(team0_count - team1_count) > 3:
            self.logger.warning(f"Synchronized basketball team imbalance: T0={team0_count}, T1={team1_count}")

        return list(teams.values())

    def _create_synchronized_analysis_result(self, video_path: str,
                                           video_info: Dict,
                                           results: Dict,
                                           processing_time: float) -> AnalysisResult:
        """Create synchronized basketball analysis result"""
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
            # Synchronized basketball metadata
            basketball_enhanced=True,
            team_classification_stats=self.team_classifier.get_basketball_statistics() if self.team_classifier else {},
            basketball_analysis_stats=self.basketball_stats,
            # NEW: Synchronized processing metadata
            synchronized_processing=True,
            frame_processor_stats=results.get('frame_processor_stats', {}),
            batch_optimizer_stats=results.get('batch_optimizer_stats', {}),
            adaptive_batching_enabled=self.config.adaptive_batching
        )

    def _save_synchronized_analytics(self, analysis_result: AnalysisResult):
        """Save synchronized basketball analytics"""
        output_path = self.config.analytics_output_path or "synchronized_basketball_analytics.json"

        import json
        with open(output_path, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)

        self.logger.info(f"Synchronized basketball analytics saved to: {output_path}")

    def get_synchronized_statistics(self) -> Dict:
        """Get comprehensive synchronized processing statistics"""
        stats = {
            'video_processor': self.basketball_stats.copy(),
            'synchronized': True,
            'basketball_optimized': True
        }

        if hasattr(self, 'frame_processor'):
            stats['frame_processor'] = self.frame_processor.get_basketball_statistics()

        if hasattr(self, 'batch_optimizer'):
            stats['batch_optimizer'] = self.batch_optimizer.get_basketball_performance_summary()

        if self.team_classifier:
            stats['team_classifier'] = self.team_classifier.get_basketball_statistics()

        return stats
