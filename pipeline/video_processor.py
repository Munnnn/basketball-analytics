"""
Main video processing pipeline
"""

import os
import logging
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
    
    # Team classification
    use_ml_classification: bool = True
    use_color_fallback: bool = True
    enforce_basketball_rules: bool = True
    
    # Analytics
    enable_possession_tracking: bool = True
    enable_play_classification: bool = True
    enable_pose_estimation: bool = True
    enable_event_detection: bool = True
    
    # Visualization
    enable_visualization: bool = True
    use_ellipse_annotation: bool = True
    
    # Output
    output_video_path: Optional[str] = None
    save_analytics: bool = True
    analytics_output_path: Optional[str] = None
    streaming_output: bool = False
    
    # Performance
    use_gpu: bool = True
    memory_optimization: bool = True
    

class VideoProcessor:
    """Main video processing pipeline orchestrator"""
    
    def __init__(self, config: ProcessorConfig):
        """
        Initialize video processor with configuration
        
        Args:
            config: Processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor() if config.memory_optimization else None
        
    def _initialize_components(self):
        """Initialize all processing components"""
        device = 'cuda' if self.config.use_gpu else 'cpu'
        
        # Detection
        self.detector = YoloDetector(
            model_path=self.config.yolo_model_path,
            device=device,
            confidence=self.config.detection_confidence
        )
        
        # Mask generation
        if self.config.sam_checkpoint_path:
            self.mask_generator = MaskGenerator(
                checkpoint_path=self.config.sam_checkpoint_path
            )
        else:
            self.mask_generator = MaskGenerator()  # Fallback mode
            
        # Batch processor
        self.batch_processor = BatchProcessor(batch_size=self.config.batch_size)
        
        # Tracking
        self.tracker = EnhancedTracker(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.min_matching_threshold,
            minimum_consecutive_frames=self.config.min_consecutive_frames,
            max_tracks=self.config.max_tracks
        )
        
        # Team classification
        self.team_classifier = UnifiedTeamClassifier(
            use_ml=self.config.use_ml_classification,
            use_color_fallback=self.config.use_color_fallback,
            enforce_basketball_rules=self.config.enforce_basketball_rules,
            device=device
        )
        
        # Analytics components
        if self.config.enable_possession_tracking:
            self.possession_tracker = PossessionTracker()
        else:
            self.possession_tracker = None
            
        if self.config.enable_play_classification:
            self.play_classifier = PlayClassifier()
        else:
            self.play_classifier = None
            
        if self.config.enable_event_detection:
            self.event_detector = EventDetector()
        else:
            self.event_detector = None
            
        if self.config.enable_pose_estimation:
            self.pose_estimator = PoseEstimator()
            self.action_detector = ActionDetector()
        else:
            self.pose_estimator = None
            self.action_detector = None
            
        # Timeline generator
        self.timeline_generator = TimelineGenerator()
        
        # Visualization
        if self.config.enable_visualization:
            self.frame_annotator = FrameAnnotator(
                use_ellipse=self.config.use_ellipse_annotation
            )
        else:
            self.frame_annotator = None
            
    def process_video(self, video_path: str, 
                     progress_callback: Optional[callable] = None) -> AnalysisResult:
        """
        Process entire video and return analysis results
        
        Args:
            video_path: Path to input video
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete analysis results
        """
        self.logger.info(f"Starting video processing: {video_path}")
        
        # Start memory monitoring
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()
            
        # Initialize video reader
        video_reader = VideoReader(video_path)
        video_info = video_reader.get_info()
        
        # Determine frame range
        start_frame = self.config.start_frame
        end_frame = self.config.end_frame or video_info['total_frames']
        total_frames = end_frame - start_frame
        
        # Initialize video writer if needed
        video_writer = None
        if self.config.output_video_path and self.config.enable_visualization:
            video_writer = VideoWriter(
                self.config.output_video_path,
                fps=video_info['fps'],
                width=video_info['width'],
                height=video_info['height']
            )
            
        # Initialize streaming writer if needed
        streaming_writer = None
        if self.config.streaming_output:
            streaming_writer = StreamingWriter(
                self.config.analytics_output_path or "analytics.pkl"
            )
            
        # Process video
        try:
            results = self._process_frames(
                video_reader, 
                video_writer,
                streaming_writer,
                start_frame, 
                end_frame,
                progress_callback
            )
            
            # Create final analysis result
            analysis_result = self._create_analysis_result(
                video_path, video_info, results
            )
            
            # Save analytics if requested
            if self.config.save_analytics:
                self._save_analytics(analysis_result)
                
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
                self.logger.info(f"Memory usage: {memory_stats}")
            cleanup_resources()
            
    def _process_frames(self, 
                       video_reader: VideoReader,
                       video_writer: Optional[VideoWriter],
                       streaming_writer: Optional[StreamingWriter],
                       start_frame: int,
                       end_frame: int,
                       progress_callback: Optional[callable]) -> Dict:
        """Process frames and collect results"""
        results = {
            'tracks': [],
            'teams': [],
            'possessions': [],
            'plays': [],
            'events': [],
            'frames_processed': 0
        }
        
        # Process in batches
        frame_generator = video_reader.read_frames(start_frame, end_frame)
        
        for batch_data in self.batch_processor.create_batches(
            frame_generator, self.config.batch_size
        ):
            frames, frame_indices = zip(*batch_data)
            
            # Detect objects
            batch_detections = self.detector.detect_batch(list(frames))
            
            # Process each frame
            for frame, frame_idx, detections in zip(frames, frame_indices, batch_detections):
                # Generate masks
                masks = self.mask_generator.generate_masks(frame, detections)
                for det, mask in zip(detections, masks):
                    det.mask = mask
                    
                # Separate by class
                player_detections = [d for d in detections if d.class_id == 5]  # PLAYER_ID
                ball_detections = [d for d in detections if d.class_id == 1]    # BALL_ID
                
                # Update tracking
                tracks = self.tracker.update(player_detections, frame)
                
                # Classify teams
                if tracks and not self.team_classifier.is_initialized():
                    crops = [self._extract_crop(frame, t.current_bbox) for t in tracks]
                    self.team_classifier.fit([c for c in crops if c is not None])
                    
                if self.team_classifier.is_initialized():
                    team_assignments = self.team_classifier.classify(tracks, frame)
                    for track in tracks:
                        track.team_id = team_assignments.get(track.id)
                        
                # Update analytics
                frame_results = self._update_analytics(
                    frame, frame_idx, tracks, ball_detections, detections
                )
                
                # Visualize if enabled
                annotated_frame = frame
                if self.config.enable_visualization and self.frame_annotator:
                    annotated_frame = self.frame_annotator.annotate(
                        frame, detections, tracks, frame_results
                    )
                    
                # Write output
                if video_writer:
                    video_writer.write_frame(annotated_frame)
                    
                # Stream results
                if streaming_writer:
                    streaming_writer.write({
                        'frame_idx': frame_idx,
                        'tracks': tracks,
                        'analytics': frame_results
                    })
                    
                results['frames_processed'] += 1
                
                # Progress callback
                if progress_callback:
                    progress = results['frames_processed'] / (end_frame - start_frame)
                    progress_callback(progress)
                    
            # Periodic cleanup
            if results['frames_processed'] % 100 == 0:
                cleanup_resources()
                
        # Finalize results
        results['tracks'] = self.tracker.get_all_tracks()
        results['teams'] = self._extract_teams(results['tracks'])
        
        return results
        
    def _update_analytics(self, frame: np.ndarray, frame_idx: int,
                         tracks: List[Track], ball_detections: List[Detection],
                         all_detections: List[Detection]) -> Dict:
        """Update analytics components"""
        frame_results = {}
        
        # Find ball track
        ball_track = None
        if ball_detections:
            # Simple ball tracking (could be enhanced)
            ball_track = Track(
                id=-1,  # Special ID for ball
                detections=ball_detections[:1],
                state='tracked'
            )
            
        # Possession tracking
        if self.possession_tracker:
            possession_info = self.possession_tracker.update_possession(
                tracks, ball_track, frame_idx
            )
            frame_results['possession'] = possession_info
            self.timeline_generator.add_possession_change(possession_info)
            
        # Pose estimation
        if self.pose_estimator and tracks:
            crops = [self._extract_crop(frame, t.current_bbox) for t in tracks]
            poses = self.pose_estimator.extract_poses(crops)
            
            if self.action_detector:
                positions = [t.current_position for t in tracks]
                actions = self.action_detector.detect_actions(poses, positions)
                frame_results['actions'] = actions
                
        # Event detection
        if self.event_detector:
            detection_dict = {
                'ball': ball_detections,
                'hoop': [d for d in all_detections if d.class_id == 3],
                'backboard': [d for d in all_detections if d.class_id == 0]
            }
            
            events = self.event_detector.detect_events(
                detection_dict,
                frame_results.get('possession'),
                frame_idx
            )
            
            for event in events:
                self.timeline_generator.add_event(event)
            frame_results['events'] = events
            
        # Play classification
        if self.play_classifier and self.possession_tracker:
            # Check if possession is complete
            if hasattr(self.possession_tracker, 'last_completed_possession'):
                possession_data = self.possession_tracker.last_completed_possession
                context = self.possession_tracker.get_context()
                
                play = self.play_classifier.classify_play(possession_data, context)
                self.timeline_generator.add_play_classification(play)
                frame_results['play'] = play
                
        return frame_results
        
    def _extract_crop(self, frame: np.ndarray, bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extract crop from frame"""
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return None
        
    def _extract_teams(self, tracks: List[Track]) -> List[Dict]:
        """Extract team information from tracks"""
        teams = {0: {'id': 0, 'players': []}, 1: {'id': 1, 'players': []}}
        
        for track in tracks:
            if track.team_id in teams:
                teams[track.team_id]['players'].append(track.id)
                
        return list(teams.values())
        
    def _create_analysis_result(self, video_path: str, 
                              video_info: Dict, 
                              results: Dict) -> AnalysisResult:
        """Create final analysis result"""
        # Get timeline data
        timeline_data = self.timeline_generator.generate_timeline()
        
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
            processing_time=0.0,  # Would need timing
            pose_estimation_enabled=self.config.enable_pose_estimation,
            context_tracking_enabled=True
        )
        
    def _save_analytics(self, analysis_result: AnalysisResult):
        """Save analytics to file"""
        output_path = self.config.analytics_output_path or "analytics.json"
        
        import json
        with open(output_path, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)
            
        self.logger.info(f"Analytics saved to: {output_path}")
