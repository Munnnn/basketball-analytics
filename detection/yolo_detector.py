"""
YOLO-based object detection with basketball optimizations
"""

import numpy as np
import torch
import supervision as sv
from typing import List, Optional, Dict, Tuple
from ultralytics import YOLO
import logging
import gc

from core import Detection
from core.interfaces import Detector
from core.constants import PLAYER_ID, BALL_ID, REF_ID, HOOP_ID, BACKBOARD_ID


class YoloDetector(Detector):
    """YOLO-based object detector optimized for basketball analytics"""
    
    def __init__(self, 
                 model_path: str, 
                 device: str = 'cuda', 
                 confidence: float = 0.2,
                 nms_threshold: float = 0.5,
                 basketball_optimized: bool = True):
        """
        Initialize YOLO detector with basketball optimizations
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on
            confidence: Minimum confidence threshold
            nms_threshold: NMS threshold for duplicate removal
            basketball_optimized: Enable basketball-specific optimizations
        """
        self.model = YOLO(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.basketball_optimized = basketball_optimized
        
        self.logger = logging.getLogger(__name__)
        
        # Basketball-specific optimization parameters
        if basketball_optimized:
            self.basketball_class_priorities = {
                PLAYER_ID: 1.0,    # Highest priority for players
                BALL_ID: 0.9,      # High priority for ball
                REF_ID: 0.7,       # Medium priority for referees
                HOOP_ID: 0.8,      # High priority for hoop
                BACKBOARD_ID: 0.8  # High priority for backboard
            }
            
            # Basketball-specific confidence adjustments
            self.basketball_confidence_boosts = {
                PLAYER_ID: 0.05,   # Slight boost for players
                BALL_ID: 0.1,      # Boost for ball (often missed)
                HOOP_ID: 0.0,      # No boost needed (usually clear)
                BACKBOARD_ID: 0.0  # No boost needed
            }
        
        # Detection statistics
        self.detection_stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'basketball_detections': 0,
            'batch_processing_count': 0,
            'class_detection_counts': {}
        }
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame with basketball optimizations"""
        if frame is None or frame.size == 0:
            return []
            
        try:
            results = self.model.predict(
                frame, 
                device=self.device, 
                conf=self.confidence, 
                verbose=False
            )
            
            detections = self._process_yolo_results(results, single_frame=True)
            
            # Update statistics
            self.detection_stats['total_frames_processed'] += 1
            self.detection_stats['total_detections'] += len(detections)
            
            # Basketball-specific post-processing
            if self.basketball_optimized:
                detections = self._apply_basketball_optimizations(detections)
                basketball_count = len([d for d in detections if d.class_id in [PLAYER_ID, BALL_ID, REF_ID]])
                self.detection_stats['basketball_detections'] += basketball_count
                
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray], basketball_optimized: bool = None) -> List[List[Detection]]:
        """
        Detect objects in multiple frames with basketball batch optimizations
        
        Args:
            frames: List of frames to process
            basketball_optimized: Override basketball optimization setting
            
        Returns:
            List of detection lists for each frame
        """
        if not frames:
            return []
            
        use_basketball_opt = basketball_optimized if basketball_optimized is not None else self.basketball_optimized
        
        try:
            # Basketball-optimized batch inference
            if use_basketball_opt:
                # Pre-process frames for basketball analysis
                preprocessed_frames = self._preprocess_basketball_frames(frames)
            else:
                preprocessed_frames = frames
            
            # YOLO batch inference with memory management
            results = self.model.predict(
                preprocessed_frames, 
                device=self.device, 
                conf=self.confidence, 
                verbose=False
            )
            
            batch_detections = []
            for i, result in enumerate(results):
                frame_detections = self._process_single_yolo_result(result, frame_idx=i)
                
                # Apply basketball optimizations per frame
                if use_basketball_opt:
                    frame_detections = self._apply_basketball_optimizations(frame_detections)
                    
                batch_detections.append(frame_detections)
                
            # Update batch statistics
            self.detection_stats['batch_processing_count'] += 1
            self.detection_stats['total_frames_processed'] += len(frames)
            
            total_detections = sum(len(dets) for dets in batch_detections)
            self.detection_stats['total_detections'] += total_detections
            
            if use_basketball_opt:
                basketball_detections = sum(
                    len([d for d in dets if d.class_id in [PLAYER_ID, BALL_ID, REF_ID]]) 
                    for dets in batch_detections
                )
                self.detection_stats['basketball_detections'] += basketball_detections
            
            # Clean up GPU memory
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return batch_detections
            
        except Exception as e:
            self.logger.warning(f"Batch inference failed: {e}. Falling back to single frame processing.")
            # Fallback to single frame processing
            batch_detections = []
            for frame in frames:
                batch_detections.append(self.detect(frame))
                
        return batch_detections
        
    def _preprocess_basketball_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess frames for basketball-optimized detection"""
        # For now, return frames as-is
        # Could add basketball-specific preprocessing like:
        # - Court detection and focusing
        # - Lighting adjustment for better player detection
        # - Resolution optimization
        return frames
        
    def _process_yolo_results(self, results, single_frame: bool = True) -> List[Detection]:
        """Process YOLO results into Detection objects"""
        detections = []
        
        if results and len(results) > 0:
            if single_frame:
                detections = self._process_single_yolo_result(results[0])
            else:
                # Batch processing
                for result in results:
                    frame_detections = self._process_single_yolo_result(result)
                    detections.extend(frame_detections)
                    
        return detections
        
    def _process_single_yolo_result(self, result, frame_idx: Optional[int] = None) -> List[Detection]:
        """Process single YOLO result into Detection objects"""
        detections = []
        
        try:
            sv_detections = sv.Detections.from_ultralytics(result)
            
            # Apply NMS with basketball-aware parameters
            if self.basketball_optimized:
                sv_detections = self._apply_basketball_nms(sv_detections)
            else:
                sv_detections = sv_detections.with_nms(
                    threshold=self.nms_threshold, 
                    class_agnostic=True
                )
            
            # Convert to Detection objects
            for i in range(len(sv_detections)):
                detection = Detection(
                    bbox=sv_detections.xyxy[i],
                    confidence=sv_detections.confidence[i],
                    class_id=int(sv_detections.class_id[i]),
                    mask=None,  # YOLO doesn't provide masks
                    frame_idx=frame_idx
                )
                detections.append(detection)
                
                # Update class statistics
                class_id = int(sv_detections.class_id[i])
                self.detection_stats['class_detection_counts'][class_id] = (
                    self.detection_stats['class_detection_counts'].get(class_id, 0) + 1
                )
                
        except Exception as e:
            self.logger.error(f"Error processing YOLO result: {e}")
            
        return detections
        
    def _apply_basketball_nms(self, sv_detections: sv.Detections) -> sv.Detections:
        """Apply basketball-aware Non-Maximum Suppression"""
        if len(sv_detections) == 0:
            return sv_detections
            
        # Standard NMS first
        sv_detections = sv_detections.with_nms(
            threshold=self.nms_threshold, 
            class_agnostic=False  # Class-specific for basketball
        )
        
        # Basketball-specific post-NMS filtering
        if len(sv_detections) > 0:
            # Keep only basketball-relevant classes with higher confidence
            keep_indices = []
            for i in range(len(sv_detections)):
                class_id = sv_detections.class_id[i]
                confidence = sv_detections.confidence[i]
                
                # Apply basketball confidence boost
                if class_id in self.basketball_confidence_boosts:
                    boosted_confidence = confidence + self.basketball_confidence_boosts[class_id]
                    if boosted_confidence >= self.confidence:
                        keep_indices.append(i)
                elif confidence >= self.confidence:
                    keep_indices.append(i)
                    
            if keep_indices:
                sv_detections = sv_detections[keep_indices]
                
        return sv_detections
        
    def _apply_basketball_optimizations(self, detections: List[Detection]) -> List[Detection]:
        """Apply basketball-specific optimizations to detections"""
        if not detections:
            return detections
            
        optimized_detections = []
        
        # Sort by basketball priority and confidence
        sorted_detections = sorted(
            detections, 
            key=lambda d: (
                self.basketball_class_priorities.get(d.class_id, 0.5),
                d.confidence
            ),
            reverse=True
        )
        
        # Basketball-specific filtering
        player_count = 0
        ball_count = 0
        
        for detection in sorted_detections:
            class_id = detection.class_id
            
            # Basketball-specific limits
            if class_id == PLAYER_ID:
                if player_count < 12:  # Max 12 players on court (10 + 2 subs)
                    optimized_detections.append(detection)
                    player_count += 1
            elif class_id == BALL_ID:
                if ball_count < 2:  # Max 2 balls (game ball + backup)
                    optimized_detections.append(detection)
                    ball_count += 1
            elif class_id in [REF_ID, HOOP_ID, BACKBOARD_ID]:
                # Always include referees and court elements
                optimized_detections.append(detection)
            else:
                # Include other detections as-is
                optimized_detections.append(detection)
                
        return optimized_detections
        
    def get_basketball_detection_stats(self) -> Dict:
        """Get basketball-specific detection statistics"""
        stats = self.detection_stats.copy()
        
        # Calculate basketball-specific metrics
        if stats['total_detections'] > 0:
            stats['basketball_detection_rate'] = (
                stats['basketball_detections'] / stats['total_detections']
            )
        else:
            stats['basketball_detection_rate'] = 0.0
            
        # Add class-specific stats
        basketball_classes = [PLAYER_ID, BALL_ID, REF_ID, HOOP_ID, BACKBOARD_ID]
        stats['basketball_class_counts'] = {
            class_id: stats['class_detection_counts'].get(class_id, 0)
            for class_id in basketball_classes
        }
        
        # Add performance metrics
        if stats['total_frames_processed'] > 0:
            stats['avg_detections_per_frame'] = (
                stats['total_detections'] / stats['total_frames_processed']
            )
            stats['avg_basketball_detections_per_frame'] = (
                stats['basketball_detections'] / stats['total_frames_processed']
            )
        else:
            stats['avg_detections_per_frame'] = 0.0
            stats['avg_basketball_detections_per_frame'] = 0.0
            
        return stats
        
    def get_detection_stats(self) -> Dict:
        """Legacy method - returns basketball detection stats"""
        return self.get_basketball_detection_stats()
        
    def filter_basketball_detections(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Filter detections by basketball-relevant classes"""
        filtered = {
            'players': [],
            'ball': [],
            'referees': [],
            'court_elements': [],
            'other': []
        }
        
        for detection in detections:
            class_id = detection.class_id
            
            if class_id == PLAYER_ID:
                filtered['players'].append(detection)
            elif class_id == BALL_ID:
                filtered['ball'].append(detection)
            elif class_id == REF_ID:
                filtered['referees'].append(detection)
            elif class_id in [HOOP_ID, BACKBOARD_ID]:
                filtered['court_elements'].append(detection)
            else:
                filtered['other'].append(detection)
                
        return filtered
        
    def optimize_for_basketball(self, enable: bool = True):
        """Enable or disable basketball optimizations"""
        self.basketball_optimized = enable
        if enable:
            self.logger.info("Basketball optimizations enabled")
        else:
            self.logger.info("Basketball optimizations disabled")
            
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'basketball_detections': 0,
            'batch_processing_count': 0,
            'class_detection_counts': {}
        }
        
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
