"""
Batch processing optimization for detection pipeline with basketball enhancements
"""

import numpy as np
from typing import List, Tuple, Optional, Generator, Dict
import logging
import cv2
import time

from core import Detection
from core.constants import PLAYER_ID, BALL_ID, REF_ID


class BatchProcessor:
    """Optimized batch processing for basketball detection pipeline"""
    
    def __init__(self, 
                 batch_size: int = 20,
                 basketball_optimized: bool = True,
                 memory_efficient: bool = True):
        """
        Initialize batch processor with basketball optimizations
        
        Args:
            batch_size: Number of frames to process in each batch
            basketball_optimized: Enable basketball-specific optimizations
            memory_efficient: Enable memory-efficient processing
        """
        self.batch_size = max(1, batch_size)
        self.basketball_optimized = basketball_optimized
        self.memory_efficient = memory_efficient
        
        self.logger = logging.getLogger(__name__)
        
        # Basketball processing statistics
        self.basketball_stats = {
            'total_batches_processed': 0,
            'total_frames_processed': 0,
            'basketball_detections_processed': 0,
            'avg_processing_time_per_batch': 0.0,
            'memory_optimizations_applied': 0
        }
        
    def create_batches(self, items: List, batch_size: Optional[int] = None) -> Generator[List, None, None]:
        """
        Create batches from a list of items with basketball optimizations
        
        Args:
            items: List of items to batch
            batch_size: Override default batch size
            
        Yields:
            Batches of items optimized for basketball processing
        """
        effective_batch_size = batch_size or self.batch_size
        
        # Basketball optimization: adjust batch size based on content
        if self.basketball_optimized and hasattr(items, '__len__') and len(items) > 0:
            # For basketball videos, smaller batches often work better for memory
            effective_batch_size = min(effective_batch_size, 16)
        
        for i in range(0, len(items), effective_batch_size):
            batch = items[i:i + effective_batch_size]
            
            # Basketball-specific batch preprocessing
            if self.basketball_optimized:
                batch = self._preprocess_basketball_batch(batch)
                
            yield batch
            
    def create_basketball_batches(self, frames: List[np.ndarray], 
                                 batch_size: Optional[int] = None) -> Generator[List[np.ndarray], None, None]:
        """
        Create basketball-optimized batches from frames
        
        Args:
            frames: List of video frames
            batch_size: Override default batch size
            
        Yields:
            Basketball-optimized batches of frames
        """
        effective_batch_size = batch_size or self.batch_size
        
        # Basketball-specific batch size optimization
        if self.basketball_optimized:
            # Analyze frame complexity for optimal batching
            avg_complexity = self._estimate_basketball_complexity(frames[:5])  # Sample first 5 frames
            
            if avg_complexity > 0.8:  # High complexity (many players)
                effective_batch_size = max(1, int(effective_batch_size * 0.7))
            elif avg_complexity < 0.3:  # Low complexity (few players)
                effective_batch_size = min(32, int(effective_batch_size * 1.3))
                
        self.logger.debug(f"Basketball batch size: {effective_batch_size} (complexity: {avg_complexity:.2f})")
        
        for i in range(0, len(frames), effective_batch_size):
            batch = frames[i:i + effective_batch_size]
            
            # Apply basketball-specific preprocessing
            if self.basketball_optimized:
                batch = self._preprocess_basketball_frame_batch(batch)
                
            yield batch
            
    def process_video_in_batches(self, video_path: str, 
                                detector, 
                                mask_generator,
                                start_frame: int = 0,
                                end_frame: Optional[int] = None,
                                basketball_enhanced: bool = None) -> Generator[Tuple[int, List[Detection]], None, None]:
        """
        Process video in batches with basketball optimizations
        
        Args:
            video_path: Path to video file
            detector: Object detector instance
            mask_generator: Mask generator instance
            start_frame: Starting frame
            end_frame: Ending frame (None for all)
            basketball_enhanced: Override basketball optimization setting
            
        Yields:
            Tuple of (frame_index, detections_with_masks)
        """
        use_basketball = basketball_enhanced if basketball_enhanced is not None else self.basketball_optimized
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames
            
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        try:
            batch_frames = []
            batch_indices = []
            batch_count = 0
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                
                # Process batch when full or at end
                if len(batch_frames) >= self.batch_size or frame_idx == end_frame - 1:
                    batch_count += 1
                    batch_start_time = time.time()
                    
                    try:
                        # Basketball-optimized batch detection
                        if use_basketball and hasattr(detector, 'detect_batch'):
                            batch_detections = detector.detect_batch(
                                batch_frames, basketball_optimized=True
                            )
                        else:
                            # Standard batch detection
                            batch_detections = detector.detect_batch(batch_frames)
                        
                        # Generate masks for each frame with basketball optimization
                        for i, (frame, detections) in enumerate(zip(batch_frames, batch_detections)):
                            frame_idx_current = batch_indices[i]
                            
                            if detections:
                                # Basketball-optimized mask generation
                                if use_basketball and hasattr(mask_generator, 'generate_basketball_masks'):
                                    masks = mask_generator.generate_basketball_masks(frame, detections)
                                else:
                                    masks = mask_generator.generate_masks(frame, detections)
                                
                                # Add masks and frame info to detections
                                for j, (det, mask) in enumerate(zip(detections, masks)):
                                    det.mask = mask
                                    det.frame_idx = frame_idx_current
                                    
                                # Basketball-specific detection filtering
                                if use_basketball:
                                    detections = self._filter_basketball_detections(detections)
                                    
                            yield frame_idx_current, detections
                            
                        # Update basketball statistics
                        if use_basketball:
                            self._update_basketball_batch_stats(batch_detections, time.time() - batch_start_time)
                            
                    except Exception as e:
                        self.logger.error(f"Basketball batch processing failed: {e}")
                        # Fallback: yield empty detections for failed batch
                        for frame_idx_current in batch_indices:
                            yield frame_idx_current, []
                        
                    # Clear batch and apply memory cleanup
                    batch_frames.clear()
                    batch_indices.clear()
                    
                    if self.memory_efficient and batch_count % 5 == 0:
                        self._cleanup_memory()
                        
        finally:
            cap.release()
            
    def merge_detections(self, all_detections: List[Detection], 
                        tracked_detections: List[Detection],
                        basketball_prioritized: bool = None) -> List[Detection]:
        """
        Merge all detections with tracked detections with basketball prioritization
        
        Args:
            all_detections: All detected objects
            tracked_detections: Tracked player detections
            basketball_prioritized: Enable basketball-specific merging
            
        Returns:
            Merged list with tracked players replacing untracked ones
        """
        use_basketball = basketball_prioritized if basketball_prioritized is not None else self.basketball_optimized
        
        if not tracked_detections:
            return all_detections
            
        # Create mapping of bbox to tracked detection
        tracked_map = {}
        for det in tracked_detections:
            key = self._create_detection_key(det.bbox)
            tracked_map[key] = det
            
        # Merge with basketball prioritization
        merged = []
        basketball_detections = []
        other_detections = []
        
        for det in all_detections:
            key = self._create_detection_key(det.bbox)
            
            if key in tracked_map:
                # Use tracked version (has team info, etc.)
                merged_det = tracked_map[key]
                if use_basketball and merged_det.class_id in [PLAYER_ID, BALL_ID, REF_ID]:
                    basketball_detections.append(merged_det)
                else:
                    other_detections.append(merged_det)
            else:
                # Use original detection
                if use_basketball and det.class_id in [PLAYER_ID, BALL_ID, REF_ID]:
                    basketball_detections.append(det)
                else:
                    other_detections.append(det)
                    
        # Basketball prioritization: put basketball detections first
        if use_basketball:
            merged = basketball_detections + other_detections
        else:
            merged = basketball_detections + other_detections
            
        return merged
        
    def _preprocess_basketball_batch(self, batch: List) -> List:
        """Preprocess batch for basketball analysis"""
        # For now, return batch as-is
        # Could add basketball-specific preprocessing like:
        # - Frame quality assessment
        # - Court detection hints
        # - Player count estimation
        return batch
        
    def _preprocess_basketball_frame_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess frame batch for basketball detection"""
        if not self.basketball_optimized:
            return frames
            
        # Basketball-specific frame preprocessing
        processed_frames = []
        
        for frame in frames:
            # Could add:
            # - Contrast enhancement for better jersey detection
            # - Resolution normalization
            # - Court boundary detection
            processed_frames.append(frame)  # For now, no preprocessing
            
        return processed_frames
        
    def _estimate_basketball_complexity(self, sample_frames: List[np.ndarray]) -> float:
        """Estimate basketball processing complexity from sample frames"""
        if not sample_frames:
            return 0.5  # Default complexity
            
        try:
            # Simple complexity estimation based on frame characteristics
            complexities = []
            
            for frame in sample_frames:
                if frame is None or frame.size == 0:
                    continue
                    
                # Estimate complexity based on image properties
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Edge density (more edges = more complex)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # Texture complexity (standard deviation)
                texture_complexity = np.std(gray) / 255.0
                
                # Combined complexity score
                complexity = (edge_density * 0.6 + texture_complexity * 0.4)
                complexities.append(complexity)
                
            return np.mean(complexities) if complexities else 0.5
            
        except Exception as e:
            self.logger.warning(f"Complexity estimation failed: {e}")
            return 0.5
            
    def _filter_basketball_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections for basketball relevance"""
        if not self.basketball_optimized:
            return detections
            
        # Sort by basketball priority
        basketball_priority = {PLAYER_ID: 3, BALL_ID: 2, REF_ID: 1}
        
        def get_priority(det):
            return (
                basketball_priority.get(det.class_id, 0),
                det.confidence
            )
            
        # Sort and apply basketball-specific limits
        sorted_detections = sorted(detections, key=get_priority, reverse=True)
        
        filtered = []
        player_count = 0
        ball_count = 0
        
        for det in sorted_detections:
            if det.class_id == PLAYER_ID:
                if player_count < 12:  # Max players on court
                    filtered.append(det)
                    player_count += 1
            elif det.class_id == BALL_ID:
                if ball_count < 2:  # Max balls
                    filtered.append(det)
                    ball_count += 1
            else:
                filtered.append(det)  # Include all other detections
                
        return filtered
        
    def _create_detection_key(self, bbox: np.ndarray) -> tuple:
        """Create a key for detection matching"""
        # Round bbox coordinates for approximate matching
        return tuple(np.round(bbox, 1))
        
    def _update_basketball_batch_stats(self, batch_detections: List[List[Detection]], processing_time: float):
        """Update basketball processing statistics"""
        self.basketball_stats['total_batches_processed'] += 1
        
        total_detections = sum(len(dets) for dets in batch_detections)
        basketball_detections = sum(
            len([d for d in dets if d.class_id in [PLAYER_ID, BALL_ID, REF_ID]])
            for dets in batch_detections
        )
        
        self.basketball_stats['total_frames_processed'] += len(batch_detections)
        self.basketball_stats['basketball_detections_processed'] += basketball_detections
        
        # Update average processing time
        prev_avg = self.basketball_stats['avg_processing_time_per_batch']
        batch_count = self.basketball_stats['total_batches_processed']
        
        self.basketball_stats['avg_processing_time_per_batch'] = (
            (prev_avg * (batch_count - 1) + processing_time) / batch_count
        )
        
    def _cleanup_memory(self):
        """Clean up memory during processing"""
        import gc
        gc.collect()
        
        # CUDA cleanup if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
        self.basketball_stats['memory_optimizations_applied'] += 1
        
    def get_basketball_statistics(self) -> Dict:
        """Get basketball processing statistics"""
        stats = self.basketball_stats.copy()
        
        # Calculate derived metrics
        if stats['total_frames_processed'] > 0:
            stats['avg_basketball_detections_per_frame'] = (
                stats['basketball_detections_processed'] / stats['total_frames_processed']
            )
        else:
            stats['avg_basketball_detections_per_frame'] = 0.0
            
        return stats
        
    def reset_statistics(self):
        """Reset processing statistics"""
        self.basketball_stats = {
            'total_batches_processed': 0,
            'total_frames_processed': 0,
            'basketball_detections_processed': 0,
            'avg_processing_time_per_batch': 0.0,
            'memory_optimizations_applied': 0
        }
