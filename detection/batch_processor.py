"""
Batch processing optimization for detection pipeline
"""

import numpy as np
from typing import List, Tuple, Optional, Generator
import logging

from core import Detection


class BatchProcessor:
    """Optimized batch processing for detection pipeline"""
    
    def __init__(self, batch_size: int = 20):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of frames to process in each batch
        """
        self.batch_size = max(1, batch_size)
        
    def create_batches(self, items: List, batch_size: Optional[int] = None) -> Generator[List, None, None]:
        """
        Create batches from a list of items
        
        Args:
            items: List of items to batch
            batch_size: Override default batch size
            
        Yields:
            Batches of items
        """
        batch_size = batch_size or self.batch_size
        
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
            
    def process_video_in_batches(self, video_path: str, 
                                detector, 
                                mask_generator,
                                start_frame: int = 0,
                                end_frame: Optional[int] = None) -> Generator[Tuple[int, List[Detection]], None, None]:
        """
        Process video in batches
        
        Args:
            video_path: Path to video file
            detector: Object detector instance
            mask_generator: Mask generator instance
            start_frame: Starting frame
            end_frame: Ending frame (None for all)
            
        Yields:
            Tuple of (frame_index, detections_with_masks)
        """
        import cv2
        
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
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                
                # Process batch when full
                if len(batch_frames) >= self.batch_size or frame_idx == end_frame - 1:
                    # Detect objects in batch
                    batch_detections = detector.detect_batch(batch_frames)
                    
                    # Generate masks for each frame
                    for i, (frame, detections) in enumerate(zip(batch_frames, batch_detections)):
                        if detections:
                            masks = mask_generator.generate_masks(frame, detections)
                            
                            # Add masks to detections
                            for j, (det, mask) in enumerate(zip(detections, masks)):
                                det.mask = mask
                                det.frame_idx = batch_indices[i]
                                
                        yield batch_indices[i], detections
                        
                    # Clear batch
                    batch_frames.clear()
                    batch_indices.clear()
                    
        finally:
            cap.release()
            
    def merge_detections(self, all_detections: List[Detection], 
                        tracked_detections: List[Detection]) -> List[Detection]:
        """
        Merge all detections with tracked detections
        
        Args:
            all_detections: All detected objects
            tracked_detections: Tracked player detections
            
        Returns:
            Merged list with tracked players replacing untracked ones
        """
        if not tracked_detections:
            return all_detections
            
        # Create mapping of bbox to tracked detection
        tracked_map = {}
        for det in tracked_detections:
            key = tuple(det.bbox)
            tracked_map[key] = det
            
        # Replace matching detections
        merged = []
        for det in all_detections:
            key = tuple(det.bbox)
            if key in tracked_map:
                merged.append(tracked_map[key])
            else:
                merged.append(det)
                
        return merged
