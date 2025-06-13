"""
Player crop extraction and management with memory optimization
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
import os
import tempfile 
import logging
import gc


class CropManager:
    """Manages player crop extraction and storage with memory optimization"""
    
    def __init__(self, max_crops: int = 5000, temp_dir: Optional[str] = None):
        """
        Initialize crop manager with memory limits
        
        Args:
            max_crops: Maximum crops to store (reduced for memory)
            temp_dir: Temporary directory for crop storage
        """
        # Reduce max crops to prevent memory issues
        self.max_crops = min(max_crops, 5000)  # Hard limit
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), "basketball_crops")
        self.crops = []
        self.crop_metadata = []
        
        # Memory optimization settings
        self.max_crop_size = (128, 128)  # Limit crop dimensions
        self.cleanup_threshold = 0.8  # Cleanup when 80% full
        
        # Create temp directory if needed
        if temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)
            
    def extract_crop(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract and resize crop to limit memory usage
        """
        if frame is None or bbox is None:
            return None
            
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Validate bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = frame[y1:y2, x1:x2].copy()
        
        # Resize crop to limit memory usage
        if crop.shape[0] > self.max_crop_size[0] or crop.shape[1] > self.max_crop_size[1]:
            crop = cv2.resize(crop, self.max_crop_size, interpolation=cv2.INTER_AREA)
            
        return crop
        
    def extract_jersey_region(self, crop: np.ndarray, ratio: float = 0.6) -> np.ndarray:
        """
        Extract jersey region with size limits
        """
        if crop is None or crop.size == 0:
            return crop
            
        h, w = crop.shape[:2]
        
        # Extract upper middle portion (jersey area)
        jersey_top = h // 6
        jersey_bottom = int(h * ratio)
        jersey_left = w // 4
        jersey_right = 3 * w // 4
        
        jersey_region = crop[jersey_top:jersey_bottom, jersey_left:jersey_right]
        
        # Fallback if extraction failed
        if jersey_region.size == 0:
            jersey_region = crop[:int(h * ratio), :]
            
        # Ensure reasonable size
        if jersey_region.shape[0] > 64 or jersey_region.shape[1] > 64:
            jersey_region = cv2.resize(jersey_region, (64, 64), interpolation=cv2.INTER_AREA)
            
        return jersey_region
        
    def add_crop(self, crop: np.ndarray, metadata: Optional[dict] = None) -> bool:
        """
        Add crop with memory management
        """
        if crop is None or crop.size == 0:
            return False
            
        # Check if we need cleanup
        if len(self.crops) >= int(self.max_crops * self.cleanup_threshold):
            self._cleanup_old_crops()
            
        if len(self.crops) >= self.max_crops:
            # Remove oldest crop
            if self.crops:
                del self.crops[0]
                del self.crop_metadata[0]
                
        # Resize crop if too large
        if crop.shape[0] > self.max_crop_size[0] or crop.shape[1] > self.max_crop_size[1]:
            crop = cv2.resize(crop, self.max_crop_size, interpolation=cv2.INTER_AREA)
            
        self.crops.append(crop.copy())
        self.crop_metadata.append(metadata or {})
        
        return True
        
    def get_training_crops(self, max_crops: Optional[int] = None, 
                          extract_jersey: bool = True) -> List[np.ndarray]:
        """
        Get crops for training with memory limits
        """
        if not self.crops:
            return []
            
        # Limit crops to prevent memory issues
        effective_max = min(max_crops or len(self.crops), 1000, len(self.crops))
        
        # Sample evenly if needed
        if len(self.crops) > effective_max:
            indices = np.linspace(0, len(self.crops) - 1, effective_max, dtype=int)
            selected_crops = [self.crops[i] for i in indices]
        else:
            selected_crops = self.crops.copy()
            
        # Extract jersey regions if requested
        if extract_jersey:
            jersey_crops = []
            for crop in selected_crops:
                try:
                    jersey = self.extract_jersey_region(crop)
                    jersey_crops.append(jersey)
                except Exception as e:
                    logging.warning(f"Failed to extract jersey region: {e}")
                    # Use original crop if jersey extraction fails
                    jersey_crops.append(crop)
            return jersey_crops
            
        return selected_crops
        
    def _cleanup_old_crops(self):
        """Remove old crops to free memory"""
        if len(self.crops) > self.max_crops // 2:
            # Keep only the most recent half
            keep_count = self.max_crops // 2
            self.crops = self.crops[-keep_count:]
            self.crop_metadata = self.crop_metadata[-keep_count:]
            
            # Force garbage collection
            gc.collect()
            
            logging.info(f"Cleaned up old crops, keeping {len(self.crops)} crops")
        
    def get_crop_count(self) -> int:
        """Get number of stored crops"""
        return len(self.crops)
        
    def clear(self):
        """Clear all stored crops"""
        self.crops.clear()
        self.crop_metadata.clear()
        gc.collect()
        
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage of stored crops"""
        if not self.crops:
            return 0.0
            
        # Estimate memory usage
        total_bytes = 0
        for crop in self.crops:
            if crop is not None:
                total_bytes += crop.nbytes
                
        return total_bytes / (1024 * 1024)  # Convert to MB

    def save_crops(self, output_dir: str, sample_size: int = 100):
        """Save sample of crops to disk for debugging"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample crops
        if len(self.crops) > sample_size:
            indices = np.random.choice(len(self.crops), sample_size, replace=False)
        else:
            indices = range(len(self.crops))
            
        for i, idx in enumerate(indices):
            crop = self.crops[idx]
            metadata = self.crop_metadata[idx]
            
            # Save crop
            filename = f"crop_{i:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, crop)
            
            # Save metadata if exists
            if metadata:
                import json
                meta_file = f"crop_{i:04d}.json"
                meta_path = os.path.join(output_dir, meta_file)
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                    
        logging.info(f"Saved {len(indices)} crops to {output_dir}")
