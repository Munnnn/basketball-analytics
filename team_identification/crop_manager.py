"""
Player crop extraction and management
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
import os
import tempfile 
import logging


class CropManager:
    """Manages player crop extraction and storage"""
    
    def __init__(self, max_crops: int = 5000, temp_dir: Optional[str] = None):
        """
        Initialize crop manager
        
        Args:
            max_crops: Maximum crops to store
            temp_dir: Temporary directory for crop storage
        """
        self.max_crops = max_crops
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), "basketball_crops")
        self.crops = []
        self.crop_metadata = []
        
        # Create temp directory if needed
        if temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)
            
    def extract_crop(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract crop from frame using bbox
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped image or None if invalid
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
            
        return frame[y1:y2, x1:x2].copy()
        
    def extract_jersey_region(self, crop: np.ndarray, ratio: float = 0.6) -> np.ndarray:
        """
        Extract jersey region from player crop
        
        Args:
            crop: Player crop
            ratio: Ratio of crop height to use for jersey
            
        Returns:
            Jersey region
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
            
        return jersey_region
        
    def add_crop(self, crop: np.ndarray, metadata: Optional[dict] = None) -> bool:
        """
        Add crop to collection
        
        Args:
            crop: Player crop
            metadata: Optional metadata
            
        Returns:
            True if added successfully
        """
        if crop is None or crop.size == 0:
            return False
            
        if len(self.crops) >= self.max_crops:
            # Remove oldest crop
            self.crops.pop(0)
            self.crop_metadata.pop(0)
            
        self.crops.append(crop.copy())
        self.crop_metadata.append(metadata or {})
        
        return True
        
    def get_training_crops(self, max_crops: Optional[int] = None, 
                          extract_jersey: bool = True) -> List[np.ndarray]:
        """
        Get crops for training
        
        Args:
            max_crops: Maximum crops to return
            extract_jersey: Whether to extract jersey regions
            
        Returns:
            List of crops
        """
        if not self.crops:
            return []
            
        # Sample evenly if needed
        if max_crops and len(self.crops) > max_crops:
            indices = np.linspace(0, len(self.crops) - 1, max_crops, dtype=int)
            selected_crops = [self.crops[i] for i in indices]
        else:
            selected_crops = self.crops
            
        # Extract jersey regions if requested
        if extract_jersey:
            jersey_crops = []
            for crop in selected_crops:
                jersey = self.extract_jersey_region(crop)
                jersey_crops.append(jersey)
            return jersey_crops
            
        return selected_crops
        
    def get_crop_count(self) -> int:
        """Get number of stored crops"""
        return len(self.crops)
        
    def clear(self):
        """Clear all stored crops"""
        self.crops.clear()
        self.crop_metadata.clear()
        
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
