"""
Object detection pipeline for basketball analytics

This module provides basketball-optimized object detection components:
- YoloDetector: Basketball-optimized YOLO detection with class priorities
- MaskGenerator: Basketball-aware mask generation with EdgeTAM/SAM support
- BatchProcessor: Memory-efficient batch processing for basketball videos

All components maintain original class names for easy integration while adding
basketball-specific optimizations from the original enhanced tracker.
"""

from .yolo_detector import YoloDetector
from .mask_generator import (
    MaskGenerator, 
    EdgeTAMMaskGenerator, 
    SAMMaskGenerator, 
    SimpleMaskGenerator
)
from .batch_processor import BatchProcessor

# Version information
__version__ = "2.0.0"
__basketball_enhanced__ = True

# Export all classes
__all__ = [
    # Main detection components
    'YoloDetector',
    'MaskGenerator', 
    'EdgeTAMMaskGenerator', 
    'SAMMaskGenerator',
    'SimpleMaskGenerator',
    'BatchProcessor',
    
    # Module metadata
    '__version__',
    '__basketball_enhanced__'
]

# Basketball detection utilities
def create_basketball_detector(model_path: str, 
                             device: str = 'cuda',
                             confidence: float = 0.2) -> YoloDetector:
    """
    Create a basketball-optimized YOLO detector
    
    Args:
        model_path: Path to YOLO model weights
        device: Device for inference ('cuda' or 'cpu')
        confidence: Minimum confidence threshold
        
    Returns:
        Configured YoloDetector with basketball optimizations
    """
    return YoloDetector(
        model_path=model_path,
        device=device,
        confidence=confidence,
        basketball_optimized=True
    )

def create_basketball_mask_generator(backend: str = 'auto',
                                   checkpoint_path: str = "./checkpoints/edgetam.pt") -> MaskGenerator:
    """
    Create a basketball-optimized mask generator
    
    Args:
        backend: Backend to use ('auto', 'edgetam', 'simple')
        checkpoint_path: Path to EdgeTAM/SAM checkpoint
        
    Returns:
        Configured MaskGenerator with basketball optimizations
    """
    return MaskGenerator(
        backend=backend,
        checkpoint_path=checkpoint_path,
        basketball_optimized=True
    )

def create_basketball_batch_processor(batch_size: int = 16) -> BatchProcessor:
    """
    Create a basketball-optimized batch processor
    
    Args:
        batch_size: Number of frames per batch (optimized for basketball)
        
    Returns:
        Configured BatchProcessor with basketball optimizations
    """
    return BatchProcessor(
        batch_size=batch_size,
        basketball_optimized=True,
        memory_efficient=True
    )

# Convenience function for complete basketball detection setup
def setup_basketball_detection_pipeline(model_path: str,
                                       device: str = 'cuda',
                                       batch_size: int = 16,
                                       mask_backend: str = 'auto') -> tuple:
    """
    Set up complete basketball detection pipeline
    
    Args:
        model_path: Path to YOLO model weights
        device: Device for inference
        batch_size: Batch size for processing
        mask_backend: Mask generation backend
        
    Returns:
        Tuple of (detector, mask_generator, batch_processor)
    """
    detector = create_basketball_detector(model_path, device)
    mask_generator = create_basketball_mask_generator(mask_backend)
    batch_processor = create_basketball_batch_processor(batch_size)
    
    return detector, mask_generator, batch_processor

# Basketball class ID constants (re-exported for convenience)
from core.constants import PLAYER_ID, BALL_ID, REF_ID, HOOP_ID, BACKBOARD_ID

# Basketball detection filters
def filter_basketball_classes(detections, include_court_elements: bool = True):
    """
    Filter detections to basketball-relevant classes
    
    Args:
        detections: List of Detection objects
        include_court_elements: Whether to include hoop/backboard
        
    Returns:
        Filtered list of basketball-relevant detections
    """
    basketball_classes = [PLAYER_ID, BALL_ID, REF_ID]
    if include_court_elements:
        basketball_classes.extend([HOOP_ID, BACKBOARD_ID])
    
    return [det for det in detections if det.class_id in basketball_classes]

def get_basketball_detection_summary(detections) -> dict:
    """
    Get summary of basketball detections
    
    Args:
        detections: List of Detection objects
        
    Returns:
        Dictionary with counts of each basketball class
    """
    summary = {
        'players': 0,
        'ball': 0,
        'referees': 0,
        'court_elements': 0,
        'total': len(detections)
    }
    
    for det in detections:
        if det.class_id == PLAYER_ID:
            summary['players'] += 1
        elif det.class_id == BALL_ID:
            summary['ball'] += 1
        elif det.class_id == REF_ID:
            summary['referees'] += 1
        elif det.class_id in [HOOP_ID, BACKBOARD_ID]:
            summary['court_elements'] += 1
            
    return summary
