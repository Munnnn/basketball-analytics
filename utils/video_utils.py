"""
Video utility functions
"""

import cv2
import numpy as np
from datetime import timedelta
from typing import Tuple, Optional


def get_video_info(video_path: str) -> dict:
    """Get video metadata"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
        
    info = {
        'path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return info


def frame_to_time(frame_idx: int, fps: float) -> str:
    """Convert frame index to time string"""
    seconds = frame_idx / fps
    return str(timedelta(seconds=seconds)).split('.')[0]


def time_to_frame(time_seconds: float, fps: float) -> int:
    """Convert time to frame index"""
    return int(time_seconds * fps)


def calculate_fps(timestamps: list) -> float:
    """Calculate FPS from timestamps"""
    if len(timestamps) < 2:
        return 0.0
    
    time_diff = timestamps[-1] - timestamps[0]
    frame_count = len(timestamps) - 1
    
    return frame_count / time_diff if time_diff > 0 else 0.0


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame maintaining aspect ratio"""
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    
    # New dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad if needed
    if new_w != target_w or new_h != target_h:
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    
    return resized
