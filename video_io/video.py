
"""
Video input/output operations
"""

import cv2
import numpy as np
from typing import Dict, Generator, Tuple, Optional, List
import logging


class VideoReader:
    """Read video frames efficiently"""

    def __init__(self, video_path: str):
        """
        Initialize video reader

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.logger = logging.getLogger(__name__)

    def get_info(self) -> Dict:
        """Get video information"""
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': self.cap.get(cv2.CAP_PROP_FOURCC)
        }

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def read_frames(self, start: int = 0,
                   end: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Read frames generator

        Args:
            start: Start frame index
            end: End frame index (None for all)

        Yields:
            Tuple of (frame_index, frame)
        """
        if end is None:
            end = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for frame_idx in range(start, end):
            ret, frame = self.cap.read()
            if not ret:
                break

            yield frame_idx, frame

    def close(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()


class VideoWriter:
    """Write video frames efficiently"""

    def __init__(self, output_path: str, fps: float,
                 width: int, height: int, codec: str = 'mp4v'):
        """
        Initialize video writer

        Args:
            output_path: Output video path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec
        """
        self.output_path = output_path

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")

        self.frame_count = 0
        self.logger = logging.getLogger(__name__)

    def write_frame(self, frame: np.ndarray):
        """Write single frame"""
        self.writer.write(frame)
        self.frame_count += 1

    def write_frames(self, frames: List[np.ndarray]):
        """Write multiple frames"""
        for frame in frames:
            self.write_frame(frame)

    def close(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
            self.logger.info(f"Video saved: {self.output_path} ({self.frame_count} frames)")
