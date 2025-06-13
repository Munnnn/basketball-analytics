"""
Video input/output operations
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple


class VideoWriter:
    """Video writer with basketball-optimized codec support and proper data type handling"""

    def __init__(self,
                 output_path: str,
                 fps: float = 30.0,
                 width: int = 1920,
                 height: int = 1080,
                 basketball_codec: bool = True):
        """
        Initialize video writer with proper frame format handling

        Args:
            output_path: Output video file path
            fps: Frames per second
            width: Frame width
            height: Frame height
            basketball_codec: Use basketball-optimized codec
        """
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.basketball_codec = basketball_codec

        # Choose codec
        if basketball_codec:
            # Basketball-optimized codec for better quality
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            # Default codec
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Initialize video writer
        self.writer = cv2.VideoWriter(
            output_path,
            self.fourcc,
            fps,
            (width, height),
            True  # isColor=True for color frames
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        self.frame_count = 0
        self.logger = logging.getLogger(__name__)

    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video with proper data type conversion

        Args:
            frame: Input frame (BGR format)
        """
        if frame is None:
            self.logger.warning("Attempted to write None frame")
            return

        try:
            # Convert frame to proper format for OpenCV VideoWriter
            processed_frame = self._prepare_frame_for_writing(frame)

            # Write the frame
            self.writer.write(processed_frame)
            self.frame_count += 1

        except Exception as e:
            self.logger.error(f"Error writing frame {self.frame_count}: {e}")

    def _prepare_frame_for_writing(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for OpenCV VideoWriter by ensuring proper data type and format

        Args:
            frame: Input frame

        Returns:
            Frame ready for VideoWriter (uint8, BGR, correct dimensions)
        """
        # Handle None frames
        if frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Convert to numpy array if needed
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Handle empty frames
        if frame.size == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fix data type issues
        if frame.dtype != np.uint8:
            # Handle different data types
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                # Check if values are normalized [0,1] or standard [0,255]
                if frame.max() <= 1.0:
                    # Normalized values - scale to [0,255]
                    frame = (frame * 255).astype(np.uint8)
                else:
                    # Already in [0,255] range but wrong type
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                # Other data types - convert to uint8
                frame = frame.astype(np.uint8)

        # Ensure frame has correct number of channels
        if len(frame.shape) == 2:
            # Grayscale - convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3:
            if frame.shape[2] == 1:
                # Single channel - convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                # RGBA - convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3:
                # Already BGR - ensure correct order
                # OpenCV expects BGR, but sometimes we get RGB
                # We'll assume it's already in the correct format
                pass
            else:
                self.logger.warning(f"Unexpected number of channels: {frame.shape[2]}")
                # Take first 3 channels
                frame = frame[:, :, :3]

        # Resize frame to match expected dimensions
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Final validation with better error handling
        if frame.dtype != np.uint8:
            self.logger.error(f"Frame dtype is {frame.dtype}, expected uint8 - attempting conversion")
            frame = frame.astype(np.uint8)

        if len(frame.shape) != 3:
            self.logger.error(f"Frame shape is {frame.shape}, expected 3D - creating default frame")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if frame.shape[2] != 3:
            self.logger.error(f"Frame has {frame.shape[2]} channels, expected 3 - taking first 3 channels")
            if frame.shape[2] > 3:
                frame = frame[:, :, :3]
            else:
                # Convert single/dual channel to 3-channel
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.shape[2] == 1 else frame

        if frame.shape[:2] != (self.height, self.width):
            self.logger.warning(f"Frame size is {frame.shape[:2]}, expected ({self.height}, {self.width}) - resizing")
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def close(self):
        """Close video writer and release resources"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()
            self.logger.info(f"Video saved: {self.output_path} ({self.frame_count} frames)")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def get_frame_count(self) -> int:
        """Get number of frames written"""
        return self.frame_count

    def is_opened(self) -> bool:
        """Check if video writer is opened"""
        return self.writer.isOpened() if hasattr(self, 'writer') else False


class VideoReader:
    """Video reader for basketball analysis"""

    def __init__(self, video_path: str):
        """Initialize video reader"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.logger = logging.getLogger(__name__)

    def get_info(self) -> dict:
        """Get video information"""
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def read_frames(self, start_frame: int = 0, end_frame: Optional[int] = None):
        """Generator to read frames from video"""
        # Set starting position
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            if end_frame is not None and frame_idx >= end_frame:
                break

            yield frame, frame_idx
            frame_idx += 1

    def close(self):
        """Close video reader"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class StreamingWriter:
    """Streaming writer for basketball analytics"""

    def __init__(self, output_path: str):
        """Initialize streaming writer"""
        self.output_path = output_path
        self.data = []

    def write(self, data: dict):
        """Write data to stream"""
        self.data.append(data)

    def close(self):
        """Close and save streaming data"""
        import pickle
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
