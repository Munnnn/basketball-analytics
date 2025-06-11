
"""
Mask generation for detected objects
"""

import numpy as np
import cv2
import torch
from typing import List, Optional
import logging

from core import Detection
from core.interfaces import MaskGenerator as MaskGeneratorInterface

# Try to import SAM components
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM2 not available. Using fallback mask generation.")


class MaskGenerator(MaskGeneratorInterface):
    """Base mask generator with automatic backend selection"""

    def __init__(self, backend: str = 'auto', **kwargs):
        if backend == 'auto':
            if SAM_AVAILABLE:
                self._generator = EdgeTAMMaskGenerator(**kwargs)
            else:
                self._generator = SimpleMaskGenerator()
        elif backend == 'edgetam':
            self._generator = EdgeTAMMaskGenerator(**kwargs)
        else:
            self._generator = SimpleMaskGenerator()

    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        return self._generator.generate_masks(frame, detections)


class EdgeTAMMaskGenerator(MaskGeneratorInterface):
    """EdgeTAM-based mask generator"""

    def __init__(self, checkpoint_path: str = "./checkpoints/edgetam.pt",
                 model_cfg: str = "edgetam.yaml",
                 confidence_threshold: float = 0.2):
        """Initialize EdgeTAM mask generator"""
        self.confidence_threshold = confidence_threshold
        self.predictor = None

        if SAM_AVAILABLE:
            try:
                self.predictor = SAM2ImagePredictor(
                    build_sam2(model_cfg, checkpoint_path)
                )
                print("✅ EdgeTAM mask generator initialized")
            except Exception as e:
                logging.warning(f"EdgeTAM initialization failed: {e}")
                self.predictor = None
        else:
            print("⚠️ EdgeTAM not available, using bbox fallback")

    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate masks for detections using EdgeTAM"""
        if not detections:
            return []

        masks = []

        # Fallback to bbox masks if EdgeTAM not available
        if self.predictor is None or frame is None:
            return self._generate_bbox_masks(frame, detections)

        try:
            # Set image for EdgeTAM
            self.predictor.set_image(frame)

            # Extract bboxes
            boxes = np.array([det.bbox for det in detections])

            # Predict masks
            predicted_masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

            # Process results
            for i, (mask, score) in enumerate(zip(predicted_masks, scores)):
                if score > self.confidence_threshold:
                    processed_mask = mask.astype(bool)
                    if processed_mask.ndim > 2:
                        processed_mask = processed_mask.squeeze()
                    masks.append(processed_mask)
                else:
                    # Fallback to bbox mask for low confidence
                    masks.append(self._create_bbox_mask(frame, detections[i].bbox))

        except Exception as e:
            logging.warning(f"EdgeTAM prediction failed: {e}")
            masks = self._generate_bbox_masks(frame, detections)

        return masks

    def _generate_bbox_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """Generate simple bbox masks as fallback"""
        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        masks = []

        for det in detections:
            masks.append(self._create_bbox_mask((h, w), det.bbox))

        return masks

    def _create_bbox_mask(self, shape, bbox: np.ndarray) -> np.ndarray:
        """Create a simple rectangular mask from bbox"""
        if isinstance(shape, np.ndarray):
            h, w = shape.shape[:2]
        else:
            h, w = shape

        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True

        return mask


class SAMMaskGenerator(EdgeTAMMaskGenerator):
    """Alias for EdgeTAM (they use the same underlying model)"""
    pass


class SimpleMaskGenerator(MaskGeneratorInterface):
    """Simple bbox-based mask generator for fast processing"""

    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate simple bbox masks"""
        if not detections:
            return []

        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        masks = []

        for det in detections:
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = det.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)

        return masks
