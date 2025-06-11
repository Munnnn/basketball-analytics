"""
Mask generation for detected objects with basketball optimizations
"""

import numpy as np
import cv2
import torch
from typing import List, Optional
import logging

from core import Detection
from core.interfaces import MaskGenerator as MaskGeneratorInterface
from core.constants import PLAYER_ID, BALL_ID, REF_ID

# Try to import SAM components
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM2 not available. Using fallback mask generation.")


class MaskGenerator(MaskGeneratorInterface):
    """Base mask generator with automatic backend selection and basketball optimizations"""

    def __init__(self, 
                 backend: str = 'auto', 
                 basketball_optimized: bool = True,
                 basketball_fallback: bool = True,
                 **kwargs):
        """
        Initialize mask generator with basketball optimizations
        
        Args:
            backend: Backend to use ('auto', 'edgetam', 'simple')
            basketball_optimized: Enable basketball-specific optimizations
            basketball_fallback: Enable basketball fallback mode
            **kwargs: Additional arguments for specific generators
        """
        self.basketball_optimized = basketball_optimized
        self.basketball_fallback = basketball_fallback
        
        if backend == 'auto':
            if SAM_AVAILABLE:
                self._generator = EdgeTAMMaskGenerator(
                    basketball_optimized=basketball_optimized, **kwargs
                )
            else:
                self._generator = SimpleMaskGenerator(
                    basketball_optimized=basketball_optimized
                )
        elif backend == 'edgetam':
            self._generator = EdgeTAMMaskGenerator(
                basketball_optimized=basketball_optimized, **kwargs
            )
        else:
            self._generator = SimpleMaskGenerator(
                basketball_optimized=basketball_optimized
            )
            
        # Basketball-specific statistics
        self.basketball_stats = {
            'total_masks_generated': 0,
            'player_masks_generated': 0,
            'ball_masks_generated': 0,
            'edgetam_successes': 0,
            'fallback_uses': 0
        }

    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate masks with basketball optimizations"""
        if self.basketball_optimized:
            return self.generate_basketball_masks(frame, detections)
        else:
            return self._generator.generate_masks(frame, detections)

    def generate_basketball_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate masks optimized for basketball analysis"""
        if not detections:
            return []
            
        # Separate basketball-relevant detections for optimized processing
        basketball_detections = []
        other_detections = []
        
        for i, det in enumerate(detections):
            if det.class_id in [PLAYER_ID, BALL_ID, REF_ID]:
                basketball_detections.append((i, det))
            else:
                other_detections.append((i, det))
        
        # Initialize results
        masks = [None] * len(detections)
        
        # Process basketball detections with high quality
        if basketball_detections:
            basketball_indices, basketball_dets = zip(*basketball_detections)
            basketball_masks = self._generator.generate_masks(frame, list(basketball_dets))
            
            for idx, mask in zip(basketball_indices, basketball_masks):
                masks[idx] = mask
                
            # Update basketball statistics
            self.basketball_stats['total_masks_generated'] += len(basketball_masks)
            self.basketball_stats['player_masks_generated'] += len([
                d for d in basketball_dets if d.class_id == PLAYER_ID
            ])
            self.basketball_stats['ball_masks_generated'] += len([
                d for d in basketball_dets if d.class_id == BALL_ID
            ])
        
        # Process other detections with simple masks for efficiency
        if other_detections:
            other_indices, other_dets = zip(*other_detections)
            simple_generator = SimpleMaskGenerator(basketball_optimized=False)
            other_masks = simple_generator.generate_masks(frame, list(other_dets))
            
            for idx, mask in zip(other_indices, other_masks):
                masks[idx] = mask
        
        return masks

    def get_basketball_statistics(self) -> dict:
        """Get basketball mask generation statistics"""
        return self.basketball_stats.copy()


class EdgeTAMMaskGenerator(MaskGeneratorInterface):
    """EdgeTAM-based mask generator with basketball optimizations"""

    def __init__(self, 
                 checkpoint_path: str = "./checkpoints/edgetam.pt",
                 model_cfg: str = "edgetam.yaml",
                 confidence_threshold: float = 0.2,
                 basketball_optimized: bool = True):
        """Initialize EdgeTAM mask generator with basketball enhancements"""
        self.confidence_threshold = confidence_threshold
        self.basketball_optimized = basketball_optimized
        self.predictor = None

        # Basketball-specific parameters
        if basketball_optimized:
            self.basketball_confidence_adjustments = {
                PLAYER_ID: 0.1,  # Lower threshold for players (they're important)
                BALL_ID: 0.15,   # Even lower for ball (often small/occluded)
                REF_ID: 0.05     # Slight adjustment for referees
            }
        
        if SAM_AVAILABLE:
            try:
                self.predictor = SAM2ImagePredictor(
                    build_sam2(model_cfg, checkpoint_path)
                )
                print("✅ EdgeTAM mask generator initialized with basketball optimizations")
            except Exception as e:
                logging.warning(f"EdgeTAM initialization failed: {e}")
                self.predictor = None
        else:
            print("⚠️ EdgeTAM not available, using bbox fallback")

    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate masks for detections using EdgeTAM with basketball optimizations"""
        if not detections:
            return []

        masks = []

        # Fallback to bbox masks if EdgeTAM not available
        if self.predictor is None or frame is None:
            return self._generate_basketball_bbox_masks(frame, detections)

        try:
            # Set image for EdgeTAM
            self.predictor.set_image(frame)

            # Process in batches for memory efficiency
            batch_size = 8  # Smaller batches for basketball processing
            for i in range(0, len(detections), batch_size):
                batch_detections = detections[i:i + batch_size]
                batch_boxes = np.array([det.bbox for det in batch_detections])

                try:
                    # Predict masks for batch
                    predicted_masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=batch_boxes,
                        multimask_output=False,
                    )

                    # Process batch results with basketball optimizations
                    for j, (mask, score, det) in enumerate(zip(predicted_masks, scores, batch_detections)):
                        # Apply basketball-specific confidence adjustments
                        adjusted_threshold = self.confidence_threshold
                        if self.basketball_optimized and det.class_id in self.basketball_confidence_adjustments:
                            adjusted_threshold -= self.basketball_confidence_adjustments[det.class_id]

                        if score > adjusted_threshold:
                            processed_mask = mask.astype(bool)
                            if processed_mask.ndim > 2:
                                processed_mask = processed_mask.squeeze()
                            
                            # Basketball-specific mask post-processing
                            if self.basketball_optimized:
                                processed_mask = self._post_process_basketball_mask(
                                    processed_mask, det, frame
                                )
                            
                            masks.append(processed_mask)
                        else:
                            # Fallback to bbox mask for low confidence
                            masks.append(self._create_basketball_bbox_mask(frame, det.bbox, det.class_id))

                except Exception as e:
                    logging.warning(f"EdgeTAM batch prediction failed: {e}")
                    # Fallback for entire batch
                    for det in batch_detections:
                        masks.append(self._create_basketball_bbox_mask(frame, det.bbox, det.class_id))

        except Exception as e:
            logging.warning(f"EdgeTAM prediction failed: {e}")
            masks = self._generate_basketball_bbox_masks(frame, detections)

        return masks

    def _post_process_basketball_mask(self, mask: np.ndarray, detection: Detection, frame: np.ndarray) -> np.ndarray:
        """Post-process mask for basketball-specific improvements"""
        class_id = detection.class_id
        
        if class_id == PLAYER_ID:
            # For players, ensure mask includes jersey area properly
            mask = self._enhance_player_mask(mask, detection.bbox)
        elif class_id == BALL_ID:
            # For ball, apply morphological operations to get better shape
            mask = self._enhance_ball_mask(mask)
        
        return mask

    def _enhance_player_mask(self, mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Enhance player mask for better jersey visibility"""
        try:
            # Apply morphological closing to fill gaps in jersey
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_uint8 = mask.astype(np.uint8) * 255
            closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            return closed.astype(bool)
        except:
            return mask

    def _enhance_ball_mask(self, mask: np.ndarray) -> np.ndarray:
        """Enhance ball mask to get better circular shape"""
        try:
            # Apply morphological operations to get better circular shape
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_uint8 = mask.astype(np.uint8) * 255
            opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed.astype(bool)
        except:
            return mask

    def _generate_basketball_bbox_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """Generate basketball-optimized bbox masks as fallback"""
        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        masks = []

        for det in detections:
            masks.append(self._create_basketball_bbox_mask((h, w), det.bbox, det.class_id))

        return masks

    def _create_basketball_bbox_mask(self, shape, bbox: np.ndarray, class_id: int) -> np.ndarray:
        """Create basketball-optimized bbox mask"""
        if isinstance(shape, np.ndarray):
            h, w = shape.shape[:2]
        else:
            h, w = shape

        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if self.basketball_optimized:
            # Basketball-specific mask adjustments
            if class_id == PLAYER_ID:
                # For players, focus on upper body (jersey area)
                height = y2 - y1
                jersey_height = int(height * 0.7)  # Top 70% for jersey
                mask[y1:y1 + jersey_height, x1:x2] = True
            elif class_id == BALL_ID:
                # For ball, create circular mask
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = min((x2 - x1), (y2 - y1)) // 2
                y_coords, x_coords = np.ogrid[:h, :w]
                ball_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                mask = ball_mask
            else:
                # Standard bbox for other classes
                mask[y1:y2, x1:x2] = True
        else:
            # Standard bbox mask
            mask[y1:y2, x1:x2] = True

        return mask


class SAMMaskGenerator(EdgeTAMMaskGenerator):
    """Alias for EdgeTAM (they use the same underlying model)"""
    pass


class SimpleMaskGenerator(MaskGeneratorInterface):
    """Simple bbox-based mask generator optimized for basketball processing"""

    def __init__(self, basketball_optimized: bool = True):
        """Initialize simple mask generator with basketball optimizations"""
        self.basketball_optimized = basketball_optimized

    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate simple bbox masks with basketball optimizations"""
        if not detections:
            return []

        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        masks = []

        for det in detections:
            if self.basketball_optimized:
                mask = self._create_basketball_optimized_mask((h, w), det)
            else:
                mask = self._create_simple_mask((h, w), det.bbox)
            masks.append(mask)

        return masks

    def _create_basketball_optimized_mask(self, shape: tuple, detection: Detection) -> np.ndarray:
        """Create basketball-optimized simple mask"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        
        x1, y1, x2, y2 = detection.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        class_id = detection.class_id
        
        if class_id == PLAYER_ID:
            # For players, create jersey-focused mask (upper portion)
            height = y2 - y1
            jersey_height = int(height * 0.6)  # Top 60% for jersey
            mask[y1:y1 + jersey_height, x1:x2] = True
        elif class_id == BALL_ID:
            # For ball, create circular mask
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            radius = min((x2 - x1), (y2 - y1)) // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            ball_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
            mask = ball_mask
        else:
            # Standard bbox for other classes
            mask[y1:y2, x1:x2] = True
            
        return mask

    def _create_simple_mask(self, shape: tuple, bbox: np.ndarray) -> np.ndarray:
        """Create simple rectangular mask from bbox"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        
        return mask
