"""
YOLO-based object detection
"""

import numpy as np
import torch
import supervision as sv
from typing import List, Optional
from ultralytics import YOLO

from core import Detection, Detector


class YoloDetector(Detector):
    """YOLO-based object detector for basketball analytics"""
    
    def __init__(self, model_path: str, device: str = 'cuda', confidence: float = 0.2):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on
            confidence: Minimum confidence threshold
        """
        self.model = YOLO(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.confidence = confidence
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame"""
        results = self.model.predict(
            frame, 
            device=self.device, 
            conf=self.confidence, 
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            sv_detections = sv.Detections.from_ultralytics(result)
            sv_detections = sv_detections.with_nms(threshold=0.5, class_agnostic=True)
            
            for i in range(len(sv_detections)):
                detection = Detection(
                    bbox=sv_detections.xyxy[i],
                    confidence=sv_detections.confidence[i],
                    class_id=sv_detections.class_id[i],
                    mask=None  # YOLO doesn't provide masks
                )
                detections.append(detection)
                
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in multiple frames"""
        if not frames:
            return []
            
        try:
            # YOLO supports batch inference
            results = self.model.predict(
                frames, 
                device=self.device, 
                conf=self.confidence, 
                verbose=False
            )
            
            batch_detections = []
            for result in results:
                frame_detections = []
                sv_detections = sv.Detections.from_ultralytics(result)
                sv_detections = sv_detections.with_nms(threshold=0.5, class_agnostic=True)
                
                for i in range(len(sv_detections)):
                    detection = Detection(
                        bbox=sv_detections.xyxy[i],
                        confidence=sv_detections.confidence[i],
                        class_id=sv_detections.class_id[i],
                        mask=None
                    )
                    frame_detections.append(detection)
                    
                batch_detections.append(frame_detections)
                
            # Clean up
            del results
            
        except Exception as e:
            print(f"Batch inference failed: {e}. Falling back to single frame processing.")
            batch_detections = []
            for frame in frames:
                batch_detections.append(self.detect(frame))
                
        return batch_detections
