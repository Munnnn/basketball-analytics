"""
Appearance feature extraction for tracking
"""

import numpy as np
import torch
import cv2
from typing import List, Optional
import logging

from core import Detection


class AppearanceExtractor:
    """Extract appearance features using deep learning models"""
    
    def __init__(self, model_name: str = 'resnet18', device: str = 'cuda'):
        """
        Initialize appearance extractor
        
        Args:
            model_name: Pretrained model to use
            device: Device for inference
        """
        self.device = device
        self.model = None
        self.transform = None
        
        try:
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Load pretrained model
            if model_name == 'resnet18':
                try:
                    from torchvision.models import ResNet18_Weights
                    self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                except (ImportError, AttributeError):
                    self.model = models.resnet18(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
            # Remove classification head
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(device)
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.available = True
            
        except ImportError:
            logging.warning("torchvision not available. Appearance features disabled.")
            self.available = False
            
    def extract_single(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """Extract features for single detection"""
        if not self.available:
            return None
            
        # Extract crop
        x1, y1, x2, y2 = detection.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = frame[y1:y2, x1:x2]
        
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Transform
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor).squeeze().cpu().numpy()
                
            return features
            
        except Exception as e:
            logging.warning(f"Feature extraction failed: {e}")
            return None
            
    def extract_batch(self, frame: np.ndarray, detections: List[Detection]) -> Optional[np.ndarray]:
        """Extract features for multiple detections"""
        if not self.available or not detections:
            return None
            
        features_list = []
        valid_indices = []
        
        # Prepare batch
        batch_tensors = []
        for i, detection in enumerate(detections):
            # Extract crop
            x1, y1, x2, y2 = detection.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                try:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(crop_rgb)
                    batch_tensors.append(tensor)
                    valid_indices.append(i)
                except Exception:
                    continue
                    
        if not batch_tensors:
            return None
            
        try:
            # Process batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                features = self.model(batch).squeeze().cpu().numpy()
                
            if features.ndim == 1:
                features = features.reshape(1, -1)
                
            # Create full feature array with None for invalid detections
            full_features = np.zeros((len(detections), features.shape[1]))
            for i, idx in enumerate(valid_indices):
                full_features[idx] = features[i]
                
            return full_features
            
        except Exception as e:
            logging.warning(f"Batch feature extraction failed: {e}")
            return None
