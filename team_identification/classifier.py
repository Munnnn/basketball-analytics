"""
Unified team classification interface with enhanced basketball 5v5 balancing
"""

import numpy as np
import cv2
import torch
import umap
from typing import List, Dict, Optional, Tuple
import logging
from collections import deque, defaultdict
import time
import gc
import os

from core import Track, TeamClassifier
from .ml_classifier import MLTeamClassifier
from .color_classifier import ColorBasedClassifier
from .basketball_rules import BasketballTeamBalancer
from .crop_manager import CropManager

# Add enhanced components from pasted version
try:
    from transformers import AutoProcessor, SiglipVisionModel
    from sklearn.cluster import KMeans
    SIGLIP_AVAILABLE = True
    SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
except ImportError:
    SIGLIP_AVAILABLE = False
    logging.warning("SigLIP not available for enhanced team classification")


class RollingCropBuffer:
    """Manages rolling buffer of new crops for adaptive retraining"""
    def __init__(self, max_crops: int = 1000):
        self.max_crops = max_crops
        self.crops_data = []
        self.crop_count = 0

    def add_crop(self, crop: np.ndarray, frame_num: int, player_idx: int) -> bool:
        if crop is not None and crop.size > 0:
            if self.crop_count >= self.max_crops:
                self.crops_data.pop(0)
                self.crop_count -= 1
            self.crops_data.append((crop.copy(), frame_num, player_idx))
            self.crop_count += 1
            return True
        return False

    def get_crops(self) -> List[np.ndarray]:
        return [crop for crop, _, _ in self.crops_data]

    def clear(self):
        self.crops_data.clear()
        self.crop_count = 0


class ConfidenceTracker:
    """Tracks prediction confidence over time for retraining decisions"""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.confidences = deque(maxlen=window_size)

    def add_confidence(self, confidence: float):
        self.confidences.append(confidence)

    def is_consistently_low(self, threshold: float = 0.6, min_count: int = 20) -> bool:
        return len(self.confidences) >= min_count and sum(1 for conf in self.confidences if conf < threshold) >= min_count


class EnhancedTeamClassifier:
    """Enhanced team classifier with SigLIP features and basketball 5v5 balancing"""
    def __init__(self, device: str = 'cuda', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.initialized = False

        if SIGLIP_AVAILABLE:
            try:
                self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
                self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
                self.reducer = umap.UMAP(n_components=3, random_state=42)
                self.cluster_model = KMeans(n_clusters=2, random_state=42)
                print("✅ Enhanced TeamClassifier initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize enhanced classifier: {e}")
                self.features_model = None
        else:
            self.features_model = None

    def calculate_brightness(self, crop: np.ndarray) -> float:
        """Enhanced brightness calculation using proper luminance formula"""
        if crop is None or crop.size == 0:
            return 128.0
        h, w = crop.shape[:2]
        jersey_area = crop[h//6:h//2, w//4:3*w//4] if h > 0 and w > 0 else crop
        if jersey_area.size == 0:
            jersey_area = crop
        if len(jersey_area.shape) == 3:
            avg_color = jersey_area.mean(axis=(0, 1))
            brightness = 0.299 * avg_color[2] + 0.587 * avg_color[1] + 0.114 * avg_color[0]
        else:
            brightness = jersey_area.mean()
        return float(brightness)

    def balance_teams_with_brightness(self, predictions: np.ndarray, crops: List[np.ndarray]) -> np.ndarray:
        """BASKETBALL ENHANCEMENT: Ensure 5v5 team balance using brightness correction"""
        if len(crops) == 0 or len(predictions) == 0:
            return predictions

        team_0_count = np.sum(predictions == 0)
        team_1_count = np.sum(predictions == 1)

        # If teams are balanced (4-6 players each), keep ML predictions
        if 4 <= team_0_count <= 6 and 4 <= team_1_count <= 6:
            return predictions

        # Apply brightness correction for imbalanced teams
        brightnesses = [self.calculate_brightness(crop) for crop in crops]
        threshold = np.median(brightnesses)
        
        new_predictions = np.zeros(len(crops), dtype=int)
        for i, brightness in enumerate(brightnesses):
            new_predictions[i] = 0 if brightness > threshold else 1
            
        return new_predictions


class UnifiedTeamClassifier(TeamClassifier):
    """Unified team classifier with enhanced features"""
    def __init__(self, use_ml: bool = True, use_color_fallback: bool = True, 
                 enforce_basketball_rules: bool = True, basketball_5v5_balancing: bool = True,
                 device: str = 'cuda', max_crops: int = 5000):
        super().__init__()
        self.use_ml = use_ml
        self.basketball_5v5_balancing = basketball_5v5_balancing
        
        # Initialize enhanced components
        self.crop_manager = CropManager(max_crops=max_crops)
        self.rolling_crop_buffer = RollingCropBuffer(max_crops=1000)
        self.confidence_tracker = ConfidenceTracker(window_size=50)
        
        # Initialize classifiers
        if use_ml and SIGLIP_AVAILABLE:
            self.enhanced_classifier = EnhancedTeamClassifier(device=device)
        else:
            self.enhanced_classifier = None
            
        if use_color_fallback:
            self.color_classifier = ColorBasedClassifier()
        else:
            self.color_classifier = None
            
        if enforce_basketball_rules:
            self.team_balancer = BasketballTeamBalancer()
        else:
            self.team_balancer = None

    def classify_basketball_teams(self, tracks: List[Track], frame: np.ndarray, 
                                enforce_5v5: bool = True) -> Dict[int, int]:
        """Basketball-specific team classification with 5v5 enforcement"""
        return self.classify(tracks, frame)

    def get_basketball_statistics(self) -> Dict:
        """Get basketball-specific classification statistics"""
        return {
            'classifier_type': 'enhanced' if self.enhanced_classifier else 'standard',
            'confidence_average': self.confidence_tracker.get_average() if hasattr(self.confidence_tracker, 'get_average') else 0.0,
            'basketball_5v5_enabled': self.basketball_5v5_balancing
        }
