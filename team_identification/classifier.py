"""
Unified team classification interface
"""

import numpy as np
import cv2  
import torch  
from collections import deque  
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
import gc

# Missing imports that need to be added based on your code
try:
    from transformers import SiglipVisionModel, AutoProcessor
    SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"  # Default model path
    SIGLIP_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).info(f"SiglipVisionModel not available: {e}")
    SiglipVisionModel = None
    AutoProcessor = None
    SIGLIP_MODEL_PATH = None
    SIGLIP_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).info(f"UMAP not available: {e}")
    umap = None
    UMAP_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).info(f"scikit-learn not available: {e}")
    KMeans = None
    SKLEARN_AVAILABLE = False

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).info(f"supervision not available: {e}")
    sv = None
    SUPERVISION_AVAILABLE = False

from core import Track, TeamClassifier
from .ml_classifier import MLTeamClassifier
from .color_classifier import ColorBasedClassifier
from .basketball_rules import BasketballTeamBalancer
from .crop_manager import CropManager


def create_batches(items, batch_size):
    """Helper function to create batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class TeamClassifierInterface(TeamClassifier):
    """Base interface for team classifiers"""

    def __init__(self):
        self.initialized = False

    def is_initialized(self) -> bool:
        """Check if classifier is ready"""
        return self.initialized


class UnifiedTeamClassifier(TeamClassifierInterface):
    """Unified team classifier that combines ML and color-based approaches"""
    def __init__(self,
                 use_ml: bool = True,
                 use_color_fallback: bool = True,
                 enforce_basketball_rules: bool = True,
                 device: str = 'cuda',
                 max_crops: int = 5000):
        """
        Initialize unified team classifier

        Args:
            use_ml: Whether to use ML-based classification
            use_color_fallback: Whether to use color-based fallback
            enforce_basketball_rules: Whether to enforce 5v5 balancing
            device: Device for ML inference
            max_crops: Maximum crops to store
        """
        super().__init__()

        self.use_ml = use_ml
        self.use_color_fallback = use_color_fallback
        self.enforce_basketball_rules = enforce_basketball_rules
        self.device = device
        self.batch_size = 32

        # Initialize components
        self.crop_manager = CropManager(max_crops=max_crops)

        if use_ml and SIGLIP_AVAILABLE and UMAP_AVAILABLE and SKLEARN_AVAILABLE:
            try:
                self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
                self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
                self.reducer = umap.UMAP(n_components=3, random_state=42)
                self.cluster_model = KMeans(n_clusters=2, random_state=42)
 
                # BASKETBALL ENHANCEMENT: Add brightness threshold tracking
                self.brightness_threshold_cache = None
                self.team_brightness_history = deque(maxlen=30)
 
                logging.getLogger(__name__).info("ML TeamClassifier initialized successfully")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to initialize ML classifier: {e}")
                self.features_model = None
        else:
            self.features_model = None
            _log = logging.getLogger(__name__)
            if not SIGLIP_AVAILABLE:
                _log.info("SiglipVisionModel not available - install transformers")
            if not UMAP_AVAILABLE:
                _log.info("UMAP not available - install umap-learn")
            if not SKLEARN_AVAILABLE:
                _log.info("scikit-learn not available - install scikit-learn")
 
        if use_color_fallback:
            self.color_classifier = ColorBasedClassifier()
        else:
            self.color_classifier = None
 
        if enforce_basketball_rules:
            self.team_balancer = BasketballTeamBalancer()
        else:
            self.team_balancer = None
 
        # Statistics
        self.classification_stats = {
            'ml_classifications': 0,
            'color_classifications': 0,
            'balance_corrections': 0
        }
 
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features using SiglipVisionModel with aggressive memory optimization"""
        if len(crops) == 0:
            return np.array([]).reshape(0, -1)
 
        # Limit crops to prevent memory issues
        max_crops = min(len(crops), 500)  # Hard limit
        if len(crops) > max_crops:
            logging.getLogger(__name__).warning(f"Limiting crops from {len(crops)} to {max_crops} to prevent memory issues")
            crops = crops[:max_crops]
 
        logging.getLogger(__name__).debug(f"Extracting features from {len(crops)} crops")
 
        if self.features_model is None or not SUPERVISION_AVAILABLE:
            logging.getLogger(__name__).debug("ML components not available")
            return np.array([]).reshape(0, -1)
 
        # Convert crops to PIL format with memory management
        crops_pil = []
        try:
            for i, crop in enumerate(crops):
                if i % 100 == 0 and i > 0:  # Cleanup every 100 crops
                    gc.collect()
                pil_crop = sv.cv2_to_pillow(crop)
                crops_pil.append(pil_crop)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to convert crops to PIL: {e}")
            return np.array([]).reshape(0, -1)
 
        # Use smaller batch size to prevent OOM
        small_batch_size = min(self.batch_size, 16)  # Reduced batch size
        batches = create_batches(crops_pil, small_batch_size)
 
        data = []
        processed_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(batches, desc='🔍 Extracting features')):
                try:
                    # Process batch
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                    outputs = self.features_model(**inputs)
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                    data.append(embeddings)
                    
                    processed_batches += 1
                    
                    # Aggressive cleanup after each batch
                    del inputs, outputs, embeddings
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection every few batches
                    if processed_batches % 3 == 0:
                        gc.collect()
                        
                    # Check memory and break if too high
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        if memory_used > 2000:  # 2GB limit
                            logging.getLogger(__name__).warning(f"GPU memory limit reached ({memory_used:.1f}MB), stopping feature extraction")
                            break
                        
                except Exception as e:
                    logging.warning(f"Feature extraction failed for batch {i}: {e}")
                    # Force cleanup on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
 
        # Clear PIL crops immediately
        del crops_pil
        gc.collect()
 
        if data:
            try:
                features = np.concatenate(data)
                logging.getLogger(__name__).debug(f"Feature extraction complete - shape: {features.shape}")
                
                # Clear data list
                del data
                gc.collect()
                
                return features
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to concatenate features: {e}")
                del data
                gc.collect()
                return np.array([]).reshape(0, -1)
        else:
            logging.getLogger(__name__).debug("Feature extraction failed - no valid features extracted")
            return np.array([]).reshape(0, -1)
 
    def fit(self, crops: List[np.ndarray]) -> None:
        """Train classifier on player crops"""
        if len(crops) < 10:
            logging.warning(f"Not enough crops for training: {len(crops)}")
            return
 
        # Store crops for later training
        for crop in crops:
            self.crop_manager.add_crop(crop)
 
        # Try to initialize classifiers
        if self.crop_manager.get_crop_count() >= 50:
            if self.features_model and not self.initialized:
                try:
                    training_crops = self.crop_manager.get_training_crops()
 
                    # Extract features
                    data = self.extract_features(training_crops)
                    if data.size == 0:
                        raise ValueError("Failed to extract features from crops")
 
                    logging.debug(f"Extracted features shape: {data.shape}")
 
                    # UMAP dimensionality reduction
                    logging.debug("Applying UMAP dimensionality reduction")
                    projections = self.reducer.fit_transform(data)
                    logging.debug(f"UMAP projections shape: {projections.shape}")
 
                    # KMeans clustering
                    logging.debug("Training KMeans clustering")
                    self.cluster_model.fit(projections)
 
                    self.initialized = True
                    logging.info("ML classifier initialized successfully")
                except Exception as e:
                    logging.warning(f"ML classifier training failed: {e}")
 
            if self.color_classifier and not self.color_classifier.is_initialized():
                try:
                    training_crops = self.crop_manager.get_training_crops()
                    self.color_classifier.fit(training_crops)
                    if not self.initialized:
                        self.initialized = True
                    logging.info("Color classifier initialized successfully")
                except Exception as e:
                    logging.warning(f"Color classifier training failed: {e}")
 
    def classify(self, tracks: List, frame: np.ndarray) -> Dict[int, int]:
        """
        Classify tracks into teams - MAINTAINING ORIGINAL INTERFACE
 
        Args:
            tracks: List of tracks to classify
            frame: Current video frame
 
        Returns:
            Dictionary mapping track_id to team_id
        """
        if not tracks:
            return {}
 
        # Extract crops for active tracks
        crops = []
        track_ids = []
 
        for track in tracks:
            if hasattr(track, 'current_bbox') and track.current_bbox is not None:
                crop = self.crop_manager.extract_crop(frame, track.current_bbox)
                if crop is not None:
                    crops.append(crop)
                    track_ids.append(track.id)
            elif hasattr(track, 'last_bbox'):
                crop = self.crop_manager.extract_crop(frame, track.last_bbox)
                if crop is not None:
                    crops.append(crop)
                    track_ids.append(track.id)
 
        if not crops:
            return {}
 
        # Try ML classification first
        predictions = None
        confidence = 0.0
 
        if self.features_model and self.initialized:
            try:
                predictions = self.predict(crops)
                confidence = 0.8  # Default confidence for ML
                self.classification_stats['ml_classifications'] += len(predictions)
            except Exception as e:
                logging.warning(f"ML classification failed: {e}")
                predictions = None
 
        # Fallback to color-based classification
        if predictions is None and self.color_classifier:
            try:
                predictions = self.color_classifier.predict(crops)
                self.classification_stats['color_classifications'] += len(predictions)
            except Exception as e:
                logging.warning(f"Color classification failed: {e}")
                predictions = np.zeros(len(crops), dtype=int)
 
        if predictions is None:
            # Default assignment
            predictions = np.array([i % 2 for i in range(len(crops))])
 
        # Apply basketball rules for team balancing
        if self.team_balancer and self.enforce_basketball_rules:
            original = predictions.copy()
            predictions = self.team_balancer.balance_teams(predictions, crops)
 
            if not np.array_equal(original, predictions):
                self.classification_stats['balance_corrections'] += 1
 
        # Create result mapping
        result = {}
        for track_id, team_id in zip(track_ids, predictions):
            result[track_id] = int(team_id)
 
        return result
 
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team labels - ORIGINAL METHOD WITH DEBUG"""
        if len(crops) == 0:
            return np.array([])
 
        logging.debug(f"Predicting teams for {len(crops)} crops")
 
        # Extract features
        data = self.extract_features(crops)
        if data.size == 0:
            logging.debug("No features extracted for prediction")
            return np.array([])
 
        # Transform with UMAP
        projections = self.reducer.transform(data)
 
        # Predict with KMeans
        predictions = self.cluster_model.predict(projections)
 
        logging.debug(f"Raw ML predictions: T0={np.sum(predictions == 0)}, T1={np.sum(predictions == 1)}")
 
        # BASKETBALL ENHANCEMENT: Apply brightness-based team balancing
        if self.enforce_basketball_rules:
            predictions = self.balance_teams_with_brightness(predictions, crops)
 
        return predictions
 
    def balance_teams_with_brightness(self, predictions: np.ndarray, crops: List[np.ndarray]) -> np.ndarray:
        """
        BASKETBALL ENHANCEMENT: Ensure 5v5 team balance using brightness correction
        """
        if len(crops) == 0 or len(predictions) == 0:
            return predictions
 
        team_0_count = np.sum(predictions == 0)
        team_1_count = np.sum(predictions == 1)
 
        logging.debug(f"Team balance check: T0={team_0_count}, T1={team_1_count}")
 
        # BASKETBALL: If teams are balanced (4-6 players each), keep ML predictions
        if 4 <= team_0_count <= 6 and 4 <= team_1_count <= 6:
            return predictions
 
        # BASKETBALL: If severely imbalanced, apply brightness correction
        if team_0_count < 3 or team_1_count < 3 or team_0_count > 7 or team_1_count > 7:
            logging.debug(f"Teams imbalanced ({team_0_count}v{team_1_count}), applying brightness correction")
            return self._apply_brightness_correction(crops)
 
        return predictions
 
    def _apply_brightness_correction(self, crops: List[np.ndarray]) -> np.ndarray:
        """BASKETBALL: Apply brightness-based team correction for 5v5 balance"""
        brightnesses = []
        valid_indices = []
 
        # Calculate brightness for each player
        for i, crop in enumerate(crops):
            try:
                brightness = self.calculate_brightness(crop)
                brightnesses.append(brightness)
                valid_indices.append(i)
                logging.debug(f"Player {i}: brightness = {brightness:.1f}")
            except Exception as e:
                logging.warning(f"Error calculating brightness for player {i}: {e}")
                continue
 
        if len(brightnesses) < 6:  # Need at least 6 players for meaningful split
            logging.debug(f"Not enough valid players ({len(brightnesses)}) for brightness correction")
            # Return alternating assignment as fallback
            return np.array([i % 2 for i in range(len(crops))])
 
        # BASKETBALL: Split into 5v5 using brightness threshold
        brightnesses_array = np.array(brightnesses)
        threshold = np.median(brightnesses_array)
 
        # Apply the correction
        new_team_assignments = np.zeros(len(crops), dtype=int)
 
        for i, valid_idx in enumerate(valid_indices):
            if i < len(brightnesses_array):
                if brightnesses_array[i] > threshold:
                    new_team_assignments[valid_idx] = 0  # Brighter team
                else:
                    new_team_assignments[valid_idx] = 1  # Darker team
 
        final_team_0 = np.sum(new_team_assignments == 0)
        final_team_1 = np.sum(new_team_assignments == 1)
        logging.debug(f"Brightness correction applied: T0={final_team_0}, T1={final_team_1}")
 
        return new_team_assignments
 
    def calculate_brightness(self, crop: np.ndarray) -> float:
        """Calculate brightness of jersey area for team balancing."""
        from utils.image_utils import calculate_jersey_brightness
        return calculate_jersey_brightness(crop)
 
    def get_statistics(self) -> Dict[str, int]:
        """Get classification statistics"""
        return self.classification_stats.copy()
 
    def get_basketball_statistics(self) -> Dict[str, int]:
        """Get basketball-specific statistics (alias for compatibility)"""
        return self.get_statistics()
 
    def update_with_feedback(self, track_id: int, correct_team_id: int):
        """Update classifier with correction feedback"""
        # Store feedback for adaptive learning
        pass
