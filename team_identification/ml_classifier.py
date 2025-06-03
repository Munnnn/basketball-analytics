"""
Machine learning based team classification
"""

import numpy as np
import torch
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import logging

try:
    from transformers import AutoProcessor, SiglipVisionModel
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False
    logging.warning("Transformers/SigLIP not available. ML classification disabled.")


class MLTeamClassifier:
    """ML-based team classifier using SigLIP + UMAP + KMeans"""
    
    def __init__(self, device: str = 'cuda', n_teams: int = 2):
        """
        Initialize ML team classifier
        
        Args:
            device: Device for inference
            n_teams: Number of teams to classify
        """
        self.device = device
        self.n_teams = n_teams
        self.initialized = False
        self.confidence = 0.0
        
        if SIGLIP_AVAILABLE:
            try:
                # Load SigLIP model
                model_name = 'google/siglip-base-patch16-224'
                self.features_model = SiglipVisionModel.from_pretrained(model_name).to(device)
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                # Initialize clustering components
                self.reducer = umap.UMAP(n_components=3, random_state=42)
                self.cluster_model = KMeans(n_clusters=n_teams, random_state=42)
                self.scaler = StandardScaler()
                
                logging.info("ML team classifier initialized")
            except Exception as e:
                logging.error(f"Failed to initialize ML classifier: {e}")
                self.features_model = None
        else:
            self.features_model = None
            
    def extract_features(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        """Extract features from crops using SigLIP"""
        if not self.features_model or not crops:
            return None
            
        try:
            import supervision as sv
            
            # Convert crops to PIL
            crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
            
            # Process in batches
            batch_size = 32
            all_features = []
            
            with torch.no_grad():
                for i in range(0, len(crops_pil), batch_size):
                    batch = crops_pil[i:i + batch_size]
                    
                    # Process batch
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                    outputs = self.features_model(**inputs)
                    
                    # Extract embeddings
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                    all_features.append(embeddings)
                    
            return np.concatenate(all_features) if all_features else None
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return None
            
    def fit(self, crops: List[np.ndarray]) -> None:
        """Train the classifier on crops"""
        if not self.features_model:
            raise RuntimeError("ML classifier not available")
            
        # Extract features
        features = self.extract_features(crops)
        if features is None or features.shape[0] < self.n_teams * 5:
            raise ValueError("Not enough features extracted")
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Dimensionality reduction
        projections = self.reducer.fit_transform(features_scaled)
        
        # Clustering
        self.cluster_model.fit(projections)
        
        # Check clustering quality
        labels = self.cluster_model.labels_
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == self.n_teams:
            self.initialized = True
            self.confidence = 0.8
            logging.info(f"ML classifier trained successfully. Clusters: {np.bincount(labels)}")
        else:
            logging.warning(f"Clustering produced {len(unique_labels)} clusters instead of {self.n_teams}")
            
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team assignments"""
        if not self.initialized:
            raise RuntimeError("Classifier not initialized")
            
        # Extract features
        features = self.extract_features(crops)
        if features is None:
            return np.zeros(len(crops), dtype=int)
            
        # Transform features
        features_scaled = self.scaler.transform(features)
        projections = self.reducer.transform(features_scaled)
        
        # Predict clusters
        predictions = self.cluster_model.predict(projections)
        
        return predictions
        
    def is_initialized(self) -> bool:
        """Check if classifier is initialized"""
        return self.initialized
        
    def get_confidence(self) -> float:
        """Get classification confidence"""
        return self.confidence
