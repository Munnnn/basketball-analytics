"""
Color and brightness based team classification
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
import logging


class ColorBasedClassifier:
    """Team classification based on jersey colors and brightness"""
    
    def __init__(self, n_teams: int = 2):
        """
        Initialize color-based classifier
        
        Args:
            n_teams: Number of teams
        """
        self.n_teams = n_teams
        self.initialized = False
        self.team_colors = {}
        self.brightness_threshold = None
        
    def extract_jersey_color_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract color features from jersey region"""
        if crop is None or crop.size == 0:
            return np.zeros(8)  # Return zero features
            
        h, w = crop.shape[:2]
        
        # Extract jersey region (upper middle portion)
        jersey_region = crop[h//6:h//2, w//4:3*w//4]
        if jersey_region.size == 0:
            jersey_region = crop[:h//2, :]  # Fallback to upper half
            
        # Convert to LAB color space for better color clustering
        jersey_lab = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
        
        # Extract dominant colors using k-means
        pixels = jersey_lab.reshape(-1, 3)
        
        # Filter out very dark and very bright pixels
        l_channel = pixels[:, 0]
        valid_mask = (l_channel > 30) & (l_channel < 220)
        valid_pixels = pixels[valid_mask]
        
        if len(valid_pixels) < 10:
            return np.zeros(8)
            
        # Get dominant colors
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(valid_pixels)
        dominant_colors = kmeans.cluster_centers_
        
        # Calculate brightness
        brightness1 = float(dominant_colors[0, 0])  # L channel
        brightness2 = float(dominant_colors[1, 0])
        
        # Create feature vector
        features = np.concatenate([
            dominant_colors[0],  # First dominant color (LAB)
            dominant_colors[1],  # Second dominant color (LAB)
            [brightness1, brightness2]  # Brightness values
        ])
        
        return features
        
    def calculate_brightness(self, crop: np.ndarray) -> float:
        """Calculate average brightness of jersey area"""
        if crop is None or crop.size == 0:
            return 128.0
            
        h, w = crop.shape[:2]
        jersey_area = crop[h//6:h//2, w//4:3*w//4]
        
        if jersey_area.size == 0:
            jersey_area = crop
            
        # Convert to grayscale
        gray = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
        
    def fit(self, crops: List[np.ndarray]) -> None:
        """Train classifier on crops"""
        if len(crops) < self.n_teams * 3:
            raise ValueError("Not enough crops for training")
            
        # Extract features for all crops
        features = []
        brightnesses = []
        
        for crop in crops:
            feature = self.extract_jersey_color_features(crop)
            brightness = self.calculate_brightness(crop)
            
            if not np.all(feature == 0):  # Valid feature
                features.append(feature)
                brightnesses.append(brightness)
                
        if len(features) < self.n_teams * 2:
            raise ValueError("Not enough valid features extracted")
            
        # Cluster based on color features
        features_array = np.array(features)
        kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_array)
        
        # Store team characteristics
        for team_id in range(self.n_teams):
            team_mask = labels == team_id
            if np.any(team_mask):
                team_features = features_array[team_mask]
                team_brightnesses = np.array(brightnesses)[team_mask]
                
                self.team_colors[team_id] = {
                    'mean_features': np.mean(team_features, axis=0),
                    'mean_brightness': np.mean(team_brightnesses),
                    'std_brightness': np.std(team_brightnesses)
                }
                
        # Set brightness threshold
        all_brightnesses = np.array(brightnesses)
        self.brightness_threshold = np.median(all_brightnesses)
        
        self.initialized = True
        logging.info(f"Color classifier initialized. Brightness threshold: {self.brightness_threshold:.1f}")
        
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team assignments based on color/brightness"""
        if not crops:
            return np.array([])
            
        predictions = []
        
        for crop in crops:
            if self.initialized and self.team_colors:
                # Extract features
                features = self.extract_jersey_color_features(crop)
                
                # Find closest team based on features
                min_distance = float('inf')
                predicted_team = 0
                
                for team_id, team_data in self.team_colors.items():
                    distance = np.linalg.norm(features - team_data['mean_features'])
                    if distance < min_distance:
                        min_distance = distance
                        predicted_team = team_id
                        
                predictions.append(predicted_team)
            else:
                # Simple brightness-based classification
                brightness = self.calculate_brightness(crop)
                threshold = self.brightness_threshold or 128.0
                predictions.append(0 if brightness > threshold else 1)
                
        return np.array(predictions)
        
    def is_initialized(self) -> bool:
        """Check if classifier is initialized"""
        return self.initialized
