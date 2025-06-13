"""
Basketball-specific team balancing rules
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
import cv2


class BasketballTeamBalancer:
    """Enforce basketball 5v5 team balance rules"""
    
    def __init__(self, min_team_size: int = 3, max_team_size: int = 7):
        """
        Initialize basketball team balancer
        
        Args:
            min_team_size: Minimum players per team
            max_team_size: Maximum players per team
        """
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.balance_history = []
        
    def balance_teams(self, predictions: np.ndarray, crops: List[np.ndarray]) -> np.ndarray:
        """
        Balance team assignments to ensure valid basketball teams
        
        Args:
            predictions: Initial team predictions
            crops: Player crops for brightness analysis
            
        Returns:
            Balanced team assignments
        """
        if len(predictions) == 0:
            return predictions
            
        team_0_count = np.sum(predictions == 0)
        team_1_count = np.sum(predictions == 1)
        
        logging.debug(f"Team balance check: T0={team_0_count}, T1={team_1_count}")
        
        # Check if teams are already balanced
        if (self.min_team_size <= team_0_count <= self.max_team_size and
            self.min_team_size <= team_1_count <= self.max_team_size):
            return predictions
            
        # Teams need balancing
        logging.info(f"Teams imbalanced ({team_0_count}v{team_1_count}), applying correction...")
        
        # Calculate brightness for each player
        brightnesses = []
        for crop in crops:
            brightness = self._calculate_brightness(crop)
            brightnesses.append(brightness)
            
        brightnesses = np.array(brightnesses)
        
        # Try different strategies
        balanced = self._balance_by_brightness(predictions, brightnesses)
        
        # Verify balance
        new_team_0 = np.sum(balanced == 0)
        new_team_1 = np.sum(balanced == 1)
        
        if (self.min_team_size <= new_team_0 <= self.max_team_size and
            self.min_team_size <= new_team_1 <= self.max_team_size):
            logging.info(f"Balance successful: T0={new_team_0}, T1={new_team_1}")
            self._record_balance(predictions, balanced)
            return balanced
            
        # If still not balanced, use equal split
        return self._equal_split(len(predictions))
        
    def _balance_by_brightness(self, predictions: np.ndarray, brightnesses: np.ndarray) -> np.ndarray:
        """Balance teams using brightness values"""
        # Try different percentiles to find good split
        for percentile in [50, 45, 55, 40, 60, 35, 65]:
            threshold = np.percentile(brightnesses, percentile)
            
            new_predictions = np.zeros_like(predictions)
            new_predictions[brightnesses > threshold] = 0  # Brighter = Team 0
            new_predictions[brightnesses <= threshold] = 1  # Darker = Team 1
            
            team_0 = np.sum(new_predictions == 0)
            team_1 = np.sum(new_predictions == 1)
            
            if (self.min_team_size <= team_0 <= self.max_team_size and
                self.min_team_size <= team_1 <= self.max_team_size):
                return new_predictions
                
        # Fallback to original if no good split found
        return predictions
        
    def _equal_split(self, n_players: int) -> np.ndarray:
        """Create equal team split"""
        predictions = np.zeros(n_players, dtype=int)
        
        # Alternate assignment
        half = n_players // 2
        predictions[half:] = 1
        
        # Shuffle for randomness
        np.random.shuffle(predictions)
        
        return predictions
        
    def _calculate_brightness(self, crop: np.ndarray) -> float:
        """Calculate brightness of crop"""
        if crop is None or crop.size == 0:
            return 128.0
            
        # Focus on jersey area
        h, w = crop.shape[:2]
        jersey_area = crop[h//6:h//2, w//4:3*w//4]
        
        if jersey_area.size == 0:
            jersey_area = crop
            
        # Convert to grayscale
        if len(jersey_area.shape) == 3:
            gray = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2GRAY)
        else:
            gray = jersey_area
            
        return float(np.mean(gray))
        
    def _record_balance(self, original: np.ndarray, balanced: np.ndarray):
        """Record balancing action for analysis"""
        self.balance_history.append({
            'original_counts': (np.sum(original == 0), np.sum(original == 1)),
            'balanced_counts': (np.sum(balanced == 0), np.sum(balanced == 1)),
            'changes': np.sum(original != balanced)
        })
        
    def get_balance_statistics(self) -> Dict[str, any]:
        """Get balancing statistics"""
        if not self.balance_history:
            return {'total_balances': 0}
            
        total_changes = sum(h['changes'] for h in self.balance_history)
        
        return {
            'total_balances': len(self.balance_history),
            'total_changes': total_changes,
            'avg_changes_per_balance': total_changes / len(self.balance_history)
        }
