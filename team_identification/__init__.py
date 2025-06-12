"""
Team identification and classification
"""

from .classifier import TeamClassifierInterface, UnifiedTeamClassifier
from .ml_classifier import MLTeamClassifier
from .color_classifier import ColorBasedClassifier
from .basketball_rules import BasketballTeamBalancer
from .crop_manager import CropManager
from .advanced_manager import AdvancedTeamClassificationManager

__all__ = [
    'TeamClassifierInterface', 'UnifiedTeamClassifier',
    'MLTeamClassifier', 'ColorBasedClassifier',
    'BasketballTeamBalancer', 'CropManager',
    'AdvancedTeamClassificationManager'
]
