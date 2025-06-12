
"""
Core domain models and interfaces for basketball analytics system
"""

from .models import (
    Detection, Track, TrackState, Team, Player,
    PossessionInfo, PlayEvent, AnalysisResult, PlayClassification, BasketballAction
)
from .interfaces import (
    Detector, Tracker, TeamClassifier, Analyzer,
    Visualizer, FrameProcessor, MaskGenerator
)
from .constants import (
    PLAYER_ID, REF_ID, BALL_ID, BACKBOARD_ID, HOOP_ID,
    COURT_WIDTH, COURT_HEIGHT, DEFAULT_FPS
)

__all__ = [
    # Models
    'Detection', 'Track', 'TrackState', 'Team', 'Player',
    'PossessionInfo', 'PlayEvent', 'AnalysisResult', 'BasketballAction'
    # Interfaces
    'Detector', 'Tracker', 'TeamClassifier', 'Analyzer',
    'Visualizer', 'FrameProcessor', 'MaskGenerator',
    # Constants
    'PLAYER_ID', 'REF_ID', 'BALL_ID', 'BACKBOARD_ID', 'HOOP_ID',
    'COURT_WIDTH', 'COURT_HEIGHT', 'DEFAULT_FPS'
]
