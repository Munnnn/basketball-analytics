"""
Basketball Analytics System

A comprehensive AI-powered system for analyzing basketball games with:
- Player and ball tracking
- Team identification
- Play classification
- Possession analysis
- Event detection
- Pose estimation
"""

__version__ = "2.0.0"

# Import main components for easy access
from .core import (
    Detection, Track, Team, Player,
    PossessionInfo, PlayEvent, AnalysisResult
)

from .pipeline import VideoProcessor, ProcessorConfig

from .apps.gradio import create_app, launch_app

__all__ = [
    # Core models
    'Detection', 'Track', 'Team', 'Player',
    'PossessionInfo', 'PlayEvent', 'AnalysisResult',
    
    # Pipeline
    'VideoProcessor', 'ProcessorConfig',
    
    # Apps
    'create_app', 'launch_app',
    
    # Version
    '__version__'
]
