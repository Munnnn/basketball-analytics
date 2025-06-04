"""
Global settings management
"""

import os
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class Settings:
    """Global application settings"""
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    model_dir: str = "./models"
    temp_dir: str = "./temp"
    
    # Processing
    default_batch_size: int = 20
    default_device: str = "cuda"
    max_frames_per_video: int = 10000
    
    # Detection
    detection_confidence: float = 0.2
    nms_threshold: float = 0.5
    
    # Tracking
    track_activation_threshold: float = 0.4
    lost_track_buffer: int = 90
    min_matching_threshold: float = 0.2
    min_consecutive_frames: int = 3
    max_tracks: int = 15
    
    # Team classification
    min_players_for_team_init: int = 6
    team_confidence_threshold: float = 0.6
    enforce_basketball_rules: bool = True
    
    # Possession
    ball_proximity_threshold: float = 80.0
    possession_change_threshold: int = 8
    min_possession_duration: int = 5
    
    # Events
    shot_distance_threshold: float = 80.0
    rebound_proximity_threshold: float = 120.0
    event_cooldown_frames: int = 30
    
    # Visualization
    use_ellipse_annotation: bool = True
    mask_opacity: float = 0.3
    
    # Performance
    memory_optimization: bool = True
    memory_cleanup_interval: int = 50
    target_memory_usage: float = 0.8
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Settings':
        """Load settings from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment variables"""
        settings = cls()
        
        # Override from environment
        for field in settings.__dataclass_fields__:
            env_key = f"BASKETBALL_{field.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # Convert types
                field_type = settings.__dataclass_fields__[field].type
                if field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                elif field_type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                setattr(settings, field, value)
                
        return settings
    
    def save(self, filepath: str):
        """Save settings to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    
    if _settings is None:
        # Try loading from file first
        config_file = os.environ.get("BASKETBALL_CONFIG", "config/settings.json")
        if os.path.exists(config_file):
            _settings = Settings.from_file(config_file)
        else:
            # Load from environment or use defaults
            _settings = Settings.from_env()
            
    return _settings


def reset_settings():
    """Reset settings (mainly for testing)"""
    global _settings
    _settings = None
