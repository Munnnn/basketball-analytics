"""
Core data models for basketball analytics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np


class TrackState(Enum):
    """State of a track"""
    TENTATIVE = "tentative"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Detection:
    """Single object detection"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    mask: Optional[np.ndarray] = None
    frame_idx: Optional[int] = None
    
    @property
    def center(self) -> np.ndarray:
        """Get center point of bbox"""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])
    
    @property
    def area(self) -> float:
        """Get area of bbox"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class Track:
    """Tracked object across frames"""
    id: int
    detections: List[Detection] = field(default_factory=list)
    team_id: Optional[int] = None
    state: TrackState = TrackState.TENTATIVE
    confidence: float = 0.0
    start_frame: int = 0
    last_seen_frame: int = 0
    
    @property
    def age(self) -> int:
        """Track age in frames"""
        return self.last_seen_frame - self.start_frame
    
    @property
    def current_bbox(self) -> Optional[np.ndarray]:
        """Get most recent bbox"""
        return self.detections[-1].bbox if self.detections else None
    
    @property
    def current_position(self) -> Optional[np.ndarray]:
        """Get most recent center position"""
        return self.detections[-1].center if self.detections else None


@dataclass
class Team:
    """Team information"""
    id: int
    name: str = ""
    color_primary: Tuple[int, int, int] = (255, 255, 255)
    color_secondary: Optional[Tuple[int, int, int]] = None
    players: List[int] = field(default_factory=list)  # Track IDs
    
    @property
    def player_count(self) -> int:
        return len(self.players)


@dataclass
class Player:
    """Player information"""
    track_id: int
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None
    position: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PossessionInfo:
    """Ball possession information"""
    frame_idx: int
    player_id: Optional[int] = None
    team_id: Optional[int] = None
    ball_position: Optional[np.ndarray] = None
    confidence: float = 0.0
    duration: int = 0
    possession_change: bool = False
    

@dataclass
class PlayEvent:
    """Basketball play event"""
    type: str  # shot_attempt, rebound, turnover, etc.
    frame_idx: int
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    position: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayClassification:
    """Classified basketball play"""
    play_type: int
    play_name: str
    confidence: float
    start_frame: int
    end_frame: int
    team_id: Optional[int] = None
    key_players: List[int] = field(default_factory=list)
    events: List[PlayEvent] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result for a video"""
    video_path: str
    fps: float
    total_frames: int
    processed_frames: int
    
    # Tracking results
    tracks: List[Track] = field(default_factory=list)
    teams: List[Team] = field(default_factory=list)
    
    # Analytics results
    possessions: List[PossessionInfo] = field(default_factory=list)
    plays: List[PlayClassification] = field(default_factory=list)
    events: List[PlayEvent] = field(default_factory=list)
    
    # Statistics
    team_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    player_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    processing_time: float = 0.0
    pose_estimation_enabled: bool = False
    context_tracking_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'video_info': {
                'path': self.video_path,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'processed_frames': self.processed_frames
            },
            'tracking': {
                'total_tracks': len(self.tracks),
                'total_teams': len(self.teams),
                'unique_players': len([t for t in self.tracks if t.team_id is not None])
            },
            'analytics': {
                'total_possessions': len(self.possessions),
                'total_plays': len(self.plays),
                'total_events': len(self.events)
            },
            'team_stats': self.team_stats,
            'player_stats': self.player_stats,
            'metadata': {
                'processing_time': self.processing_time,
                'pose_estimation': self.pose_estimation_enabled,
                'context_tracking': self.context_tracking_enabled
            }
        }
