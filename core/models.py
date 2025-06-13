"""
Core data models for basketball analytics - Enhanced with basketball-specific fields
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


class PlayType(Enum):
    """Basketball play types from original enhanced play analysis"""
    ISOLATION = 0
    PICK_AND_ROLL = 1
    POST_UP = 2
    FAST_BREAK = 3
    SPOT_UP = 4
    OFF_SCREEN = 5
    HANDOFF = 6
    CUT = 7
    OFF_REBOUND = 8
    TRANSITION = 9
    HALF_COURT_SET = 10


@dataclass
class Detection:
    """Single object detection with basketball enhancements"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    mask: Optional[np.ndarray] = None
    frame_idx: Optional[int] = None

    # Basketball-specific fields
    jersey_crop: Optional[np.ndarray] = None  # For team classification
    pose_keypoints: Optional[Dict] = None     # For pose estimation

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

    @property
    def class_name(self) -> str:
        """Get class name for basketball objects"""
        class_names = {
            0: "backboard",
            1: "ball",
            2: "gameclock",
            3: "hoop",
            4: "period",
            5: "player",
            6: "referee",
            7: "scoreboard",
            8: "team_points"
        }
        return class_names.get(self.class_id, "unknown")


@dataclass
class Track:
    """Tracked object across frames with basketball enhancements"""
    id: int
    detections: List[Detection] = field(default_factory=list)
    team_id: Optional[int] = None
    state: TrackState = TrackState.TENTATIVE
    confidence: float = 0.0
    start_frame: int = 0
    last_seen_frame: int = 0

    # Basketball-specific tracking fields
    team_confidence: float = 0.0              # Confidence in team assignment
    velocity: Optional[np.ndarray] = None     # Movement velocity
    appearance_features: Optional[np.ndarray] = None  # For re-identification
    jersey_color: Optional[Tuple[int, int, int]] = None  # Dominant jersey color
    basketball_actions: List[str] = field(default_factory=list)  # Detected actions

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

    @property
    def is_active(self) -> bool:
        """Check if track is currently active"""
        return self.state in [TrackState.TRACKED, TrackState.TENTATIVE]


@dataclass
class Team:
    """Basketball team information"""
    id: int
    name: str = ""
    color_primary: Tuple[int, int, int] = (255, 255, 255)
    color_secondary: Optional[Tuple[int, int, int]] = None
    players: List[int] = field(default_factory=list)  # Track IDs

    # Basketball-specific team data
    possessions: int = 0
    total_possession_time: float = 0.0
    shots_attempted: int = 0
    shots_made: int = 0
    turnovers: int = 0
    plays_run: Dict[PlayType, int] = field(default_factory=dict)

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def avg_possession_time(self) -> float:
        """Average possession duration"""
        return self.total_possession_time / max(self.possessions, 1)

    @property
    def shooting_percentage(self) -> float:
        """Field goal percentage"""
        return self.shots_made / max(self.shots_attempted, 1) * 100


@dataclass
class Player:
    """Basketball player information"""
    track_id: int
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None
    position: Optional[str] = None

    # Basketball statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    possessions: int = 0
    time_with_ball: float = 0.0
    actions_detected: List[str] = field(default_factory=list)

    def add_action(self, action: str, frame_idx: int):
        """Add detected basketball action"""
        self.actions_detected.append(f"{action}@{frame_idx}")


@dataclass
class PossessionInfo:
    """Basketball possession information with context"""
    frame_idx: int
    player_id: Optional[int] = None
    team_id: Optional[int] = None
    ball_position: Optional[np.ndarray] = None
    confidence: float = 0.0
    duration: int = 0
    possession_change: bool = False

    # Basketball context from enhanced play analysis
    play_type: Optional[PlayType] = None
    momentum: float = 0.0  # Team momentum (-1 to 1)
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class PlayEvent:
    """Basketball play event with enhanced classification"""
    type: str  # shot_attempt, rebound, turnover, etc.
    frame_idx: int
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    position: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Basketball-specific event data
    play_context: Optional[Dict] = None  # Context when event occurred
    outcome: Optional[str] = None        # Event outcome (made/missed for shots)
    distance: Optional[float] = None     # Distance to basket for shots


@dataclass
class PlayClassification:
    """Basketball play classification with context"""
    play_type: PlayType
    play_name: str
    confidence: float
    start_frame: int
    end_frame: int
    team_id: Optional[int] = None
    key_players: List[int] = field(default_factory=list)
    events: List[PlayEvent] = field(default_factory=list)

    # Enhanced basketball context
    possession_context: Optional[Dict] = None  # Context at start of play
    pose_actions: Dict[str, List] = field(default_factory=dict)  # Detected poses/actions
    momentum_before: float = 0.0               # Team momentum before play
    momentum_after: float = 0.0                # Team momentum after play

    @property
    def duration(self) -> int:
        """Play duration in frames"""
        return self.end_frame - self.start_frame


@dataclass
class BasketballAction:
    """Detected basketball action from pose estimation"""
    action_type: str  # screen, cut, shot_form, etc.
    player_id: int
    frame_idx: int
    confidence: float
    position: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete basketball analysis result"""
    video_path: str
    fps: float
    total_frames: int
    processed_frames: int

    # Tracking results
    tracks: List[Track] = field(default_factory=list)
    teams: List[Team] = field(default_factory=list)

    # Basketball analytics results
    possessions: List[PossessionInfo] = field(default_factory=list)
    plays: List[PlayClassification] = field(default_factory=list)
    events: List[PlayEvent] = field(default_factory=list)
    basketball_actions: List[BasketballAction] = field(default_factory=list)

    # Statistics
    team_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    player_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Basketball-specific metadata
    processing_time: float = 0.0
    pose_estimation_enabled: bool = False
    context_tracking_enabled: bool = False
    basketball_enhanced: bool = True  # NEW: Indicates basketball-specific processing
    synchronized_processing: bool = False  # FIXED: Added missing field for synchronized processing

    # Processing statistics - comprehensive stats fields
    frame_processor_stats: Dict[str, Any] = field(default_factory=dict)  # Frame processor statistics
    batch_optimizer_stats: Dict[str, Any] = field(default_factory=dict)  # Batch optimizer statistics
    tracking_stats: Dict[str, Any] = field(default_factory=dict)  # Tracking performance stats
    detection_stats: Dict[str, Any] = field(default_factory=dict)  # Object detection stats
    team_assignment_stats: Dict[str, Any] = field(default_factory=dict)  # Team assignment stats
    possession_tracking_stats: Dict[str, Any] = field(default_factory=dict)  # Possession tracking stats
    event_detection_stats: Dict[str, Any] = field(default_factory=dict)  # Event detection stats
    performance_metrics: Dict[str, Any] = field(default_factory=dict)  # Overall performance metrics

    # Enhanced basketball analytics
    team_classification_stats: Dict[str, int] = field(default_factory=dict)
    basketball_analysis_stats: Dict[str, Any] = field(default_factory=dict)
    possession_timeline: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization with basketball data"""
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
                'unique_players': len([t for t in self.tracks if t.team_id is not None]),
                'team_0_players': len([t for t in self.tracks if t.team_id == 0]),
                'team_1_players': len([t for t in self.tracks if t.team_id == 1])
            },
            'basketball_analytics': {
                'total_possessions': len(self.possessions),
                'total_plays': len(self.plays),
                'total_events': len(self.events),
                'total_actions': len(self.basketball_actions),
                'play_types_detected': self._count_play_types(),
                'possession_statistics': self._get_possession_stats(),
                'team_balance_analysis': self._analyze_team_balance()
            },
            'team_stats': self.team_stats,
            'player_stats': self.player_stats,
            'enhanced_metadata': {
                'processing_time': self.processing_time,
                'pose_estimation': self.pose_estimation_enabled,
                'context_tracking': self.context_tracking_enabled,
                'basketball_enhanced': self.basketball_enhanced,
                'synchronized_processing': self.synchronized_processing,
                'team_classification_stats': self.team_classification_stats,
                'basketball_analysis_stats': self.basketball_analysis_stats
            },
            'processing_statistics': {
                'frame_processor_stats': self.frame_processor_stats,
                'batch_optimizer_stats': self.batch_optimizer_stats,
                'tracking_stats': self.tracking_stats,
                'detection_stats': self.detection_stats,
                'team_assignment_stats': self.team_assignment_stats,
                'possession_tracking_stats': self.possession_tracking_stats,
                'event_detection_stats': self.event_detection_stats,
                'performance_metrics': self.performance_metrics
            },
            'timeline': {
                'possessions': [p.__dict__ for p in self.possessions],
                'plays': [p.__dict__ for p in self.plays],
                'events': [e.__dict__ for e in self.events]
            }
        }

    def _count_play_types(self) -> Dict[str, int]:
        """Count occurrences of each play type"""
        play_counts = {}
        for play in self.plays:
            play_name = play.play_name
            play_counts[play_name] = play_counts.get(play_name, 0) + 1
        return play_counts

    def _get_possession_stats(self) -> Dict[str, Any]:
        """Get possession statistics"""
        if not self.possessions:
            return {}

        team_possessions = {0: 0, 1: 0}
        total_duration = {0: 0, 1: 0}

        for possession in self.possessions:
            if possession.team_id in [0, 1]:
                team_possessions[possession.team_id] += 1
                total_duration[possession.team_id] += possession.duration

        return {
            'team_possessions': team_possessions,
            'avg_possession_duration': {
                0: total_duration[0] / max(team_possessions[0], 1),
                1: total_duration[1] / max(team_possessions[1], 1)
            },
            'total_possessions': len(self.possessions)
        }

    def _analyze_team_balance(self) -> Dict[str, Any]:
        """Analyze team balance (5v5 enforcement)"""
        team_player_counts = {0: 0, 1: 0}

        for track in self.tracks:
            if track.team_id in [0, 1]:
                team_player_counts[track.team_id] += 1

        total_players = sum(team_player_counts.values())
        balance_score = 1.0 - abs(team_player_counts[0] - team_player_counts[1]) / max(total_players, 1)

        return {
            'team_0_players': team_player_counts[0],
            'team_1_players': team_player_counts[1],
            'balance_score': balance_score,  # 1.0 = perfectly balanced
            'is_basketball_balanced': 4 <= team_player_counts[0] <= 6 and 4 <= team_player_counts[1] <= 6
        }

    def get_basketball_summary(self) -> str:
        """Get human-readable basketball analysis summary"""
        summary_lines = [
            f"🏀 Basketball Analysis Summary",
            f"📹 Video: {self.video_path}",
            f"⏱️  Processed: {self.processed_frames}/{self.total_frames} frames",
            f"🏃 Players Tracked: {len([t for t in self.tracks if t.team_id is not None])}",
            ""
        ]

        # Team balance
        balance_analysis = self._analyze_team_balance()
        summary_lines.extend([
            f"👥 Team Balance:",
            f"   Team 0: {balance_analysis['team_0_players']} players",
            f"   Team 1: {balance_analysis['team_1_players']} players",
            f"   Balance: {'✅ Good' if balance_analysis['is_basketball_balanced'] else '⚠️ Imbalanced'}",
            ""
        ])

        # Analytics
        summary_lines.extend([
            f"📊 Basketball Analytics:",
            f"   Possessions: {len(self.possessions)}",
            f"   Plays Classified: {len(self.plays)}",
            f"   Events Detected: {len(self.events)}",
            f"   Actions Detected: {len(self.basketball_actions)}",
            ""
        ])

        # Performance
        if self.processing_time > 0:
            fps_processed = self.processed_frames / self.processing_time
            summary_lines.extend([
                f"⚡ Performance:",
                f"   Processing Time: {self.processing_time:.1f}s",
                f"   Speed: {fps_processed:.1f} fps",
                ""
            ])

        # Features enabled
        features = []
        if self.pose_estimation_enabled:
            features.append("Pose Estimation")
        if self.context_tracking_enabled:
            features.append("Context Tracking")
        if self.basketball_enhanced:
            features.append("Basketball Intelligence")
        if self.synchronized_processing:
            features.append("Synchronized Processing")

        if features:
            summary_lines.extend([
                f"🔧 Features: {', '.join(features)}",
                ""
            ])

        return "\n".join(summary_lines)


# Basketball-specific constants from original code
PLAYER_ID = 5
REF_ID = 6
BALL_ID = 1
BACKBOARD_ID = 0
HOOP_ID = 3
GAMECLOCK_ID = 2
PERIOD_ID = 4
SCOREBOARD_ID = 7
TEAM_POINTS_ID = 8

# Basketball court constants
BALL_PROXIMITY_THRESHOLD = 80
POSSESSION_CHANGE_THRESHOLD = 8
MIN_POSSESSION_DURATION = 5

# Team classification constants
MIN_PLAYERS_FOR_TEAM_INIT = 6
BASKETBALL_5V5_MIN_PLAYERS = 4
BASKETBALL_5V5_MAX_PLAYERS = 6
