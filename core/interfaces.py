"""
Abstract interfaces for basketball analytics components
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .models import (
    Detection, Track, Team, PossessionInfo, 
    PlayEvent, PlayClassification, AnalysisResult
)


class Detector(ABC):
    """Abstract interface for object detection"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame"""
        pass
    
    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in multiple frames"""
        pass


class MaskGenerator(ABC):
    """Abstract interface for mask generation"""
    
    @abstractmethod
    def generate_masks(self, frame: np.ndarray, detections: List[Detection]) -> List[Optional[np.ndarray]]:
        """Generate masks for detections"""
        pass


class Tracker(ABC):
    """Abstract interface for object tracking"""
    
    @abstractmethod
    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """Update tracks with new detections"""
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state"""
        pass


class TeamClassifier(ABC):
    """Abstract interface for team classification"""
    
    @abstractmethod
    def classify(self, tracks: List[Track], frame: np.ndarray) -> Dict[int, int]:
        """Classify tracks into teams. Returns {track_id: team_id}"""
        pass
    
    @abstractmethod
    def fit(self, crops: List[np.ndarray]) -> None:
        """Train classifier on player crops"""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if classifier is ready"""
        pass


class PossessionAnalyzer(ABC):
    """Abstract interface for possession analysis"""
    
    @abstractmethod
    def analyze_possession(self, 
                         player_tracks: List[Track], 
                         ball_track: Optional[Track],
                         frame_idx: int) -> PossessionInfo:
        """Analyze ball possession for current frame"""
        pass


class PlayClassifier(ABC):
    """Abstract interface for play classification"""
    
    @abstractmethod
    def classify_play(self, 
                     possession_data: Dict[str, Any],
                     context: Dict[str, Any]) -> PlayClassification:
        """Classify the type of basketball play"""
        pass


class EventDetector(ABC):
    """Abstract interface for event detection"""
    
    @abstractmethod
    def detect_events(self,
                     detections: Dict[str, List[Detection]],
                     possession_info: PossessionInfo,
                     frame_idx: int) -> List[PlayEvent]:
        """Detect basketball events (shots, rebounds, etc.)"""
        pass


class Analyzer(ABC):
    """Abstract interface for complete game analysis"""
    
    @abstractmethod
    def analyze_frame(self, 
                     frame_data: Dict[str, Any],
                     frame_idx: int) -> Dict[str, Any]:
        """Analyze a single frame"""
        pass
    
    @abstractmethod
    def get_results(self) -> AnalysisResult:
        """Get complete analysis results"""
        pass


class Visualizer(ABC):
    """Abstract interface for visualization"""
    
    @abstractmethod
    def draw_annotations(self,
                        frame: np.ndarray,
                        detections: Dict[str, List[Detection]],
                        tracks: List[Track],
                        **kwargs) -> np.ndarray:
        """Draw annotations on frame"""
        pass


class FrameProcessor(ABC):
    """Abstract interface for complete frame processing"""
    
    @abstractmethod
    def process_frame(self,
                     frame: np.ndarray,
                     frame_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame and return annotated frame + metadata"""
        pass
