"""
Basketball event detection
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from core import Detection, PlayEvent, PossessionInfo
from core.constants import (
    BALL_ID, HOOP_ID, BACKBOARD_ID,
    SHOT_DISTANCE_THRESHOLD, REBOUND_PROXIMITY_THRESHOLD,
    EVENT_COOLDOWN_FRAMES
)


class EventDetector:
    """Base class for basketball event detection - NOW WITH DEFAULT IMPLEMENTATION"""

    def __init__(self):
        """Initialize event detector"""
        self.last_event_frame = -EVENT_COOLDOWN_FRAMES
        self.event_cooldown = EVENT_COOLDOWN_FRAMES

    def detect_events(self,
                     detections: Dict[str, List[Detection]],
                     possession_info: PossessionInfo,
                     frame_idx: int) -> List[PlayEvent]:
        """
        Detect basketball events in current frame - DEFAULT IMPLEMENTATION

        Args:
            detections: Dictionary of detections by type
            possession_info: Current possession information
            frame_idx: Current frame index

        Returns:
            List of detected events
        """
        # Default implementation - return empty list instead of raising error
        return []


class ShotDetector(EventDetector):
    """Detect shot attempts"""

    def __init__(self, distance_threshold: float = SHOT_DISTANCE_THRESHOLD):
        """
        Initialize shot detector

        Args:
            distance_threshold: Maximum distance for shot detection
        """
        super().__init__()
        self.distance_threshold = distance_threshold

    def detect_events(self,
                     detections: Dict[str, List[Detection]],
                     possession_info: PossessionInfo,
                     frame_idx: int) -> List[PlayEvent]:
        """Detect shot attempts"""
        events = []

        # Check cooldown
        if frame_idx - self.last_event_frame < self.event_cooldown:
            return events

        # Get ball and hoop detections
        ball_detections = detections.get('ball', [])
        hoop_detections = detections.get('hoop', [])

        if not ball_detections or not hoop_detections:
            return events

        # Use first ball and hoop
        ball = ball_detections[0]
        hoop = hoop_detections[0]

        # Calculate distance
        ball_center = ball.center
        hoop_center = hoop.center
        distance = np.linalg.norm(ball_center - hoop_center)

        # Check shot criteria
        is_close = distance < self.distance_threshold
        has_possession = possession_info.team_id is not None
        ball_above_hoop = ball_center[1] < hoop_center[1]

        if is_close and has_possession and ball_above_hoop:
            event = PlayEvent(
                type='shot_attempt',
                frame_idx=frame_idx,
                team_id=possession_info.team_id,
                player_id=possession_info.player_id,
                position=ball_center,
                confidence=min(1.0, (self.distance_threshold - distance) / self.distance_threshold),
                metadata={
                    'distance': float(distance),
                    'shot_type': self._classify_shot_type(distance)
                }
            )
            events.append(event)
            self.last_event_frame = frame_idx

        return events

    def _classify_shot_type(self, distance: float) -> str:
        """Classify shot type based on distance"""
        if distance < 50:
            return 'layup'
        elif distance < 150:
            return 'mid_range'
        else:
            return 'three_pointer'


class ReboundDetector(EventDetector):
    """Detect rebounds"""

    def __init__(self, proximity_threshold: float = REBOUND_PROXIMITY_THRESHOLD):
        """
        Initialize rebound detector

        Args:
            proximity_threshold: Maximum distance for rebound detection
        """
        super().__init__()
        self.proximity_threshold = proximity_threshold
        self.shot_detected_frame = -1

    def detect_events(self,
                     detections: Dict[str, List[Detection]],
                     possession_info: PossessionInfo,
                     frame_idx: int) -> List[PlayEvent]:
        """Detect rebounds"""
        events = []

        # Check cooldown
        if frame_idx - self.last_event_frame < self.event_cooldown:
            return events

        # Get ball and backboard detections
        ball_detections = detections.get('ball', [])
        backboard_detections = detections.get('backboard', [])

        if not ball_detections or not backboard_detections:
            return events

        # Use first ball and backboard
        ball = ball_detections[0]
        backboard = backboard_detections[0]

        # Calculate distance
        ball_center = ball.center
        backboard_center = backboard.center
        distance = np.linalg.norm(ball_center - backboard_center)

        # Check rebound criteria
        if distance < self.proximity_threshold:
            # Check if this follows a shot attempt
            is_potential_rebound = (
                self.shot_detected_frame > 0 and
                frame_idx - self.shot_detected_frame < 60  # Within 2 seconds
            )

            event = PlayEvent(
                type='potential_rebound' if is_potential_rebound else 'ball_near_backboard',
                frame_idx=frame_idx,
                team_id=possession_info.team_id,
                player_id=possession_info.player_id,
                position=ball_center,
                confidence=min(1.0, (self.proximity_threshold - distance) / self.proximity_threshold),
                metadata={
                    'distance': float(distance),
                    'frames_since_shot': frame_idx - self.shot_detected_frame if is_potential_rebound else None
                }
            )
            events.append(event)
            self.last_event_frame = frame_idx

        return events

    def register_shot(self, frame_idx: int):
        """Register that a shot was detected"""
        self.shot_detected_frame = frame_idx


class UnifiedEventDetector(EventDetector):
    """Unified event detector that combines multiple detectors"""

    def __init__(self):
        """Initialize unified event detector"""
        super().__init__()
        self.shot_detector = ShotDetector()
        self.rebound_detector = ReboundDetector()

    def detect_events(self,
                     detections: Dict[str, List[Detection]],
                     possession_info: PossessionInfo,
                     frame_idx: int) -> List[PlayEvent]:
        """Detect all types of events"""
        events = []

        # Detect shots
        shot_events = self.shot_detector.detect_events(
            detections, possession_info, frame_idx
        )
        events.extend(shot_events)

        # Register shots for rebound detection
        for event in shot_events:
            if event.type == 'shot_attempt':
                self.rebound_detector.register_shot(frame_idx)

        # Detect rebounds
        rebound_events = self.rebound_detector.detect_events(
            detections, possession_info, frame_idx
        )
        events.extend(rebound_events)

        return events


class BasketballEventDetector(UnifiedEventDetector):
    """
    BASKETBALL ENHANCED: Main event detector with basketball-specific enhancements
    This is the class your FrameProcessor should use
    """

    def __init__(self):
        """Initialize basketball-enhanced event detector"""
        super().__init__()
        self.basketball_enhanced = True
        print("🏀 Basketball-enhanced event detector initialized")

    def detect_events(self,
                     detections: Dict[str, List[Detection]],
                     possession_info: PossessionInfo,
                     frame_idx: int) -> List[PlayEvent]:
        """
        Basketball-enhanced event detection
        """
        # Use the unified detector logic
        events = super().detect_events(detections, possession_info, frame_idx)

        # Add basketball-specific enhancements
        enhanced_events = []
        for event in events:
            # Enhance event with basketball context
            enhanced_event = self._enhance_basketball_event(event, possession_info)
            enhanced_events.append(enhanced_event)

        return enhanced_events

    def _enhance_basketball_event(self, event: PlayEvent, possession_info: PossessionInfo) -> PlayEvent:
        """Add basketball-specific enhancements to events"""
        # Add basketball context to metadata
        if event.metadata is None:
            event.metadata = {}

        event.metadata.update({
            'basketball_enhanced': True,
            'possession_duration': possession_info.duration if possession_info else 0,
            'possession_confidence': possession_info.confidence if possession_info else 0.0
        })

        return event


# Default detector for backwards compatibility
def create_event_detector() -> BasketballEventDetector:
    """Create default basketball event detector"""
    return BasketballEventDetector()
