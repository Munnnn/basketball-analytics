#!/usr/bin/env python3
"""
Basketball Analytics Test Suite
Run: .venv/bin/python tests/run_tests.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

passed = 0
total = 0
failures = []


def check(name, condition):
    global passed, total
    total += 1
    if condition:
        passed += 1
    else:
        failures.append(name)
        print(f"  FAIL: {name}")


def section(name):
    print(f"\n--- {name} ---")


# ============================================================================
# 1. Constants
# ============================================================================
section("Constants")

from core.constants import (
    BACKBOARD_ID, BALL_ID, GAMECLOCK_ID, HOOP_ID, PERIOD_ID,
    PLAYER_ID, REF_ID, SCOREBOARD_ID, TEAM_POINTS_ID,
    JERSEY_REGION_RATIO, scale_threshold, PLAY_TYPES,
)

ids = [BACKBOARD_ID, BALL_ID, GAMECLOCK_ID, HOOP_ID, PERIOD_ID,
       PLAYER_ID, REF_ID, SCOREBOARD_ID, TEAM_POINTS_ID]
check("unique class IDs", len(ids) == len(set(ids)))
check("jersey ratio in (0,1)", 0 < JERSEY_REGION_RATIO < 1)
check("scale_threshold at reference", scale_threshold(80, 1280) == 80.0)
check("scale_threshold 2x", abs(scale_threshold(80, 2560) - 160.0) < 0.01)
check("scale_threshold 0.5x", abs(scale_threshold(80, 640) - 40.0) < 0.01)
check("play types 0-10", all(i in PLAY_TYPES for i in range(11)))

# ============================================================================
# 2. Core Models
# ============================================================================
section("Core Models")

from core.models import Detection, Track, TrackState, PossessionInfo, AnalysisResult

d = Detection(bbox=np.array([10, 20, 30, 40]), confidence=0.9, class_id=PLAYER_ID)
check("detection center", list(d.center) == [20.0, 30.0])
check("detection area", d.area == 400.0)
check("class_name player", d.class_name == "player")
check("class_name ball", Detection(bbox=np.array([0, 0, 1, 1]), confidence=0.5, class_id=BALL_ID).class_name == "ball")
check("class_name unknown", Detection(bbox=np.array([0, 0, 1, 1]), confidence=0.5, class_id=99).class_name == "unknown")

check("TrackState.TRACKED value", TrackState.TRACKED.value == "tracked")
check("TrackState.LOST value", TrackState.LOST.value == "lost")
check("TrackState.TENTATIVE value", TrackState.TENTATIVE.value == "tentative")
check("TrackState.REMOVED value", TrackState.REMOVED.value == "removed")

t = Track(id=1, state=TrackState.TRACKED, confidence=0.8, start_frame=5, last_seen_frame=15)
check("track age", t.age == 10)
check("track active (tracked)", t.is_active)
check("track active (tentative)", Track(id=2, state=TrackState.TENTATIVE).is_active)
check("track inactive (lost)", not Track(id=3, state=TrackState.LOST).is_active)
check("track inactive (removed)", not Track(id=4, state=TrackState.REMOVED).is_active)
check("empty track bbox", Track(id=5).current_bbox is None)

det = Detection(bbox=np.array([1, 2, 3, 4]), confidence=0.9, class_id=PLAYER_ID)
check("current_bbox from detection", np.array_equal(Track(id=6, detections=[det]).current_bbox, [1, 2, 3, 4]))

pi = PossessionInfo(frame_idx=10, player_id=3, team_id=0,
                    ball_position=np.array([100, 200]), confidence=0.8)
check("possession to_dict ball", pi.to_dict()["ball_position"] == [100, 200])
check("possession to_dict none ball", PossessionInfo(frame_idx=0).to_dict()["ball_position"] is None)

ar = AnalysisResult(video_path="test.mp4", fps=30, total_frames=100, processed_frames=50)
check("result to_dict", ar.to_dict()["video_info"]["path"] == "test.mp4")
check("empty segments", ar.get_possession_segments() == [])
check("balance empty", ar._analyze_team_balance()["balance_score"] == 1.0)

# ============================================================================
# 3. Image Utils (shared brightness)
# ============================================================================
section("Image Utils")

from utils.image_utils import calculate_jersey_brightness

check("brightness None", calculate_jersey_brightness(None) == 128.0)
check("brightness empty", calculate_jersey_brightness(np.array([])) == 128.0)
check("brightness white", calculate_jersey_brightness(np.ones((100, 100, 3), dtype=np.uint8) * 255) > 200)
check("brightness black", calculate_jersey_brightness(np.zeros((100, 100, 3), dtype=np.uint8)) < 30)

# ============================================================================
# 4. Memory Utils Consolidation
# ============================================================================
section("Memory Utils Consolidation")

from utils.gpu_utils import get_memory_info, cleanup_memory, check_memory_threshold, MemoryMonitor, MemoryManager
from utils.memory_utils import get_memory_info as gmi2

check("same function both paths", get_memory_info is gmi2)
info = get_memory_info()
check("rss_mb positive", info["rss_mb"] > 0)
check("threshold high ok", check_memory_threshold(99.9) is False)

mon = MemoryMonitor()
mon.start_monitoring()
check("monitor summary", mon.get_summary()["cpu_start_mb"] > 0)

with MemoryManager("test") as mm:
    check("memory manager name", mm.operation_name == "test")

# ============================================================================
# 5. Team Balancing
# ============================================================================
section("Team Balancing")

from team_identification.basketball_rules import BasketballTeamBalancer

bal = BasketballTeamBalancer(min_team_size=3, max_team_size=7)

balanced = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
crops = [np.ones((64, 64, 3), dtype=np.uint8) * i * 25 for i in range(10)]
check("already balanced unchanged", np.array_equal(bal.balance_teams(balanced, crops), balanced))

imbalanced = np.array([0] * 10)
bright_crops = [np.ones((64, 64, 3), dtype=np.uint8) * (50 + i * 20) for i in range(10)]
result = bal.balance_teams(imbalanced, bright_crops)
check("rebalanced t0 in range", 3 <= np.sum(result == 0) <= 7)
check("rebalanced t1 in range", 3 <= np.sum(result == 1) <= 7)
check("brightness None fallback", bal._calculate_brightness(None) == 128.0)
check("stats dict has key", "total_balances" in bal.get_balance_statistics())

# ============================================================================
# 6. Event Detection Cooldowns
# ============================================================================
section("Event Detection")

from analytics.events.detector import ShotDetector, ReboundDetector, EventDetector
from core.constants import EVENT_COOLDOWN_FRAMES

check("base detector empty", EventDetector().detect_events({}, PossessionInfo(frame_idx=0), 100) == [])

sd = ShotDetector(distance_threshold=100)
dets = {
    "ball": [Detection(bbox=np.array([95., 75., 105., 85.]), confidence=0.9, class_id=BALL_ID)],
    "hoop": [Detection(bbox=np.array([90., 90., 110., 110.]), confidence=0.9, class_id=HOOP_ID)],
}
poss = PossessionInfo(frame_idx=0, team_id=0, player_id=1, confidence=0.8)

ev = sd.detect_events(dets, poss, 100)
check("shot detected", len(ev) == 1 and ev[0].type == "shot_attempt")
check("cooldown blocks", sd.detect_events(dets, poss, 101) == [])
check("resumes after cooldown", len(sd.detect_events(dets, poss, 100 + EVENT_COOLDOWN_FRAMES + 1)) == 1)
check("no ball no events", sd.detect_events({"hoop": dets["hoop"]}, poss, 200) == [])
check("no hoop no events", sd.detect_events({"ball": dets["ball"]}, poss, 300) == [])

rd = ReboundDetector(proximity_threshold=100)
rd.register_shot(40)
rd_dets = {
    "ball": [Detection(bbox=np.array([115., 95., 125., 105.]), confidence=0.9, class_id=BALL_ID)],
    "backboard": [Detection(bbox=np.array([105., 85., 135., 115.]), confidence=0.9, class_id=BACKBOARD_ID)],
}
rb = rd.detect_events(rd_dets, poss, 50)
check("rebound after shot", len(rb) == 1 and rb[0].type == "potential_rebound")

# ============================================================================
# 7. Visualization (constants, colors, optimized annotator)
# ============================================================================
section("Visualization")

from visualization.colors import TeamColorManager, ColorPalette

cm = TeamColorManager()
check("ball color yellow", cm.get_detection_color(Detection(bbox=np.array([0, 0, 1, 1]), confidence=0.9, class_id=BALL_ID)) == (0, 255, 255))
check("hoop color orange", cm.get_detection_color(Detection(bbox=np.array([0, 0, 1, 1]), confidence=0.9, class_id=HOOP_ID)) == (0, 165, 255))
check("backboard gray", cm.get_detection_color(Detection(bbox=np.array([0, 0, 1, 1]), confidence=0.9, class_id=BACKBOARD_ID)) == (128, 128, 128))

p = ColorPalette([(1, 2, 3), (4, 5, 6)])
check("palette wraparound", p.get_color(2) == (1, 2, 3))
check("hex parse", ColorPalette.from_hex(["#FF0000"]).get_color(0) == (0, 0, 255))
check("distinct colors", len(ColorPalette.generate_distinct_colors(5).colors) == 5)

from visualization.annotators.shapes import ShapeAnnotator
sa = ShapeAnnotator()
frame = np.zeros((100, 100, 3), dtype=np.uint8)
ball_det = Detection(bbox=np.array([10., 10., 50., 50.]), confidence=0.9, class_id=BALL_ID)
check("annotate runs", sa.annotate(frame, [ball_det], []).shape == (100, 100, 3))

# ============================================================================
# 8. StreamingWriter Consolidation
# ============================================================================
section("StreamingWriter")

from video_io import StreamingWriter
import inspect
check("format param exists", "format" in inspect.signature(StreamingWriter.__init__).parameters)

# ============================================================================
# 9. Config
# ============================================================================
section("Config")

from config.settings import Settings, reset_settings
reset_settings()
check("default batch size", Settings().default_batch_size == 20)

# ============================================================================
# 10. ProcessorConfig
# ============================================================================
section("ProcessorConfig")

from pipeline.video_processor import ProcessorConfig
import dataclasses
fields = [f.name for f in dataclasses.fields(ProcessorConfig)]
check("no duplicate fields", len(fields) == len(set(fields)))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 55)
print(f"  RESULTS: {passed}/{total} checks passed")
if passed == total:
    print("  ALL TESTS PASSED")
else:
    print(f"  {total - passed} FAILURES:")
    for f in failures:
        print(f"    - {f}")
print("=" * 55)

sys.exit(0 if passed == total else 1)
