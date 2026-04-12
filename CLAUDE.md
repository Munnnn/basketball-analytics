# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based AI system for analyzing basketball game video. Processes video frames through a pipeline of YOLO detection, multi-object tracking, team classification, possession analysis, play classification, event detection, and pose estimation. Version 2.0.0.

## Running the Application

```bash
# CLI - analyze a video
python -m apps.cli.cli analyze <video_path> --output output.mp4 --analytics analytics.json

# CLI - show results
python -m apps.cli.cli show analytics.json --format summary

# Gradio web UI (default port 7860)
python -m apps.gradio.app --port 7860

# REST API (Flask, default port 5000)
python -m apps.api.rest_api
```

## Key Dependencies

- `ultralytics` (YOLO), `torch`, `supervision` for detection
- `opencv-python` (`cv2`) for video I/O
- `numpy` for all numeric/array operations
- `click` for CLI, `gradio` for web UI, `flask` for REST API
- `sklearn` for ML-based team classification

## Architecture

### Processing Pipeline

`VideoProcessor` (pipeline/video_processor.py) is the main orchestrator. It owns all components and drives the pipeline:

1. **Detection** (`detection/yolo_detector.py`) — YOLO-based object detection. Class IDs: player=5, ball=1, referee=6, hoop=3, backboard=0, gameclock=2, period=4, scoreboard=7, team_points=8
2. **Mask Generation** (`detection/mask_generator.py`) — SAM-based or fallback masks
3. **Tracking** (`tracking/tracker.py` → `EnhancedTracker`) — Multi-object tracking with Kalman filtering, appearance features, and basketball-specific logic (5v5 balancing). Internally owns `AdvancedTeamClassificationManager` and `EnhancedPossessionTracker`
4. **Team Classification** (`team_identification/`) — ML classifier + color fallback + basketball rules. `AdvancedTeamClassificationManager` coordinates multiple classifiers. Jersey crops extracted from upper 60% of player bbox (or pose-based torso region)
5. **Possession Tracking** (`analytics/possession/`) — Ball-proximity based. `EnhancedPossessionTracker` adds context tracking and momentum
6. **Play Classification** (`analytics/plays/classifier.py`) — Classifies 11 play types (isolation, pick-and-roll, post-up, fast break, etc.)
7. **Event Detection** (`analytics/events/detector.py`) — Detects shots, rebounds, turnovers
8. **Pose Estimation** (`analytics/pose/`) — Keypoint extraction and basketball action detection (screens, cuts)
9. **Visualization** (`visualization/`) — Frame annotation with team-aware colors, overlays for timeline/statistics/possession

### Frame Processing Flow

`VideoProcessor.process_video()` → reads frames → `BatchOptimizer` groups into adaptive batches → for each frame, `FrameProcessor.process_basketball_frame()` runs steps 1-8 above → writes annotated frames + streams analytics.

### Core Data Models (core/models.py)

- `Detection` — single YOLO detection with bbox, confidence, class_id
- `Track` — tracked object across frames with team_id, velocity, appearance features
- `PossessionInfo` — per-frame possession state with play_type and momentum
- `PlayEvent` — discrete events (shot_attempt, rebound, turnover)
- `PlayClassification` — classified play segment with type, confidence, key players
- `AnalysisResult` — complete output container with all tracks, possessions, plays, events, stats

### Abstract Interfaces (core/interfaces.py)

All major components implement abstract interfaces: `Detector`, `Tracker`, `TeamClassifier`, `PossessionAnalyzer`, `PlayClassifier`, `EventDetector`, `Visualizer`, `FrameProcessor`.

### Three Application Frontends

All frontends create a `ProcessorConfig` → `VideoProcessor` → call `process_video()`:
- **CLI** (`apps/cli/cli.py`) — Click-based, supports analyze/show/list-models
- **Gradio** (`apps/gradio/`) — Web UI with file upload or direct file path input
- **REST API** (`apps/api/rest_api.py`) — Flask endpoints: POST /analyze, GET /health, GET /models

## Configuration

- `ProcessorConfig` (pipeline/video_processor.py) — main config dataclass for the pipeline
- `Settings` (config/settings.py) — global settings, loads from JSON file (`BASKETBALL_CONFIG` env var) or `BASKETBALL_*` env vars
- `config/model_paths.py` — model weight file resolution
- Constants in `core/constants.py` — class IDs, thresholds, court dimensions

## Key Conventions

- All bbox format is `[x1, y1, x2, y2]` as numpy arrays
- Team IDs are 0 and 1; basketball 5v5 balancing enforced (4-6 players per team)
- Ball track uses special ID `-1`
- Visualization colors in BGR format (OpenCV convention)
- Frame sampling: `target_fps` config controls processing speed (default 10fps from source fps)
- Memory cleanup runs every `memory_cleanup_interval` frames (default 50)

## Running Tests

```bash
# Run the full test suite (56 checks across 11 modules)
.venv/bin/python tests/run_tests.py

# Or directly:
.venv/bin/python -c "exec(open('tests/run_tests.py').read())"
```

Note: pytest has trouble with the root `__init__.py` (relative imports). Use the test runner script above.

## Current State / Known Issues

- `scale_threshold()` helper exists in `core/constants.py` but not yet wired into play classifier or event detector thresholds (needs manual call-site updates per use case)
- KMeans is re-initialized per crop in `color_classifier.py` (inefficient)
- O(n²) track matching remains in `masks.py` (shapes.py was optimized)
- Root `__init__.py` uses relative imports incompatible with running pytest directly from project root
