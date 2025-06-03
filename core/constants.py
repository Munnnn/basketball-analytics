"""
Constants for basketball analytics system
"""

# Object class IDs (from YOLO model)
BACKBOARD_ID = 0
BALL_ID = 1
GAMECLOCK_ID = 2
HOOP_ID = 3
PERIOD_ID = 4
PLAYER_ID = 5
REF_ID = 6
SCOREBOARD_ID = 7
TEAM_POINTS_ID = 8

# Court dimensions (in pixels for standard video)
COURT_WIDTH = 1280
COURT_HEIGHT = 720

# Video processing
DEFAULT_FPS = 30.0
DEFAULT_BATCH_SIZE = 20

# Tracking parameters
TRACK_ACTIVATION_THRESHOLD = 0.4
LOST_TRACK_BUFFER = 90
MINIMUM_MATCHING_THRESHOLD = 0.2
MINIMUM_CONSECUTIVE_FRAMES = 3
MAX_TRACKS = 15

# Team classification
MIN_PLAYERS_FOR_TEAM_INIT = 6
TEAM_CONFIDENCE_THRESHOLD = 0.6

# Possession tracking
BALL_PROXIMITY_THRESHOLD = 80
POSSESSION_CHANGE_THRESHOLD = 8
MIN_POSSESSION_DURATION = 5

# Play classification
PLAY_TYPES = {
    0: "Isolation",
    1: "Pick and Roll",
    2: "Post Up",
    3: "Fast Break",
    4: "Spot Up",
    5: "Off Screen",
    6: "Handoff",
    7: "Cut",
    8: "Off Rebound",
    9: "Transition",
    10: "Half Court Set"
}

# Event detection
SHOT_DISTANCE_THRESHOLD = 80
REBOUND_PROXIMITY_THRESHOLD = 120
EVENT_COOLDOWN_FRAMES = 30

# Visualization colors (BGR format)
TEAM_COLORS = {
    0: [(0, 0, 255), (0, 0, 220), (0, 0, 180)],  # Red team
    1: [(255, 0, 0), (220, 0, 0), (180, 0, 0)]   # Blue team
}

DEFAULT_COLORS = [(128, 128, 128), (192, 192, 192), (96, 96, 96)]

# Memory optimization
MEMORY_CLEANUP_INTERVAL = 50
MAX_FRAME_HISTORY = 100
