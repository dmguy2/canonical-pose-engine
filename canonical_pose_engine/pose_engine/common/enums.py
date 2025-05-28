# canonical_pose_engine/pose_engine/common/enums.py
from enum import Enum

class PoseState(str, Enum):
    """Defines the operational state of the PoseProcessor."""
    INITIALIZING = "INITIALIZING"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST_TARGET = "LOST_TARGET"
    ERROR = "ERROR"

class LogLevel(str, Enum):
    """Defines logging levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"