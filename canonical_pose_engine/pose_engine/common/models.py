# canonical_pose_engine/pose_engine/common/models.py
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Dict
from .enums import PoseState

class FrameMetadata(BaseModel):
    """Metadata associated with a single camera frame."""
    frame_id: int
    timestamp: float
    source_resolution: Tuple[int, int]

class PoseResult(BaseModel):
    """Encapsulates the complete result of a single frame's pose processing."""
    timestamp: float
    frame_id: int
    processing_time_ms: float
    status: PoseState
    smoothed_landmarks: Optional[np.ndarray] = None
    world_landmarks: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    prediction_confidence: float = 0.0
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True