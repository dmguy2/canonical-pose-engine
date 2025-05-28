# canonical_pose_engine/pose_engine/processing/pose_processor.py
import mediapipe as mp
import numpy as np
import time
import cv2 # Added import for cv2.cvtColor
from ..common.models import PoseResult, FrameMetadata
from ..common.enums import PoseState
from .one_euro_filter import OneEuroFilter

class PoseProcessor:
    """Orchestrates MediaPipe pose estimation with advanced filtering and state management."""

    def __init__(self, config: dict):
        self.config = config
        self.state = PoseState.INITIALIZING
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config['model_complexity'],
            smooth_landmarks=False, # We use a superior, custom filter
            enable_segmentation=False,
            min_detection_confidence=config['min_detection_confidence'],
            min_tracking_confidence=config['min_tracking_confidence']
        )
        
        self.filter = OneEuroFilter(**config['filter'])
        self.state = PoseState.SEARCHING
        self.last_detection_time = 0
        self.last_known_landmarks = None

    def process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> PoseResult:
        """Processes a single frame to detect and track pose."""
        start_time = time.perf_counter()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Performance optimization
        
        results = self.pose.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        if results.pose_landmarks:
            self.state = PoseState.TRACKING
            self.last_detection_time = metadata.timestamp
            
            landmarks_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
            
            # Apply One-Euro Filter for buttery-smooth motion
            smoothed_landmarks = self.filter(landmarks_np, metadata.timestamp)
            
            self.last_known_landmarks = smoothed_landmarks
            
            return PoseResult(
                timestamp=metadata.timestamp,
                frame_id=metadata.frame_id,
                processing_time_ms=processing_time_ms,
                status=self.state,
                smoothed_landmarks=smoothed_landmarks,
                world_landmarks=np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]) if results.pose_world_landmarks else None,
                confidence_scores=np.array([lm.visibility for lm in results.pose_landmarks.landmark]),
            )
        else:
            self.state = PoseState.SEARCHING
            return PoseResult(
                timestamp=metadata.timestamp,
                frame_id=metadata.frame_id,
                processing_time_ms=processing_time_ms,
                status=self.state
            )

    def close(self):
        self.pose.close()