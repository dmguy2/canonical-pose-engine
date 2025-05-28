# canonical_pose_engine/pose_engine/visualization/visualizer.py
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from ..common.models import PoseResult

class Visualizer:
    """Handles all visual output with professional-grade aesthetics and adaptive LOD."""

    def __init__(self, config: dict):
        self.config = config
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, result: PoseResult, current_fps: float) -> np.ndarray:
        """Renders the pose results and HUD onto the frame."""
        output_frame = frame.copy()
        
        # Adaptive Level of Detail (LOD)
        lod_reduced = self.config['adaptive_lod'] and current_fps < self.config['lod_threshold_fps']

        if result.smoothed_landmarks is not None and self.config['draw_landmarks']:
            landmarks_proto = self._numpy_to_proto(result.smoothed_landmarks)
            
            connection_color = (200, 200, 200)
            landmark_color = (0, 255, 0)
            
            if lod_reduced:
                connection_color = (100, 100, 100) # Dim connections on low perf
                
            self.mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=landmarks_proto,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2),
            )
            
        if self.config['draw_hud']:
            self._draw_hud(output_frame, result, current_fps, lod_reduced)
            
        return output_frame

    def _draw_hud(self, frame: np.ndarray, result: PoseResult, fps: float, lod_reduced: bool):
        """Draws the Heads-Up Display with performance metrics."""
        hud_elements = [
            f"FPS: {fps:.1f}",
            f"Processing: {result.processing_time_ms:.1f} ms",
            f"State: {result.status.value}",
        ]
        if lod_reduced:
            hud_elements.append("LOD: REDUCED")

        for i, text in enumerate(hud_elements):
            cv2.putText(frame, text, (10, 30 + i * 30), self.font, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    def _numpy_to_proto(self, landmarks_np: np.ndarray):
        """Converts NumPy array back to MediaPipe's LandmarkList format for drawing."""
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        
        left_hand_indices = {
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_PINKY.value,
            self.mp_pose.PoseLandmark.LEFT_INDEX.value,
            self.mp_pose.PoseLandmark.LEFT_THUMB.value
        }
        # Get the threshold from config, default to 0.0 if not present
        # (though we added it, good practice for robustness)
        left_hand_threshold = self.config.get('left_hand_min_visibility_threshold', 0.0)

        for i in range(landmarks_np.shape[0]):
            x, y, z, original_visibility = landmarks_np[i]
            
            current_visibility = original_visibility

            if i in left_hand_indices and original_visibility < left_hand_threshold:
                # If it's a left hand landmark and below threshold,
                # set its visibility to 0.0 so mp_drawing.draw_landmarks
                # will typically not render it or its connections,
                # while keeping the landmark count consistent.
                current_visibility = 0.0
            
            lm = landmark_list.landmark.add()
            lm.x = x
            lm.y = y
            lm.z = z
            lm.visibility = current_visibility
            
        return landmark_list