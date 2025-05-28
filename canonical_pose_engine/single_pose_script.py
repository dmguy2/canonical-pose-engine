import cv2
import time
import yaml
import numpy as np
from collections import deque
import threading
from typing import Tuple, Optional, Dict
from enum import Enum
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from pydantic import BaseModel, Field
import os

class PoseState(str, Enum):
    INITIALIZING = "INITIALIZING"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST_TARGET = "LOST_TARGET"
    ERROR = "ERROR"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class FrameMetadata(BaseModel):
    frame_id: int
    timestamp: float
    source_resolution: Tuple[int, int]

class PoseResult(BaseModel):
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

class OneEuroFilter:
    def __init__(self, min_cutoff=0.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, te, cutoff):
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)
    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x
        te = t - self.t_prev
        if te < 1e-6:
            return self.x_prev
        alpha_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._smoothing_factor(te, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class CameraManager:
    def __init__(self, config: dict):
        self.config = config
        self._source = config['source']
        self._resolution = tuple(config['resolution'])
        self._target_fps = config['target_fps']
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open camera source: {self._source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        self._buffer = deque(maxlen=config.get('buffer_size', 5))
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._running = False
        self._frame_id = 0
        self._dropped_frames = 0

    def _update(self):
        while self._running:
            grabbed = self._cap.grab()
            if not grabbed:
                self._dropped_frames += 1
                time.sleep(0.01)
                continue
            ret, frame = self._cap.retrieve()
            if ret:
                timestamp = time.perf_counter()
                self._frame_id += 1
                with self._lock:
                    self._buffer.append((frame, self._frame_id, timestamp))
            else:
                self._dropped_frames += 1
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        with self._lock:
            if not self._buffer:
                return None, None
            frame, frame_id, timestamp = self._buffer[-1]
        metadata = FrameMetadata(
            frame_id=frame_id,
            timestamp=timestamp,
            source_resolution=(frame.shape[1], frame.shape[0])
        )
        return frame.copy(), metadata
    def get_stats(self) -> dict:
        return {
            "is_running": self.is_running(),
            "buffer_size": len(self._buffer),
            "dropped_frames": self._dropped_frames,
            "target_fps": self._target_fps,
            "actual_resolution": (self._cap.get(cv2.CAP_PROP_FRAME_WIDTH), self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    def is_running(self) -> bool:
        return self._running
    def __enter__(self):
        self._running = True
        self._thread.start()
        print("CameraManager started.")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        self._thread.join()
        self._cap.release()
        print("CameraManager stopped and resources released.")

class PoseProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.state = PoseState.INITIALIZING
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config['model_complexity'],
            smooth_landmarks=False, 
            enable_segmentation=False,
            min_detection_confidence=config['min_detection_confidence'],
            min_tracking_confidence=config['min_tracking_confidence']
        )
        
        self.filter = OneEuroFilter(**config['filter'])
        self.state = PoseState.SEARCHING
        self.last_detection_time = 0
        self.last_known_landmarks = None

    def process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> PoseResult:
        start_time = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        if results.pose_landmarks:
            self.state = PoseState.TRACKING
            self.last_detection_time = metadata.timestamp
            landmarks_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
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

class Visualizer:
    def __init__(self, config: dict):
        self.config = config
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, result: PoseResult, current_fps: float) -> np.ndarray:
        output_frame = frame.copy()
        lod_reduced = self.config['adaptive_lod'] and current_fps < self.config['lod_threshold_fps']
        if result.smoothed_landmarks is not None and self.config['draw_landmarks']:
            landmarks_proto = self._numpy_to_proto(result.smoothed_landmarks)
            connection_color = (200, 200, 200)
            landmark_color = (0, 255, 0)
            if lod_reduced:
                connection_color = (100, 100, 100)
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
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        left_hand_indices = {
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_PINKY.value,
            self.mp_pose.PoseLandmark.LEFT_INDEX.value,
            self.mp_pose.PoseLandmark.LEFT_THUMB.value
        }
        left_hand_threshold = self.config.get('left_hand_min_visibility_threshold', 0.0)
        for i in range(landmarks_np.shape[0]):
            x, y, z, original_visibility = landmarks_np[i]
            current_visibility = original_visibility
            if i in left_hand_indices and original_visibility < left_hand_threshold:
                current_visibility = 0.0
            lm = landmark_list.landmark.add()
            lm.x = x
            lm.y = y
            lm.z = z
            lm.visibility = current_visibility
        return landmark_list

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found. Expected it in the same directory as the script.")
        return
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file '{config_path}'. {e}")
        return


    fps_history = deque(maxlen=100)
    try:
        with CameraManager(config['camera']) as camera:
            processor = PoseProcessor(config['pose'])
            visualizer = Visualizer(config['visualization'])
            while camera.is_running():
                frame_start_time = time.perf_counter()
                frame, metadata = camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                result = processor.process_frame(frame, metadata)
                frame_end_time = time.perf_counter()
                latency = frame_end_time - frame_start_time
                current_fps = 1.0 / latency if latency > 0 else 0
                fps_history.append(current_fps)
                avg_fps = np.mean(fps_history) if fps_history else 0
                output_frame = visualizer.render(frame, result, avg_fps)
                cv2.imshow('Canonical Pose Engine (Single Script)', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Shutdown signal received.")
                    break
    except (IOError, yaml.YAMLError) as e:
        print(f"ERROR: Failed to initialize. {e}")
    except KeyError as e:
        print(f"ERROR: Missing configuration key: {e}. Please check your '{config_path}'.")
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
    finally:
        if 'processor' in locals() and processor:
            processor.close()
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    main()