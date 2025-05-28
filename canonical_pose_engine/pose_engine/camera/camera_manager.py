# canonical_pose_engine/pose_engine/camera/camera_manager.py
import cv2
import time
import threading
import numpy as np
from collections import deque
from typing import Tuple, Optional
from ..common.models import FrameMetadata

class CameraManager:
    """Manages high-performance, non-blocking camera I/O in a separate thread."""

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
        """The core frame-grabbing loop running in a dedicated thread."""
        while self._running:
            grabbed = self._cap.grab()
            if not grabbed:
                self._dropped_frames += 1
                time.sleep(0.01) # Avoid busy-waiting on error
                continue
            
            # Always retrieve the frame after a successful grab.
            # The deque's maxlen will handle dropping the oldest frame if the buffer is full.
            ret, frame = self._cap.retrieve()
            if ret:
                timestamp = time.perf_counter()
                self._frame_id += 1
                with self._lock:
                    self._buffer.append((frame, self._frame_id, timestamp))
            else:
                # Frame grab was successful, but retrieve failed.
                self._dropped_frames += 1


    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Returns the latest frame and its metadata from the buffer."""
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
        """Returns comprehensive camera health and performance statistics."""
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
        """Context manager entry to start the camera thread."""
        self._running = True
        self._thread.start()
        print("CameraManager started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit to gracefully stop and release resources."""
        self._running = False
        self._thread.join()
        self._cap.release()
        print("CameraManager stopped and resources released.")