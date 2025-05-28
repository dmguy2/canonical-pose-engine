# canonical_pose_engine/main.py
import cv2
import time
import yaml
import numpy as np
from collections import deque

from pose_engine.camera.camera_manager import CameraManager
from pose_engine.processing.pose_processor import PoseProcessor
from pose_engine.visualization.visualizer import Visualizer

def main():
    """
    The canonical main application loop.
    Initializes, runs, and gracefully shuts down the pose engine components.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    fps_history = deque(maxlen=100)
    
    try:
        with CameraManager(config['camera']) as camera:
            processor = PoseProcessor(config['pose'])
            visualizer = Visualizer(config['visualization'])
            
            while camera.is_running():
                frame_start_time = time.perf_counter()
                
                frame, metadata = camera.get_frame()
                if frame is None:
                    time.sleep(0.001) # Wait briefly if no frame is available
                    continue
                
                # --- Core Processing Pipeline ---
                result = processor.process_frame(frame, metadata)
                
                # --- FPS Calculation ---
                frame_end_time = time.perf_counter()
                latency = frame_end_time - frame_start_time
                current_fps = 1.0 / latency if latency > 0 else 0
                fps_history.append(current_fps)
                avg_fps = np.mean(fps_history)
                
                # --- Visualization ---
                output_frame = visualizer.render(frame, result, avg_fps)
                
                # --- Display Output ---
                cv2.imshow('Canonical Pose Engine', output_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Shutdown signal received.")
                    break

    except (IOError, yaml.YAMLError) as e:
        print(f"ERROR: Failed to initialize. {e}")
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    main()