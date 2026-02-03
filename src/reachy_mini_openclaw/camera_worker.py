"""Camera worker thread with frame buffering and face tracking.

Provides:
- 30Hz+ camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation
- Latest frame always available for tools
- Smooth return to neutral when face is lost

Based on pollen-robotics/reachy_mini_conversation_app camera worker.
"""

import time
import logging
import threading
from typing import Any, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(self, reachy_mini: ReachyMini, head_tracker: Any = None) -> None:
        """Initialize camera worker.
        
        Args:
            reachy_mini: Connected ReachyMini instance
            head_tracker: Optional head tracker (YOLO or MediaPipe)
        """
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        # Thread-safe frame storage
        self.latest_frame: Optional[NDArray[np.uint8]] = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        # Face tracking timing (for smooth interpolation back to neutral)
        self.last_face_detected_time: Optional[float] = None
        self.interpolation_start_time: Optional[float] = None
        self.interpolation_start_pose: Optional[NDArray[np.float32]] = None
        self.face_lost_delay = 2.0  # seconds to wait before starting interpolation
        self.interpolation_duration = 1.0  # seconds to interpolate back to neutral

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled
        
        # Tracking scale factor (adjust responsiveness)
        self.tracking_scale = 0.6  # Scale down movements for smoother tracking

    def get_latest_frame(self) -> Optional[NDArray[np.uint8]]:
        """Get the latest frame (thread-safe).
        
        Returns:
            Copy of latest frame in BGR format, or None if no frame available
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe).
        
        Returns:
            Tuple of (x, y, z, roll, pitch, yaw) offsets
        """
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking.
        
        Args:
            enabled: Whether to enable face tracking
        """
        self.is_head_tracking_enabled = enabled
        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True)
        self._thread.start()
        logger.info("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Camera worker stopped")

    def _working_loop(self) -> None:
        """Main camera worker loop.
        
        Runs at ~25Hz, captures frames and processes face tracking.
        """
        logger.debug("Starting camera working loop")

        # Neutral pose for interpolation target
        neutral_pose = np.eye(4, dtype=np.float32)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Get frame from robot
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame

                    # Check if face tracking was just disabled
                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        # Face tracking was just disabled - start interpolation to neutral
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    # Update tracking state
                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    # Handle face tracking if enabled and head tracker available
                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        self._process_face_tracking(frame, current_time, neutral_pose)
                    elif self.last_face_detected_time is not None:
                        # Handle interpolation back to neutral when tracking disabled
                        self._interpolate_to_neutral(current_time, neutral_pose)

                # Sleep to maintain ~25Hz
                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")

    def _process_face_tracking(
        self, 
        frame: NDArray[np.uint8], 
        current_time: float,
        neutral_pose: NDArray[np.float32]
    ) -> None:
        """Process face tracking from frame.
        
        Args:
            frame: Current camera frame
            current_time: Current timestamp
            neutral_pose: Neutral pose matrix for interpolation
        """
        eye_center, _ = self.head_tracker.get_head_position(frame)

        if eye_center is not None:
            # Face detected - immediately switch to tracking
            self.last_face_detected_time = current_time
            self.interpolation_start_time = None  # Stop any interpolation

            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            eye_center_norm = (eye_center + 1) / 2
            eye_center_pixels = [
                eye_center_norm[0] * w,
                eye_center_norm[1] * h,
            ]

            # Get the head pose needed to look at the target
            target_pose = self.reachy_mini.look_at_image(
                eye_center_pixels[0],
                eye_center_pixels[1],
                duration=0.0,
                perform_movement=False,
            )

            # Extract translation and rotation from the target pose
            translation = target_pose[:3, 3]
            rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

            # Scale down for smoother tracking
            translation *= self.tracking_scale
            rotation *= self.tracking_scale

            # Thread-safe update of face tracking offsets
            with self.face_tracking_lock:
                self.face_tracking_offsets = [
                    translation[0], translation[1], translation[2],
                    rotation[0], rotation[1], rotation[2],
                ]

        else:
            # No face detected - handle smooth interpolation back to neutral
            self._interpolate_to_neutral(current_time, neutral_pose)

    def _interpolate_to_neutral(
        self, 
        current_time: float,
        neutral_pose: NDArray[np.float32]
    ) -> None:
        """Interpolate face tracking offsets back to neutral when face is lost.
        
        Args:
            current_time: Current timestamp
            neutral_pose: Target neutral pose matrix
        """
        if self.last_face_detected_time is None:
            return

        time_since_face_lost = current_time - self.last_face_detected_time

        if time_since_face_lost >= self.face_lost_delay:
            # Start interpolation if not already started
            if self.interpolation_start_time is None:
                self.interpolation_start_time = current_time
                # Capture current pose as start of interpolation
                with self.face_tracking_lock:
                    current_translation = self.face_tracking_offsets[:3]
                    current_rotation_euler = self.face_tracking_offsets[3:]
                    # Convert to 4x4 pose matrix
                    pose_matrix = np.eye(4, dtype=np.float32)
                    pose_matrix[:3, 3] = current_translation
                    pose_matrix[:3, :3] = R.from_euler(
                        "xyz", current_rotation_euler
                    ).as_matrix()
                    self.interpolation_start_pose = pose_matrix

            # Calculate interpolation progress (t from 0 to 1)
            elapsed_interpolation = current_time - self.interpolation_start_time
            t = min(1.0, elapsed_interpolation / self.interpolation_duration)

            # Interpolate between current pose and neutral pose
            interpolated_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                neutral_pose,
                t,
            )

            # Extract translation and rotation from interpolated pose
            translation = interpolated_pose[:3, 3]
            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

            # Thread-safe update of face tracking offsets
            with self.face_tracking_lock:
                self.face_tracking_offsets = [
                    translation[0], translation[1], translation[2],
                    rotation[0], rotation[1], rotation[2],
                ]

            # If interpolation is complete, reset timing
            if t >= 1.0:
                self.last_face_detected_time = None
                self.interpolation_start_time = None
                self.interpolation_start_pose = None
