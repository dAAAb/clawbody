"""Head tracker factory for selecting the best available tracker."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_head_tracker(tracker_type: Optional[str] = None) -> Optional[Any]:
    """Get a head tracker instance based on availability and preference.
    
    Args:
        tracker_type: One of 'yolo', 'mediapipe', or None for auto-detect
        
    Returns:
        Head tracker instance or None if no tracker available
    """
    if tracker_type == "yolo":
        return _try_yolo_tracker()
    elif tracker_type == "mediapipe":
        return _try_mediapipe_tracker()
    elif tracker_type is None:
        # Auto-detect: try MediaPipe first (lighter), then YOLO
        tracker = _try_mediapipe_tracker()
        if tracker is not None:
            return tracker
        return _try_yolo_tracker()
    else:
        logger.warning(f"Unknown tracker type: {tracker_type}")
        return None


def _try_yolo_tracker() -> Optional[Any]:
    """Try to create a YOLO head tracker."""
    try:
        from reachy_mini_openclaw.vision.yolo_head_tracker import HeadTracker
        tracker = HeadTracker()
        logger.info("Using YOLO head tracker")
        return tracker
    except ImportError as e:
        logger.debug(f"YOLO tracker not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize YOLO tracker: {e}")
        return None


def _try_mediapipe_tracker() -> Optional[Any]:
    """Try to create a MediaPipe head tracker."""
    try:
        # First try the toolbox version
        from reachy_mini_toolbox.vision import HeadTracker
        tracker = HeadTracker()
        logger.info("Using MediaPipe head tracker (from toolbox)")
        return tracker
    except ImportError:
        pass
    
    try:
        # Fall back to our own MediaPipe implementation
        from reachy_mini_openclaw.vision.mediapipe_tracker import HeadTracker
        tracker = HeadTracker()
        logger.info("Using MediaPipe head tracker")
        return tracker
    except ImportError as e:
        logger.debug(f"MediaPipe tracker not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize MediaPipe tracker: {e}")
        return None
