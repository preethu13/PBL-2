"""
Video Stabilization Module for Flyover Enforcement System.

Implements optical flow-based video stabilization to compensate
for camera shake/vibration in CCTV footage. Wind-induced vibration
on flyover-mounted cameras is common.

DIP Techniques:
    - Lucas-Kanade Optical Flow (cv2.calcOpticalFlowPyrLK)
    - Affine transformation estimation
    - Trajectory smoothing via moving average
"""

import cv2
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)


class VideoStabilizer:
    """
    Optical flow-based video stabilizer.

    Tracks feature points between consecutive frames, computes the
    inter-frame motion (translation + rotation), smooths the
    cumulative trajectory, and warps frames to cancel jitter.
    """

    def __init__(self, window_size: int = 30, max_features: int = 200):
        """
        Initialize VideoStabilizer.

        Args:
            window_size: Number of frames for trajectory smoothing.
                Larger = smoother but more latency.
            max_features: Maximum feature points to track per frame.
        """
        self.window_size = window_size
        self.max_features = max_features

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        # State
        self.prev_gray = None
        self.prev_points = None
        self.transforms = deque(maxlen=window_size * 2)
        self.trajectory = np.zeros((1, 3), dtype=np.float64)  # dx, dy, da
        self.smoothed_trajectory = np.zeros((1, 3), dtype=np.float64)
        self.frame_count = 0

        # Trajectory history for smoothing
        self.trajectory_history = deque(maxlen=window_size)

        logger.info(
            f"VideoStabilizer initialized "
            f"(window={window_size}, max_features={max_features})"
        )

    def reset(self):
        """Reset stabilizer state for a new video."""
        self.prev_gray = None
        self.prev_points = None
        self.transforms.clear()
        self.trajectory = np.zeros((1, 3), dtype=np.float64)
        self.smoothed_trajectory = np.zeros((1, 3), dtype=np.float64)
        self.frame_count = 0
        self.trajectory_history.clear()
        logger.info("Stabilizer state reset")

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect good features to track using Shi-Tomasi corners.

        Args:
            gray: Grayscale frame.

        Returns:
            Array of corner points (N, 1, 2).
        """
        points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        if points is None:
            points = np.empty((0, 1, 2), dtype=np.float32)
        return points

    def _estimate_transform(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_points: np.ndarray
    ) -> tuple:
        """
        Estimate affine transform between two frames using optical flow.

        Args:
            prev_gray: Previous grayscale frame.
            curr_gray: Current grayscale frame.
            prev_points: Feature points from previous frame.

        Returns:
            Tuple of (dx, dy, da) — translation x, translation y, rotation angle.
        """
        if prev_points is None or len(prev_points) == 0:
            return 0.0, 0.0, 0.0

        # Track points with Lucas-Kanade optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params
        )

        if curr_points is None:
            return 0.0, 0.0, 0.0

        # Filter good matches
        status = status.flatten()
        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_prev) < 3:
            return 0.0, 0.0, 0.0

        # Estimate rigid transform (translation + rotation)
        transform_matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)

        if transform_matrix is None:
            return 0.0, 0.0, 0.0

        # Extract translation and rotation
        dx = transform_matrix[0, 2]
        dy = transform_matrix[1, 2]
        da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

        return float(dx), float(dy), float(da)

    def _smooth_trajectory(self, trajectory_point: np.ndarray) -> np.ndarray:
        """
        Smooth trajectory using a moving average window.

        Args:
            trajectory_point: Current cumulative trajectory [dx, dy, da].

        Returns:
            Smoothed trajectory point.
        """
        self.trajectory_history.append(trajectory_point.copy())

        if len(self.trajectory_history) == 0:
            return trajectory_point

        # Moving average over the window
        window = np.array(list(self.trajectory_history))
        smoothed = np.mean(window, axis=0)

        return smoothed

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a single frame in a video sequence.

        Must be called sequentially for each frame in order.
        The first frame is returned unchanged (establishes baseline).

        Args:
            frame: Input BGR frame.

        Returns:
            Stabilized BGR frame with camera shake removed.
        """
        h, w = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        # First frame — initialize
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_points = self._detect_features(curr_gray)
            logger.debug("Stabilizer: first frame initialized")
            return frame

        # Estimate inter-frame transform
        dx, dy, da = self._estimate_transform(
            self.prev_gray, curr_gray, self.prev_points
        )

        # Accumulate trajectory
        self.trajectory += np.array([[dx, dy, da]])

        # Smooth trajectory
        smoothed = self._smooth_trajectory(self.trajectory[0])

        # Compute correction: difference between actual and smoothed
        diff = smoothed - self.trajectory[0]
        d_dx, d_dy, d_da = diff

        # Build stabilization affine matrix
        cos_a = np.cos(d_da)
        sin_a = np.sin(d_da)
        stabilize_matrix = np.array(
            [
                [cos_a, -sin_a, d_dx],
                [sin_a, cos_a, d_dy],
            ],
            dtype=np.float64,
        )

        # Warp frame
        stabilized = cv2.warpAffine(
            frame,
            stabilize_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Update state for next frame
        self.prev_gray = curr_gray
        # Re-detect features periodically (every 10 frames) to avoid drift
        if self.frame_count % 10 == 0:
            self.prev_points = self._detect_features(curr_gray)
        else:
            self.prev_points = self._detect_features(curr_gray)

        logger.debug(
            f"Frame {self.frame_count} stabilized "
            f"(dx={dx:.2f}, dy={dy:.2f}, da={da:.4f})"
        )
        return stabilized
