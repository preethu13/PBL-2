"""
Motion Filter Module for Flyover Enforcement System.

Implements background subtraction (MOG2) and frame differencing
to isolate moving objects. Pre-filters YOLO detections to only
process vehicles that are actually moving (ignores parked vehicles
or static false positives).

DIP Techniques:
    - MOG2 Background Subtraction
    - Absolute Frame Differencing
    - Morphological Operations (opening, closing)
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class MotionFilter:
    """
    Motion detection and filtering for CCTV frames.

    Uses Gaussian Mixture Model (MOG2) background subtraction
    with morphological cleanup to produce clean motion masks.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16,
        detect_shadows: bool = True,
    ):
        """
        Initialize MotionFilter with MOG2 background subtractor.

        Args:
            history: Number of frames for background model history.
            var_threshold: Threshold on squared Mahalanobis distance
                to decide if a pixel is well-described by the model.
            detect_shadows: If True, shadow pixels are labeled separately.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.prev_frame = None
        self.frame_count = 0

        logger.info(
            f"MotionFilter initialized "
            f"(history={history}, varThreshold={var_threshold})"
        )

    def frame_diff(
        self, prev: np.ndarray, curr: np.ndarray, threshold: int = 25
    ) -> np.ndarray:
        """
        Compute absolute difference between two frames.

        Simple and fast motion detection method. Works well as a
        complement to MOG2 for sudden illumination changes.

        Args:
            prev: Previous BGR frame.
            curr: Current BGR frame.
            threshold: Pixel difference threshold for motion (0-255).

        Returns:
            Binary motion mask.
        """
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if len(prev.shape) == 3 else prev
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if len(curr.shape) == 3 else curr

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        logger.debug("Frame difference computed")
        return motion_mask

    def get_moving_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get foreground motion mask using MOG2 background subtraction.

        The MOG2 model learns the background over time and classifies
        each pixel as foreground (moving) or background (static).
        Shadow pixels (gray value 127) are treated as background.

        Args:
            frame: Input BGR frame.

        Returns:
            Clean binary motion mask (255 = motion, 0 = static).
        """
        # Apply MOG2
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (MOG2 marks shadows as 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        fg_mask = self.morphological_clean(fg_mask)

        self.frame_count += 1
        logger.debug(f"Motion mask generated (frame {self.frame_count})")
        return fg_mask

    def morphological_clean(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean motion mask with morphological operations.

        Pipeline:
        1. MORPH_OPEN: removes small noise blobs (false motion)
        2. MORPH_CLOSE: fills gaps within vehicle silhouettes

        Uses elliptical kernels for natural, round structuring.

        Args:
            mask: Raw binary motion mask.

        Returns:
            Cleaned binary motion mask.
        """
        # Opening: remove small noise (erosion then dilation)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)

        # Closing: fill gaps within vehicles (dilation then erosion)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        logger.debug("Morphological cleanup applied")
        return cleaned

    def filter_detections(
        self,
        detections: list,
        motion_mask: np.ndarray,
        min_overlap: float = 0.2,
    ) -> list:
        """
        Filter YOLO detections to keep only those overlapping with motion.

        Removes static false positives (parked vehicles, sign boards
        misclassified as motorcycles, etc.).

        Args:
            detections: List of detection dicts with 'bbox' key
                containing (x1, y1, x2, y2) coordinates.
            motion_mask: Binary motion mask from get_moving_mask().
            min_overlap: Minimum ratio of bbox pixels that must be
                in motion to keep the detection.

        Returns:
            Filtered list of detections with motion.
        """
        filtered = []

        for det in detections:
            bbox = det.get("bbox", det.get("box", None))
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Clamp to frame bounds
            h, w = motion_mask.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Compute overlap with motion mask
            roi = motion_mask[y1:y2, x1:x2]
            motion_pixels = np.count_nonzero(roi)
            total_pixels = roi.size

            if total_pixels == 0:
                continue

            overlap_ratio = motion_pixels / total_pixels

            if overlap_ratio >= min_overlap:
                det["motion_ratio"] = overlap_ratio
                filtered.append(det)
                logger.debug(
                    f"Detection kept (motion overlap: {overlap_ratio:.2f})"
                )
            else:
                logger.debug(
                    f"Detection filtered out (motion overlap: {overlap_ratio:.2f})"
                )

        logger.info(
            f"Motion filter: {len(filtered)}/{len(detections)} detections kept"
        )
        return filtered

    def visualize_motion(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """
        Overlay motion mask on frame for visualization.

        Motion regions are highlighted in red semi-transparent overlay.

        Args:
            frame: Input BGR frame.
            motion_mask: Binary motion mask.

        Returns:
            Frame with motion overlay.
        """
        overlay = frame.copy()

        # Create colored motion overlay (red)
        motion_colored = np.zeros_like(frame)
        motion_colored[:, :, 2] = motion_mask  # Red channel

        # Blend
        result = cv2.addWeighted(overlay, 0.7, motion_colored, 0.3, 0)
        return result
