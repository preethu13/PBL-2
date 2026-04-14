"""
ROI (Region of Interest) Manager for Flyover Enforcement System.

Manages the configurable polygon that defines the flyover entry zone.
Only vehicles detected within this ROI are considered for violation
processing, reducing false positives from adjacent roads.

DIP Techniques:
    - Polygon mask creation (cv2.fillPoly)
    - Bitwise masking
    - Overlay visualization
"""

import cv2
import numpy as np
import yaml
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ROIManager:
    """
    Manages the Region of Interest polygon for the flyover entry zone.

    Loads polygon coordinates from config, creates binary masks,
    and provides visualization utilities.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize ROIManager.

        Args:
            config_path: Path to settings.yaml containing ROI polygon.
                If None, uses a default polygon.
        """
        self.roi_points = None
        self.mask_cache = {}
        self.config_path = config_path

        if config_path:
            self.load_roi(config_path)
        else:
            # Default ROI polygon
            self.roi_points = np.array(
                [[100, 400], [500, 400], [520, 600], [80, 600]], dtype=np.int32
            )
            logger.info("ROIManager initialized with default polygon")

    def load_roi(self, config_path: str) -> None:
        """
        Load ROI polygon coordinates from settings.yaml.

        Expected YAML structure:
            roi:
              entry_polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Args:
            config_path: Path to settings.yaml.
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            polygon = config.get("roi", {}).get("entry_polygon", [])
            if not polygon:
                raise ValueError("No ROI polygon found in config")

            self.roi_points = np.array(polygon, dtype=np.int32)
            self.mask_cache.clear()

            logger.info(f"ROI loaded from {config_path}: {len(self.roi_points)} points")
        except Exception as e:
            logger.error(f"Failed to load ROI from {config_path}: {e}")
            # Fallback to default
            self.roi_points = np.array(
                [[100, 400], [500, 400], [520, 600], [80, 600]], dtype=np.int32
            )
            logger.warning("Using default ROI polygon")

    def create_mask(self, frame: np.ndarray, roi_points: np.ndarray = None) -> np.ndarray:
        """
        Create a binary mask from the ROI polygon.

        Pixels inside the polygon are white (255), outside are black (0).
        Uses caching based on frame dimensions to avoid recomputation.

        Args:
            frame: Input frame (used for dimensions).
            roi_points: Optional alternative polygon. Uses loaded ROI if None.

        Returns:
            Binary mask (single channel, uint8).
        """
        points = roi_points if roi_points is not None else self.roi_points
        h, w = frame.shape[:2]
        cache_key = (h, w)

        if cache_key in self.mask_cache and roi_points is None:
            return self.mask_cache[cache_key]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        if roi_points is None:
            self.mask_cache[cache_key] = mask

        logger.debug(f"ROI mask created ({w}×{h})")
        return mask

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply ROI mask to frame — zero out everything outside the ROI.

        Args:
            frame: Input BGR frame.
            mask: Binary mask. If None, creates one from loaded ROI.

        Returns:
            Masked frame with only ROI region visible.
        """
        if mask is None:
            mask = self.create_mask(frame)

        result = cv2.bitwise_and(frame, frame, mask=mask)
        logger.debug("ROI mask applied to frame")
        return result

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.3,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Visualize ROI polygon on frame with semi-transparent overlay.

        Draws a filled semi-transparent polygon with a solid border.

        Args:
            frame: Input BGR frame.
            color: BGR color for ROI overlay.
            alpha: Transparency of fill (0=invisible, 1=opaque).
            thickness: Border line thickness.

        Returns:
            Frame with ROI visualization overlay.
        """
        overlay = frame.copy()

        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [self.roi_points], color)
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw solid border
        cv2.polylines(result, [self.roi_points], True, color, thickness)

        # Add label
        centroid = np.mean(self.roi_points, axis=0).astype(int)
        cv2.putText(
            result,
            "FLYOVER ENTRY ZONE",
            (centroid[0] - 80, centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        logger.debug("ROI visualization drawn")
        return result

    def is_in_roi(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside the ROI polygon.

        Uses cv2.pointPolygonTest for O(n) point-in-polygon check.

        Args:
            point: (x, y) coordinate to test.

        Returns:
            True if the point is inside the ROI.
        """
        result = cv2.pointPolygonTest(
            self.roi_points.astype(np.float32),
            (float(point[0]), float(point[1])),
            False,
        )
        return result >= 0

    def is_bbox_in_roi(self, bbox: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """
        Check if a bounding box overlaps with the ROI.

        Computes the overlap ratio between the bbox and ROI mask.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.
            threshold: Minimum overlap ratio to consider "in ROI".

        Returns:
            True if overlap ratio exceeds threshold.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Create bbox mask
        # Need a reference frame size — use ROI bounds as estimate
        max_y = max(self.roi_points[:, 1]) + 100
        max_x = max(self.roi_points[:, 0]) + 100
        roi_mask = np.zeros((max(max_y, y2 + 1), max(max_x, x2 + 1)), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [self.roi_points], 255)

        # Compute overlap
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area <= 0:
            return False

        roi_region = roi_mask[y1:y2, x1:x2]
        overlap = np.count_nonzero(roi_region)
        ratio = overlap / bbox_area

        return ratio >= threshold

    def update_roi(self, new_points: List[List[int]]) -> None:
        """
        Update ROI polygon with new coordinates.

        Args:
            new_points: New polygon coordinates [[x1,y1], ...].
        """
        self.roi_points = np.array(new_points, dtype=np.int32)
        self.mask_cache.clear()
        logger.info(f"ROI updated: {len(self.roi_points)} points")
