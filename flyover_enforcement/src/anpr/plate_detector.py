"""
Number Plate Detector Module for Flyover Enforcement System.

Detects license/number plates within vehicle crop images using
YOLOv8 (fine-tuned for Indian plates) with a contour-based
fallback using aspect ratio filtering.

Detection Pipeline:
    1. Primary: YOLOv8 plate detection model
    2. Fallback: Contour-based detection with aspect ratio filter
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. PlateDetector uses fallback mode.")


class PlateDetector:
    """
    License plate detector optimized for Indian number plates.

    Uses YOLOv8 model fine-tuned for plate localization, with a
    contour-based fallback for environments without the model.
    """

    def __init__(
        self,
        model_path: str = "models/plate_detector.pt",
        conf_threshold: float = 0.4,
        min_aspect: float = 2.0,
        max_aspect: float = 5.5,
    ):
        """
        Initialize PlateDetector.

        Args:
            model_path: Path to YOLOv8 plate detection model.
            conf_threshold: Minimum confidence for plate detection.
            min_aspect: Minimum width/height ratio for contour fallback.
            max_aspect: Maximum width/height ratio for contour fallback.
        """
        self.conf_threshold = conf_threshold
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.model = None

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logger.info(f"Plate detection model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load plate model: {e}. Using fallback.")
        else:
            logger.info("Using contour-based plate detection (fallback)")

    def detect_plate(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect a number plate in a vehicle crop image.

        Args:
            vehicle_crop: Cropped image of the vehicle.

        Returns:
            Bounding box (x1, y1, x2, y2) of the plate, or None.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        if self.model is not None:
            return self._detect_yolo(vehicle_crop)
        else:
            return self._detect_contour(vehicle_crop)

    def _detect_yolo(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect plate using YOLOv8 model.

        Args:
            vehicle_crop: Cropped vehicle image.

        Returns:
            Best plate bbox or None.
        """
        results = self.model(
            vehicle_crop, conf=self.conf_threshold, verbose=False
        )

        best_conf = 0
        best_bbox = None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    best_bbox = (int(x1), int(y1), int(x2), int(y2))

        if best_bbox:
            logger.info(f"YOLO plate detected (conf={best_conf:.2f})")
        else:
            logger.debug("No plate detected by YOLO, trying contour fallback")
            best_bbox = self._detect_contour(vehicle_crop)

        return best_bbox

    def _detect_contour(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Fallback contour-based plate detection using aspect ratio filter.

        Indian number plates have a characteristic aspect ratio of
        approximately 2.5:1 to 5.0:1 (width:height).

        Pipeline:
        1. Convert to grayscale
        2. Bilateral filter (preserve plate edges)
        3. Canny edge detection
        4. Find contours
        5. Filter by aspect ratio and area
        6. Return best candidate

        Args:
            vehicle_crop: Cropped vehicle image.

        Returns:
            Plate bbox (x1, y1, x2, y2) or None.
        """
        h, w = vehicle_crop.shape[:2]
        if h == 0 or w == 0:
            return None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)

        # Dilate to connect edge fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        min_area = (h * w) * 0.005  # Plate is at least 0.5% of crop
        max_area = (h * w) * 0.5    # Plate is at most 50% of crop

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Plates are roughly rectangular (4 corners)
            if len(approx) >= 4 and len(approx) <= 8:
                x, y, bw, bh = cv2.boundingRect(approx)

                if bh == 0:
                    continue

                aspect = bw / bh

                if self.min_aspect <= aspect <= self.max_aspect:
                    candidates.append({
                        "bbox": (x, y, x + bw, y + bh),
                        "area": area,
                        "aspect": aspect,
                    })

        if not candidates:
            logger.debug("No plate candidates found via contour method")
            return None

        # Sort by area (largest plate-like contour wins)
        candidates.sort(key=lambda c: c["area"], reverse=True)
        best = candidates[0]

        logger.info(
            f"Contour plate detected (aspect={best['aspect']:.2f}, area={best['area']:.0f})"
        )
        return best["bbox"]

    def crop_plate(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop the plate region from the frame.

        Adds small padding to ensure plate edges are fully captured.

        Args:
            frame: Source image (vehicle crop or full frame).
            bbox: (x1, y1, x2, y2) plate bounding box.

        Returns:
            Cropped plate image.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add 5% padding
        pad_x = max(2, int((x2 - x1) * 0.05))
        pad_y = max(2, int((y2 - y1) * 0.05))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        plate_crop = frame[y1:y2, x1:x2]
        logger.debug(f"Plate cropped: {plate_crop.shape}")
        return plate_crop

    def draw_plate(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        text: str = "",
    ) -> np.ndarray:
        """
        Draw plate detection box and recognized text on frame.

        Args:
            frame: Input frame.
            bbox: Plate bbox (x1, y1, x2, y2).
            text: Recognized plate text to display.

        Returns:
            Annotated frame.
        """
        result = frame.copy()
        x1, y1, x2, y2 = bbox

        # Yellow box for plate
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if text:
            cv2.putText(
                result,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return result
