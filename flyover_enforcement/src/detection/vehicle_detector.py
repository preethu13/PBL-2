"""
Vehicle Detection Module for Flyover Enforcement System.

Uses YOLOv8 (Ultralytics) to detect and classify vehicles in
CCTV frames. Identifies two-wheelers (motorcycles, bicycles) as
restricted vehicles and cars/buses/trucks as allowed.

Detection Pipeline:
    1. Run YOLOv8 inference on frame
    2. Filter by confidence threshold
    3. Classify as two-wheeler or allowed vehicle
    4. Return structured detection results
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try importing ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. VehicleDetector will use fallback mode.")


@dataclass
class Detection:
    """Structured detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    is_restricted: bool = False
    label: str = ""


# COCO class mappings relevant to vehicles
COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Two-wheeler class IDs (restricted on flyover)
TWO_WHEELER_IDS = {1, 3}  # bicycle, motorcycle

# Allowed vehicle class IDs
ALLOWED_VEHICLE_IDS = {2, 5, 7}  # car, bus, truck


class VehicleDetector:
    """
    YOLOv8-based vehicle detector with two-wheeler classification.

    Loads a YOLOv8 model, runs inference, and classifies detections
    as restricted (two-wheelers) or allowed (cars, buses, trucks).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = None,
    ):
        """
        Initialize VehicleDetector.

        Args:
            model_path: Path to YOLOv8 weights file.
            conf_threshold: Minimum confidence to accept detection.
            iou_threshold: IoU threshold for NMS.
            device: Device to run inference on ('cpu', 'cuda', or None for auto).
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = device

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                if device:
                    self.model.to(device)
                logger.info(f"YOLOv8 model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                logger.warning("Vehicle detection will use fallback contour-based method")
        else:
            logger.warning("YOLO not available, using fallback detection")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run vehicle detection on a frame.

        Args:
            frame: Input BGR frame.

        Returns:
            List of Detection objects for all detected vehicles.
        """
        if self.model is not None:
            return self._detect_yolo(frame)
        else:
            return self._detect_fallback(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference and extract vehicle detections.

        Args:
            frame: Input BGR frame.

        Returns:
            List of Detection objects.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                # Only keep vehicle classes
                if class_id not in COCO_VEHICLE_CLASSES:
                    continue

                # Extract bbox coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                class_name = COCO_VEHICLE_CLASSES[class_id]
                is_restricted = class_id in TWO_WHEELER_IDS

                label = "TWO-WHEELER" if is_restricted else class_name.upper()

                detection = Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    is_restricted=is_restricted,
                    label=label,
                )
                detections.append(detection)

        logger.info(
            f"YOLO detected {len(detections)} vehicles "
            f"({sum(1 for d in detections if d.is_restricted)} two-wheelers)"
        )
        return detections

    def _detect_fallback(self, frame: np.ndarray) -> List[Detection]:
        """
        Fallback contour-based vehicle detection when YOLO is unavailable.

        Uses background subtraction + contour analysis. Less accurate
        but functional for testing without GPU/model weights.

        Args:
            frame: Input BGR frame.

        Returns:
            List of Detection objects (class will be 'unknown').
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Simple edge-based detection
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:  # Skip small contours
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Rough heuristic: two-wheelers are taller than wide
            if 0.3 < aspect_ratio < 1.5 and area < 15000:
                class_name = "motorcycle"
                class_id = 3
                is_restricted = True
            else:
                class_name = "car"
                class_id = 2
                is_restricted = False

            detection = Detection(
                bbox=(x, y, x + w, y + h),
                class_id=class_id,
                class_name=class_name,
                confidence=0.5,
                is_restricted=is_restricted,
                label="TWO-WHEELER" if is_restricted else "VEHICLE",
            )
            detections.append(detection)

        logger.info(f"Fallback detected {len(detections)} objects")
        return detections

    @staticmethod
    def is_two_wheeler(class_name: str) -> bool:
        """
        Check if a class name represents a two-wheeler.

        Args:
            class_name: Vehicle class name (e.g., 'motorcycle', 'bicycle').

        Returns:
            True if the vehicle is a two-wheeler.
        """
        two_wheeler_names = {"motorcycle", "bicycle", "scooter", "moped", "e-bike"}
        return class_name.lower() in two_wheeler_names

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Draw bounding boxes with labels on frame.

        Restricted vehicles (two-wheelers) are drawn in RED.
        Allowed vehicles are drawn in GREEN.

        Args:
            frame: Input BGR frame.
            detections: List of Detection objects.

        Returns:
            Annotated frame with detection overlays.
        """
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            if det.is_restricted:
                color = (0, 0, 255)   # Red for violations
                border = 3
            else:
                color = (0, 255, 0)   # Green for allowed
                border = 2

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, border)

            # Draw label background
            label_text = f"{det.label} {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                result,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return result

    def get_vehicle_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract a vehicle crop from the frame.

        Adds a small padding around the bbox for context.

        Args:
            frame: Input BGR frame.
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            Cropped vehicle image.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add 10% padding
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return frame[y1:y2, x1:x2]
