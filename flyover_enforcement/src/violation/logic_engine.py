"""
Violation Logic Engine for Flyover Enforcement System.

Evaluates whether a detected two-wheeler in the ROI with a valid
plate reading constitutes a violation. Returns structured Violation
objects for downstream processing (logging, PDF, notification).
"""

import os
import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Structured violation record."""
    id: str = ""
    plate: str = ""
    vehicle_class: str = ""
    timestamp: str = ""
    snapshot_path: str = ""
    confidence: float = 0.0
    fine_amount: int = 500
    location: str = ""
    rule: str = ""
    status: str = "pending"  # pending | sent | manual_review
    pdf_path: str = ""
    bbox: tuple = field(default_factory=tuple)


class ViolationEngine:
    """
    Violation evaluation engine.

    Determines if a detection constitutes a violation based on:
    1. Vehicle is a two-wheeler (restricted class)
    2. Vehicle is within the flyover entry ROI
    3. A valid number plate was recognized
    """

    def __init__(
        self,
        fine_amount: int = 500,
        location: str = "Palarivattom Flyover, Kochi, Kerala",
        rule: str = "Motor Vehicles Act Section 119",
        snapshot_dir: str = "data/violations",
    ):
        """
        Initialize ViolationEngine.

        Args:
            fine_amount: Fine amount in INR.
            location: Location name for the violation report.
            rule: Applicable legal rule/section.
            snapshot_dir: Directory to save violation snapshots.
        """
        self.fine_amount = fine_amount
        self.location = location
        self.rule = rule
        self.snapshot_dir = snapshot_dir
        self.violation_count = 0

        os.makedirs(snapshot_dir, exist_ok=True)
        logger.info(
            f"ViolationEngine initialized "
            f"(fine=₹{fine_amount}, location='{location}')"
        )

    def evaluate(
        self,
        detection: dict,
        plate_text: Optional[str],
        timestamp: datetime,
        frame: np.ndarray,
        is_in_roi: bool = True,
    ) -> Optional[Violation]:
        """
        Evaluate whether a detection constitutes a violation.

        Conditions for a violation:
        1. Vehicle must be a two-wheeler (is_restricted == True)
        2. Vehicle must be in the ROI (flyover entry zone)
        3. Plate text must be valid (not None)

        Args:
            detection: Detection dict with keys: class_name, confidence, bbox, is_restricted.
            plate_text: Recognized plate text (or None).
            timestamp: Detection timestamp.
            frame: Full annotated frame for snapshot.
            is_in_roi: Whether the detection is within the ROI.

        Returns:
            Violation object if conditions met, None otherwise.
        """
        # Condition 1: Must be a two-wheeler
        is_restricted = detection.get("is_restricted", False)
        if not is_restricted:
            logger.debug("Not a two-wheeler, no violation")
            return None

        # Condition 2: Must be in ROI
        if not is_in_roi:
            logger.debug("Two-wheeler not in ROI, no violation")
            return None

        # Condition 3: Plate text (allow even without plate for logging)
        plate = plate_text if plate_text else "UNKNOWN"

        # Generate violation
        self.violation_count += 1
        violation_id = f"VIO-{timestamp.strftime('%Y%m%d%H%M%S')}-{self.violation_count:04d}"

        # Save snapshot
        snapshot_path = self._save_snapshot(frame, violation_id, detection)

        violation = Violation(
            id=violation_id,
            plate=plate,
            vehicle_class=detection.get("class_name", "motorcycle"),
            timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            snapshot_path=snapshot_path,
            confidence=detection.get("confidence", 0.0),
            fine_amount=self.fine_amount,
            location=self.location,
            rule=self.rule,
            status="pending" if plate != "UNKNOWN" else "manual_review",
            bbox=detection.get("bbox", ()),
        )

        logger.info(
            f"VIOLATION DETECTED: {violation_id} | "
            f"Plate={plate} | Class={violation.vehicle_class} | "
            f"Confidence={violation.confidence:.2f}"
        )
        return violation

    def _save_snapshot(
        self,
        frame: np.ndarray,
        violation_id: str,
        detection: dict,
    ) -> str:
        """
        Save annotated violation snapshot.

        Draws the violation bounding box with a red border and
        saves the annotated frame to disk.

        Args:
            frame: Frame to save.
            violation_id: Unique violation ID for filename.
            detection: Detection with bbox for annotation.

        Returns:
            Path to saved snapshot file.
        """
        annotated = frame.copy()

        # Draw violation box
        bbox = detection.get("bbox", None)
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"VIOLATION: {violation_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Add timestamp overlay
        cv2.putText(
            annotated,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        snapshot_path = os.path.join(self.snapshot_dir, f"{violation_id}.jpg")
        cv2.imwrite(snapshot_path, annotated)
        logger.debug(f"Snapshot saved: {snapshot_path}")
        return snapshot_path
