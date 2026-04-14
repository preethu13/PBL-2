"""
Main Pipeline Orchestrator for Flyover Enforcement System.

Coordinates all modules in the correct processing order:
1. Video Stabilization
2. Night Detection & Enhancement / Day Enhancement
3. Motion Filtering
4. ROI Masking
5. Vehicle Detection (YOLOv8)
6. Motion-based Detection Filtering
7. For each two-wheeler:
   a. Plate Detection
   b. Plate DIP Preprocessing
   c. OCR Recognition
   d. Violation Evaluation
   e. Deduplication
   f. Logging, PDF Generation, Notification

Supports both video file and live RTSP stream processing.
"""

import os
import sys
import cv2
import yaml
import time
import logging
import numpy as np
from datetime import datetime
from typing import Optional

# Project root = parent of src/
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# Add project root to path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _abs(path: str) -> str:
    """Return absolute path, resolving relative paths against PROJECT_ROOT."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

from src.preprocessing.noise_removal import NoiseRemover
from src.preprocessing.enhancement import Enhancer
from src.preprocessing.night_mode import NightModeProcessor
from src.preprocessing.stabilizer import VideoStabilizer
from src.detection.vehicle_detector import VehicleDetector
from src.detection.roi_manager import ROIManager
from src.detection.motion_filter import MotionFilter
from src.anpr.plate_detector import PlateDetector
from src.anpr.plate_dip import PlateDIP
from src.anpr.ocr_engine import OCREngine
from src.violation.logic_engine import ViolationEngine
from src.violation.deduplicator import ViolationDeduplicator
from src.violation.logger import ViolationLogger
from src.reporting.pdf_generator import FineReportGenerator
from src.reporting.notifier import Notifier

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main processing pipeline that orchestrates all system modules.

    Processes video frames through the complete enforcement pipeline
    from raw CCTV footage to violation detection and reporting.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize Pipeline with all sub-modules.

        Args:
            config_path: Path to main configuration file (absolute or relative
                         to project root).
        """
        config_path = _abs(config_path)
        self.config = self._load_config(config_path)

        # Setup logging
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.config.get("logging", {}).get("log_file", "system.log"),
                    mode="a",
                ),
            ],
        )

        # Initialize all modules
        thresholds = self.config.get("thresholds", {})
        fine_config = self.config.get("fine", {})
        notif_config = self.config.get("notification", {})
        db_path = _abs(
            self.config.get("database", {}).get("path", "data/violations/violations.db")
        )

        # Preprocessing
        self.stabilizer = VideoStabilizer(
            window_size=self.config.get("stabilizer", {}).get("window_size", 30)
        )
        self.noise_remover = NoiseRemover(
            laplacian_threshold=thresholds.get("noise_laplacian_threshold", 100)
        )
        self.enhancer = Enhancer()
        self.night_processor = NightModeProcessor(
            brightness_threshold=thresholds.get("night_brightness_threshold", 80)
        )

        # Detection
        self.roi_manager = ROIManager(config_path)
        self.motion_filter = MotionFilter()
        self.vehicle_detector = VehicleDetector(
            model_path=self.config.get("models", {}).get("vehicle_detector", "yolov8n.pt"),
            conf_threshold=thresholds.get("detection_confidence", 0.45),
            iou_threshold=thresholds.get("iou_threshold", 0.5),
        )

        # ANPR
        self.plate_detector = PlateDetector(
            model_path=self.config.get("models", {}).get("plate_detector", "models/plate_detector.pt")
        )
        self.plate_dip = PlateDIP()
        self.ocr_engine = OCREngine(
            min_confidence=thresholds.get("ocr_confidence", 0.7)
        )

        # Violation
        self.violation_engine = ViolationEngine(
            fine_amount=fine_config.get("amount", 500),
            location=fine_config.get("location", "Palarivattom Flyover, Kochi, Kerala"),
            rule=fine_config.get("rule", "Motor Vehicles Act Section 119"),
        )
        self.deduplicator = ViolationDeduplicator(
            cooldown_seconds=thresholds.get("violation_cooldown_seconds", 300)
        )
        self.logger_db = ViolationLogger(db_path=db_path)

        # Reporting
        self.pdf_generator = FineReportGenerator()
        self.notifier = Notifier(
            smtp_host=notif_config.get("email", {}).get("smtp_host", "smtp.gmail.com"),
            smtp_port=notif_config.get("email", {}).get("smtp_port", 587),
            smtp_user=notif_config.get("email", {}).get("smtp_user", ""),
            smtp_password=notif_config.get("email", {}).get("smtp_password", ""),
            sender_email=notif_config.get("email", {}).get("sender", "enforcement@kerala.gov.in"),
            twilio_sid=notif_config.get("sms", {}).get("twilio_sid", ""),
            twilio_token=notif_config.get("sms", {}).get("twilio_token", ""),
            twilio_from=notif_config.get("sms", {}).get("from_number", ""),
        )

        # Processing state
        self.frame_count = 0
        self.fps_process = self.config.get("camera", {}).get("fps_process", 5)
        self.violations_detected = 0
        self.debug_mode = False

        logger.info("=" * 60)
        logger.info("Flyover Enforcement Pipeline initialized")
        logger.info(f"Config: {config_path}")
        logger.info(f"Processing every {self.fps_process}th frame")
        logger.info("=" * 60)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.

        Pipeline stages:
        1. Stabilize (remove camera shake)
        2. Night detection → night DIP OR day enhancement
        3. Noise removal
        4. Motion mask generation
        5. ROI visualization
        6. Vehicle detection
        7. Motion-based filtering
        8. For each two-wheeler:
           - Plate detection → DIP → OCR
           - Violation evaluation
           - Deduplication → Logging → PDF → Notification

        Args:
            frame: Raw BGR frame from camera/video.

        Returns:
            Annotated frame with all detection visualizations.
        """
        self.frame_count += 1
        timestamp = datetime.now()

        # Working copy
        processed = frame.copy()
        annotated = frame.copy()

        # ─── Stage 1: Stabilization ───
        processed = self.stabilizer.stabilize(processed)

        # ─── Stage 2: Night Detection & Enhancement ───
        is_night = self.night_processor.detect_night(processed)

        if is_night:
            # Night mode: full DIP pipeline
            processed = self.noise_remover.auto_select(processed, is_night=True)
            processed = self.night_processor.process(processed)
        else:
            # Day mode: light enhancement
            processed = self.noise_remover.auto_select(processed, is_night=False)
            processed = self.enhancer.auto_enhance(processed, is_night=False)

        # ─── Stage 3: Motion Mask ───
        motion_mask = self.motion_filter.get_moving_mask(processed)

        # ─── Stage 4: ROI Overlay ───
        annotated = self.roi_manager.draw_roi(processed)

        # ─── Stage 5: Vehicle Detection ───
        detections = self.vehicle_detector.detect(processed)

        # ─── Stage 6: Motion Filtering ───
        detection_dicts = []
        for det in detections:
            detection_dicts.append({
                "bbox": det.bbox,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "is_restricted": det.is_restricted,
                "label": det.label,
            })

        moving_detections = self.motion_filter.filter_detections(
            detection_dicts, motion_mask
        )

        # ─── Stage 7: Process Two-Wheelers ───
        for det in moving_detections:
            # Draw all detections on annotated frame
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = (0, 0, 255) if det["is_restricted"] else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            # Only process two-wheelers for violations
            if not det["is_restricted"]:
                continue

            # Check if in ROI
            bbox_center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2,
            )
            if not self.roi_manager.is_in_roi(bbox_center):
                continue

            # ─── Stage 7a: Plate Detection ───
            vehicle_crop = processed[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            plate_bbox = self.plate_detector.detect_plate(vehicle_crop)
            plate_text = None

            if plate_bbox is not None:
                # ─── Stage 7b: Plate DIP ───
                plate_crop = self.plate_detector.crop_plate(vehicle_crop, plate_bbox)
                if plate_crop.size > 0:
                    cleaned_plate = self.plate_dip.preprocess(plate_crop)

                    # ─── Stage 7c: OCR ───
                    plate_text, is_valid, ocr_conf = self.ocr_engine.process(cleaned_plate)

                    # Draw plate detection on annotated frame
                    px1, py1, px2, py2 = plate_bbox
                    cv2.rectangle(
                        annotated,
                        (x1 + px1, y1 + py1),
                        (x1 + px2, y1 + py2),
                        (0, 255, 255), 2,
                    )
                    if plate_text:
                        cv2.putText(
                            annotated,
                            f"PLATE: {plate_text}",
                            (x1 + px1, y1 + py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2,
                        )

            # ─── Stage 7d: Violation Evaluation ───
            violation = self.violation_engine.evaluate(
                detection=det,
                plate_text=plate_text,
                timestamp=timestamp,
                frame=annotated,
                is_in_roi=True,
            )

            if violation is None:
                continue

            # ─── Stage 7e: Deduplication ───
            if not self.deduplicator.check_and_record(violation.plate):
                logger.info(f"Duplicate violation skipped: {violation.plate}")
                continue

            # ─── Stage 7f: Log, PDF, Notify ───
            self.violations_detected += 1

            # Log to database
            self.logger_db.log_violation(violation)

            # Generate PDF
            pdf_path = self.pdf_generator.generate_pdf(violation)
            if pdf_path:
                violation.pdf_path = pdf_path
                self.logger_db.update_pdf_path(violation.id, pdf_path)

            # Send notifications (async in production)
            self.notifier.notify_violation(violation, pdf_path)

            # Draw violation alert on frame
            cv2.putText(
                annotated,
                f"!! VIOLATION: {violation.plate} !!",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA,
            )

        # ─── Stage 8: Frame Info Overlay ───
        self._draw_info_overlay(annotated, is_night, len(moving_detections))

        return annotated

    def _draw_info_overlay(
        self, frame: np.ndarray, is_night: bool, detections_count: int
    ) -> None:
        """Draw info overlay on the frame (frame count, mode, stats)."""
        h, w = frame.shape[:2]
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Mode: {'NIGHT' if is_night else 'DAY'}",
            f"Detections: {detections_count}",
            f"Violations: {self.violations_detected}",
        ]

        y_offset = 25
        for line in info_lines:
            cv2.putText(
                frame, line, (w - 200, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            y_offset += 20

    def run_on_video(self, video_path: str, output_path: str = None, display: bool = True) -> None:
        """
        Process a video file through the pipeline.

        Args:
            video_path: Path to input video file.
            output_path: Optional path to save output video.
            display: Whether to display processed frames in a window.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {video_path} ({width}×{height} @ {fps}fps, {total_frames} frames)")

        # Output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Process every Nth frame
                if frame_idx % self.fps_process != 0:
                    continue

                # Process frame through pipeline
                annotated = self.process_frame(frame)

                # Write output
                if writer:
                    writer.write(annotated)

                # Display
                if display:
                    cv2.imshow("Flyover Enforcement System", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User quit (pressed 'q')")
                        break
                    elif key == ord('d'):
                        self.debug_mode = not self.debug_mode
                        logger.info(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

                # Progress
                if frame_idx % (fps * 5) == 0:
                    elapsed = time.time() - start_time
                    progress = frame_idx / max(total_frames, 1) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) "
                        f"| Elapsed: {elapsed:.1f}s | Violations: {self.violations_detected}"
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            logger.info(
                f"Video processing complete. "
                f"Frames: {frame_idx}, Violations: {self.violations_detected}, "
                f"Time: {elapsed:.1f}s"
            )

    def run_on_stream(self, rtsp_url: str, display: bool = True) -> None:
        """
        Process a live CCTV stream.

        Args:
            rtsp_url: RTSP URL of the camera stream.
            display: Whether to display frames.
        """
        logger.info(f"Connecting to stream: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            logger.error(f"Failed to connect to stream: {rtsp_url}")
            return

        logger.info("Stream connected. Processing...")
        frame_idx = 0
        reconnect_attempts = 0
        max_reconnects = 5

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    reconnect_attempts += 1
                    if reconnect_attempts > max_reconnects:
                        logger.error("Max reconnection attempts reached")
                        break
                    logger.warning(
                        f"Stream read failed. Reconnecting ({reconnect_attempts}/{max_reconnects})..."
                    )
                    time.sleep(2)
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    continue

                reconnect_attempts = 0
                frame_idx += 1

                if frame_idx % self.fps_process != 0:
                    continue

                annotated = self.process_frame(frame)

                if display:
                    cv2.imshow("Flyover Enforcement - Live", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # Periodic cache cleanup
                if frame_idx % 1000 == 0:
                    self.deduplicator.clear_expired()

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info(f"Stream processing stopped. Total violations: {self.violations_detected}")

    def get_stats(self) -> dict:
        """Get current pipeline statistics."""
        db_stats = self.logger_db.get_stats()
        return {
            "frames_processed": self.frame_count,
            "violations_detected": self.violations_detected,
            "dedup_cache_size": self.deduplicator.get_cache_size(),
            **db_stats,
        }


def main():
    """Entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Flyover Enforcement System - Two-Wheeler Detection Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="Path to config file (relative to project root or absolute)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video file path or RTSP URL. Relative paths resolved from project root."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output annotated video path (optional)"
    )
    parser.add_argument(
        "--no-display", action="store_true", default=True,
        help="Disable OpenCV display window (default: True for headless)"
    )
    parser.add_argument(
        "--display", dest="no_display", action="store_false",
        help="Enable OpenCV display window"
    )

    args = parser.parse_args()

    # ── Resolve all paths against project root ──────────────────────────────
    config_abs = _abs(args.config)

    # Initialize pipeline
    pipeline = Pipeline(config_path=config_abs)

    # Determine source — resolve relative paths
    source = args.source or pipeline.config.get("camera", {}).get("source", "")
    if not source:
        logger.error(
            "No video source specified. "
            "Use --source data/raw_footage/test.mp4 "
            "or set camera.source in config/settings.yaml"
        )
        return

    # Convert relative source paths to absolute
    source = _abs(source)

    # Resolve output path
    output = _abs(args.output) if args.output else None

    logger.info(f"Source  : {source}")
    logger.info(f"Output  : {output}")
    logger.info(f"Display : {not args.no_display}")

    # ── Run ─────────────────────────────────────────────────────────────────
    if source.startswith("rtsp://") or source.startswith("http://"):
        pipeline.run_on_stream(source, display=not args.no_display)
    else:
        if not os.path.exists(source):
            logger.error(
                f"Video file not found: {source}\n"
                f"Place your footage in: {os.path.join(PROJECT_ROOT, 'data', 'raw_footage')}"
            )
            return
        pipeline.run_on_video(source, output, display=not args.no_display)


if __name__ == "__main__":
    main()
