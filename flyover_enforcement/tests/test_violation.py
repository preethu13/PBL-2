"""
Violation Tests for Flyover Enforcement System.

Tests for:
- Violation logic engine (evaluation conditions)
- Deduplicator (cooldown window)
- SQLite violation logger (CRUD operations)
"""

import os
import sys
import cv2
import time
import numpy as np
import pytest
import tempfile

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datetime import datetime
from src.violation.logic_engine import ViolationEngine, Violation
from src.violation.deduplicator import ViolationDeduplicator
from src.violation.logger import ViolationLogger


@pytest.fixture
def sample_frame():
    """Create a test frame."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def violation_engine(tmp_dir):
    """Create a ViolationEngine with temp directory."""
    return ViolationEngine(
        fine_amount=500,
        location="Test Flyover",
        rule="Test Rule",
        snapshot_dir=tmp_dir,
    )


@pytest.fixture
def sample_detection():
    """Create a sample two-wheeler detection."""
    return {
        "bbox": (100, 100, 200, 250),
        "class_id": 3,
        "class_name": "motorcycle",
        "confidence": 0.85,
        "is_restricted": True,
        "label": "TWO-WHEELER",
    }


@pytest.fixture
def allowed_detection():
    """Create a sample allowed vehicle detection."""
    return {
        "bbox": (300, 100, 500, 300),
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.90,
        "is_restricted": False,
        "label": "CAR",
    }


class TestViolationEngine:
    """Test suite for ViolationEngine."""

    def test_two_wheeler_violation(self, violation_engine, sample_detection, sample_frame):
        """Two-wheeler in ROI with plate generates violation."""
        result = violation_engine.evaluate(
            detection=sample_detection,
            plate_text="KL07BJ4545",
            timestamp=datetime.now(),
            frame=sample_frame,
            is_in_roi=True,
        )
        assert result is not None
        assert result.plate == "KL07BJ4545"
        assert result.vehicle_class == "motorcycle"
        assert result.fine_amount == 500

    def test_allowed_vehicle_no_violation(self, violation_engine, allowed_detection, sample_frame):
        """Allowed vehicle (car) does not generate violation."""
        result = violation_engine.evaluate(
            detection=allowed_detection,
            plate_text="KL07BJ4545",
            timestamp=datetime.now(),
            frame=sample_frame,
            is_in_roi=True,
        )
        assert result is None

    def test_out_of_roi_no_violation(self, violation_engine, sample_detection, sample_frame):
        """Two-wheeler outside ROI does not generate violation."""
        result = violation_engine.evaluate(
            detection=sample_detection,
            plate_text="KL07BJ4545",
            timestamp=datetime.now(),
            frame=sample_frame,
            is_in_roi=False,
        )
        assert result is None

    def test_no_plate_manual_review(self, violation_engine, sample_detection, sample_frame):
        """Two-wheeler with no plate text is flagged for manual review."""
        result = violation_engine.evaluate(
            detection=sample_detection,
            plate_text=None,
            timestamp=datetime.now(),
            frame=sample_frame,
            is_in_roi=True,
        )
        assert result is not None
        assert result.plate == "UNKNOWN"
        assert result.status == "manual_review"

    def test_snapshot_saved(self, violation_engine, sample_detection, sample_frame, tmp_dir):
        """Violation snapshot is saved to disk."""
        result = violation_engine.evaluate(
            detection=sample_detection,
            plate_text="KL07BJ4545",
            timestamp=datetime.now(),
            frame=sample_frame,
            is_in_roi=True,
        )
        assert result is not None
        assert os.path.exists(result.snapshot_path)


class TestDeduplicator:
    """Test suite for ViolationDeduplicator."""

    def test_first_occurrence_not_duplicate(self):
        """First occurrence of a plate is not duplicate."""
        dedup = ViolationDeduplicator(cooldown_seconds=300)
        assert dedup.is_duplicate("KL07BJ4545") is False

    def test_second_occurrence_is_duplicate(self):
        """Same plate within cooldown is duplicate."""
        dedup = ViolationDeduplicator(cooldown_seconds=300)
        dedup.record("KL07BJ4545")
        assert dedup.is_duplicate("KL07BJ4545") is True

    def test_different_plates_not_duplicate(self):
        """Different plates are not duplicates of each other."""
        dedup = ViolationDeduplicator(cooldown_seconds=300)
        dedup.record("KL07BJ4545")
        assert dedup.is_duplicate("KL01AB1234") is False

    def test_expired_cooldown(self):
        """Plate after cooldown is not duplicate."""
        dedup = ViolationDeduplicator(cooldown_seconds=1)  # 1 second cooldown
        dedup.record("KL07BJ4545")
        time.sleep(1.5)  # Wait for cooldown
        assert dedup.is_duplicate("KL07BJ4545") is False

    def test_check_and_record(self):
        """Check-and-record returns True for new, False for duplicate."""
        dedup = ViolationDeduplicator(cooldown_seconds=300)
        assert dedup.check_and_record("KL07BJ4545") is True  # New
        assert dedup.check_and_record("KL07BJ4545") is False  # Duplicate

    def test_unknown_plate_not_duplicate(self):
        """UNKNOWN plates are never marked as duplicate."""
        dedup = ViolationDeduplicator(cooldown_seconds=300)
        assert dedup.is_duplicate("UNKNOWN") is False
        dedup.record("UNKNOWN")
        assert dedup.is_duplicate("UNKNOWN") is False  # Still not duplicate

    def test_clear_expired(self):
        """Clear expired entries removes old entries."""
        dedup = ViolationDeduplicator(cooldown_seconds=1)
        dedup.record("KL07BJ4545")
        time.sleep(1.5)
        cleared = dedup.clear_expired()
        assert cleared == 1
        assert dedup.get_cache_size() == 0

    def test_reset(self):
        """Reset clears all entries."""
        dedup = ViolationDeduplicator()
        dedup.record("KL07BJ4545")
        dedup.record("KL01AB1234")
        dedup.reset()
        assert dedup.get_cache_size() == 0


class TestViolationLogger:
    """Test suite for ViolationLogger."""

    def test_log_and_retrieve(self, tmp_dir):
        """Log a violation and retrieve it."""
        db_path = os.path.join(tmp_dir, "test.db")
        logger = ViolationLogger(db_path=db_path)

        violation = Violation(
            id="VIO-TEST-001",
            plate="KL07BJ4545",
            vehicle_class="motorcycle",
            timestamp="2024-01-15 14:30:00",
            fine_amount=500,
            location="Test Flyover",
            status="pending",
        )

        assert logger.log_violation(violation) is True

        # Retrieve
        all_violations = logger.get_all_violations()
        assert len(all_violations) == 1
        assert all_violations[0]['plate'] == "KL07BJ4545"

    def test_get_by_plate(self, tmp_dir):
        """Retrieve violations filtered by plate."""
        db_path = os.path.join(tmp_dir, "test.db")
        logger = ViolationLogger(db_path=db_path)

        # Log two violations
        v1 = Violation(id="VIO-001", plate="KL07BJ4545", timestamp="2024-01-15 14:30:00")
        v2 = Violation(id="VIO-002", plate="KL01AB1234", timestamp="2024-01-15 14:31:00")
        logger.log_violation(v1)
        logger.log_violation(v2)

        results = logger.get_by_plate("KL07BJ4545")
        assert len(results) == 1
        assert results[0]['id'] == "VIO-001"

    def test_update_status(self, tmp_dir):
        """Update violation status."""
        db_path = os.path.join(tmp_dir, "test.db")
        logger = ViolationLogger(db_path=db_path)

        violation = Violation(id="VIO-001", plate="KL07BJ4545",
                              timestamp="2024-01-15 14:30:00", status="pending")
        logger.log_violation(violation)

        assert logger.update_status("VIO-001", "sent") is True

        results = logger.get_all_violations()
        assert results[0]['status'] == "sent"

    def test_get_stats(self, tmp_dir):
        """Statistics computation works."""
        db_path = os.path.join(tmp_dir, "test.db")
        logger = ViolationLogger(db_path=db_path)

        stats = logger.get_stats()
        assert 'total' in stats
        assert 'today' in stats
        assert 'pending' in stats

    def test_export_csv(self, tmp_dir):
        """CSV export creates a file."""
        db_path = os.path.join(tmp_dir, "test.db")
        logger = ViolationLogger(db_path=db_path)

        violation = Violation(id="VIO-001", plate="KL07BJ4545",
                              timestamp="2024-01-15 14:30:00")
        logger.log_violation(violation)

        csv_path = os.path.join(tmp_dir, "export.csv")
        assert logger.export_csv(csv_path) is True
        assert os.path.exists(csv_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
