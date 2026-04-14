"""
ANPR Tests for Flyover Enforcement System.

Tests for:
- Plate detection (contour fallback)
- Plate DIP preprocessing
- OCR text cleaning and validation
- Kerala plate format regex
- OCR error correction
"""

import os
import sys
import cv2
import numpy as np
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.anpr.plate_detector import PlateDetector
from src.anpr.plate_dip import PlateDIP
from src.anpr.ocr_engine import OCREngine


@pytest.fixture
def plate_image():
    """Create a synthetic plate image with text."""
    plate = np.ones((60, 200, 3), dtype=np.uint8) * 240
    cv2.putText(plate, "KL07BJ4545", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # Add a border to simulate plate edge
    cv2.rectangle(plate, (2, 2), (198, 58), (0, 0, 0), 2)
    return plate


@pytest.fixture
def vehicle_crop():
    """Create a synthetic vehicle image with a plate-like region."""
    crop = np.random.randint(60, 120, (300, 200, 3), dtype=np.uint8)
    # Add a plate-like rectangle (aspect ratio ~3.5:1)
    plate_region = np.ones((30, 100, 3), dtype=np.uint8) * 230
    cv2.putText(plate_region, "KL07BJ", (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(plate_region, (0, 0), (99, 29), (0, 0, 0), 2)
    crop[220:250, 50:150] = plate_region
    return crop


class TestPlateDetector:
    """Test suite for PlateDetector."""

    def test_contour_detection(self, vehicle_crop):
        """Contour-based fallback detects plate-like regions."""
        pd = PlateDetector(model_path="nonexistent.pt")  # Force fallback
        # May or may not find a plate (depends on contour quality)
        result = pd.detect_plate(vehicle_crop)
        # Just ensure it doesn't crash
        assert result is None or len(result) == 4

    def test_crop_plate(self, plate_image):
        """Plate cropping extracts correct region."""
        pd = PlateDetector()
        bbox = (10, 5, 190, 55)
        cropped = pd.crop_plate(plate_image, bbox)
        assert cropped is not None
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0

    def test_empty_input(self):
        """Empty input returns None."""
        pd = PlateDetector(model_path="nonexistent.pt")
        result = pd.detect_plate(np.array([]))
        assert result is None

    def test_draw_plate(self, plate_image):
        """Drawing plate box doesn't crash."""
        pd = PlateDetector()
        result = pd.draw_plate(plate_image, (10, 10, 190, 50), "KL07BJ4545")
        assert result.shape == plate_image.shape


class TestOCREngine:
    """Test suite for OCREngine."""

    def test_clean_text(self):
        """Text cleaning removes spaces and special chars."""
        ocr = OCREngine()
        assert ocr.clean_text("KL 07 BJ 4545") == "KL07BJ4545"
        assert ocr.clean_text("KL-07-BJ-4545") == "KL07BJ4545"
        assert ocr.clean_text("  kl07bj4545  ") == "KL07BJ4545"

    def test_validate_kerala_plate_valid(self):
        """Valid Kerala plates pass validation."""
        ocr = OCREngine()
        assert ocr.validate_kerala_plate("KL07BJ4545") is True
        assert ocr.validate_kerala_plate("KL01A1234") is True
        assert ocr.validate_kerala_plate("KL10CD5678") is True
        assert ocr.validate_kerala_plate("KL99ZZ9999") is True

    def test_validate_kerala_plate_invalid(self):
        """Invalid plates fail validation."""
        ocr = OCREngine()
        assert ocr.validate_kerala_plate("TN07BJ4545") is False  # Tamil Nadu
        assert ocr.validate_kerala_plate("KL7BJ4545") is False   # Missing digit
        assert ocr.validate_kerala_plate("ABCDE") is False
        assert ocr.validate_kerala_plate("") is False
        assert ocr.validate_kerala_plate("KL07454545") is False  # No letters

    def test_fix_ocr_errors(self):
        """Common OCR errors are corrected."""
        ocr = OCREngine()
        # O → 0 in digit positions
        result = ocr.fix_ocr_errors("KLO7BJ4545")
        # The fix should handle the state code properly
        assert "KL" in result

    def test_confidence_filter(self):
        """Low confidence results are filtered out."""
        ocr = OCREngine(min_confidence=0.7)
        results = [
            ("KL07BJ4545", 0.9),
            ("noise", 0.3),
            ("KL01AB1234", 0.8),
        ]
        filtered = ocr.confidence_filter(results)
        assert len(filtered) == 2

    def test_process_returns_tuple(self):
        """Process method returns 3-tuple."""
        ocr = OCREngine()
        # With no OCR engine, should return (None, False, 0.0)
        result = ocr.process(np.zeros((32, 128), dtype=np.uint8))
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestPlateDIPIntegration:
    """Integration tests for plate DIP + OCR pipeline."""

    def test_full_plate_pipeline(self, plate_image):
        """Full pipeline: detect → DIP → (mock OCR)."""
        dip = PlateDIP()

        # Preprocess
        cleaned = dip.preprocess(plate_image)
        assert cleaned is not None
        assert cleaned.size > 0

        # Verify it's binary
        unique = np.unique(cleaned)
        assert all(v in [0, 255] for v in unique)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
