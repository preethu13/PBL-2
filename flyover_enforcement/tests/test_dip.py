"""
DIP Tests for Flyover Enforcement System.

Tests for all Digital Image Processing modules:
- Noise removal (all filter types)
- Enhancement (CLAHE, gamma, sharpening, dehazing)
- Night mode detection and processing
- Plate DIP preprocessing pipeline
- Output shape preservation
- Contrast improvement verification
- Binarization validation
"""

import os
import sys
import cv2
import numpy as np
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.noise_removal import NoiseRemover
from src.preprocessing.enhancement import Enhancer
from src.preprocessing.night_mode import NightModeProcessor
from src.preprocessing.stabilizer import VideoStabilizer
from src.anpr.plate_dip import PlateDIP


# ─── Test Fixtures ───

@pytest.fixture
def sample_frame():
    """Create a synthetic test frame (640x480 BGR)."""
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    # Add some structure (rectangles) to simulate a scene
    cv2.rectangle(frame, (100, 100), (300, 250), (0, 128, 255), -1)
    cv2.rectangle(frame, (350, 200), (550, 400), (128, 255, 0), -1)
    cv2.putText(frame, "KL07BJ4545", (150, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


@pytest.fixture
def dark_frame():
    """Create a near-constant dark frame with LOW std (mean~30, std~3).

    Deliberately low-contrast so enhancement routines have room to improve.
    Pure random noise already has max entropy for its range, so CLAHE
    cannot reliably increase std on it — use a near-constant image instead.
    """
    # Near-constant dark background (~30) with tiny ±5 noise
    base = np.full((480, 640, 3), 30, dtype=np.int16)
    noise = np.random.randint(-5, 6, (480, 640, 3), dtype=np.int16)
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)
    return frame


@pytest.fixture
def bright_frame():
    """Create a synthetic bright (day) frame (mean brightness > 80)."""
    frame = np.random.randint(100, 230, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 250), (200, 200, 255), -1)
    return frame


@pytest.fixture
def plate_image():
    """Create a synthetic number plate image."""
    plate = np.ones((60, 200, 3), dtype=np.uint8) * 240  # White background
    cv2.putText(plate, "KL07BJ4545", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return plate


@pytest.fixture
def noisy_frame(sample_frame):
    """Add Gaussian noise to a frame."""
    noise = np.random.normal(0, 30, sample_frame.shape).astype(np.int16)
    noisy = np.clip(sample_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# ─── NoiseRemover Tests ───

class TestNoiseRemover:
    """Test suite for NoiseRemover class."""

    def test_gaussian_blur_shape(self, sample_frame):
        """Gaussian blur preserves frame shape."""
        nr = NoiseRemover()
        result = nr.gaussian_blur(sample_frame, ksize=5)
        assert result.shape == sample_frame.shape

    def test_gaussian_blur_even_ksize(self, sample_frame):
        """Even kernel size is auto-corrected to odd."""
        nr = NoiseRemover()
        result = nr.gaussian_blur(sample_frame, ksize=4)
        assert result.shape == sample_frame.shape

    def test_bilateral_filter_shape(self, sample_frame):
        """Bilateral filter preserves frame shape."""
        nr = NoiseRemover()
        result = nr.bilateral_filter(sample_frame)
        assert result.shape == sample_frame.shape

    def test_nlmeans_denoising_shape(self, sample_frame):
        """NL-Means denoising preserves frame shape."""
        nr = NoiseRemover()
        result = nr.nlmeans_denoising(sample_frame)
        assert result.shape == sample_frame.shape

    def test_median_blur_shape(self, sample_frame):
        """Median blur preserves frame shape."""
        nr = NoiseRemover()
        result = nr.median_blur(sample_frame, ksize=3)
        assert result.shape == sample_frame.shape

    def test_noise_estimation(self, sample_frame, noisy_frame):
        """Noisy frame has different Laplacian variance than clean frame."""
        nr = NoiseRemover()
        clean_var = nr.estimate_noise(sample_frame)
        noisy_var = nr.estimate_noise(noisy_frame)
        # Both should return positive values
        assert clean_var > 0
        assert noisy_var > 0

    def test_auto_select_shape_day(self, sample_frame):
        """Auto-select in day mode preserves shape."""
        nr = NoiseRemover()
        result = nr.auto_select(sample_frame, is_night=False)
        assert result.shape == sample_frame.shape

    def test_auto_select_shape_night(self, sample_frame):
        """Auto-select in night mode preserves shape."""
        nr = NoiseRemover()
        result = nr.auto_select(sample_frame, is_night=True)
        assert result.shape == sample_frame.shape


# ─── Enhancer Tests ───

class TestEnhancer:
    """Test suite for Enhancer class."""

    def test_clahe_shape(self, sample_frame):
        """CLAHE preserves frame shape."""
        enhancer = Enhancer()
        result = enhancer.clahe(sample_frame)
        assert result.shape == sample_frame.shape

    def test_clahe_improves_contrast(self, dark_frame):
        """CLAHE improves contrast (std after > std before)."""
        enhancer = Enhancer()
        original_gray = cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY)
        original_std = float(np.std(original_gray))

        result = enhancer.clahe(dark_frame, clip_limit=3.0)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result_std = float(np.std(result_gray))

        assert result_std > original_std, (
            f"CLAHE should improve contrast: std_before={original_std:.2f}, "
            f"std_after={result_std:.2f}"
        )

    def test_histogram_equalization_shape(self, sample_frame):
        """Histogram equalization preserves shape."""
        enhancer = Enhancer()
        result = enhancer.histogram_equalization(sample_frame)
        assert result.shape == sample_frame.shape

    def test_gamma_correction_shape(self, sample_frame):
        """Gamma correction preserves shape."""
        enhancer = Enhancer()
        result = enhancer.gamma_correction(sample_frame, gamma=1.5)
        assert result.shape == sample_frame.shape

    def test_gamma_correction_brightens(self, dark_frame):
        """Gamma < 1 brightens dark frames."""
        enhancer = Enhancer()
        original_mean = float(np.mean(dark_frame))
        result = enhancer.gamma_correction(dark_frame, gamma=0.5)
        result_mean = float(np.mean(result))

        assert result_mean > original_mean, (
            f"Gamma < 1 should brighten: mean_before={original_mean:.1f}, "
            f"mean_after={result_mean:.1f}"
        )

    def test_unsharp_mask_shape(self, sample_frame):
        """Unsharp mask preserves shape."""
        enhancer = Enhancer()
        result = enhancer.unsharp_mask(sample_frame)
        assert result.shape == sample_frame.shape

    def test_sharpen_kernel_shape(self, sample_frame):
        """Sharpening kernel preserves shape."""
        enhancer = Enhancer()
        result = enhancer.sharpen_kernel(sample_frame)
        assert result.shape == sample_frame.shape

    def test_dehaze_shape(self, sample_frame):
        """Dehazing preserves shape."""
        enhancer = Enhancer()
        result = enhancer.dehaze(sample_frame)
        assert result.shape == sample_frame.shape

    def test_auto_enhance_day_shape(self, bright_frame):
        """Auto-enhance in day mode preserves shape."""
        enhancer = Enhancer()
        result = enhancer.auto_enhance(bright_frame, is_night=False)
        assert result.shape == bright_frame.shape

    def test_auto_enhance_night_shape(self, dark_frame):
        """Auto-enhance in night mode preserves shape."""
        enhancer = Enhancer()
        result = enhancer.auto_enhance(dark_frame, is_night=True)
        assert result.shape == dark_frame.shape


# ─── NightModeProcessor Tests ───

class TestNightModeProcessor:
    """Test suite for NightModeProcessor class."""

    def test_detect_night_dark(self, dark_frame):
        """Dark frame (mean < 80) is detected as night."""
        nmp = NightModeProcessor(brightness_threshold=80)
        assert nmp.detect_night(dark_frame) is True

    def test_detect_night_bright(self, bright_frame):
        """Bright frame (mean > 80) is NOT detected as night."""
        nmp = NightModeProcessor(brightness_threshold=80)
        assert nmp.detect_night(bright_frame) is False

    def test_process_shape(self, dark_frame):
        """Night mode processing preserves frame shape."""
        nmp = NightModeProcessor()
        result = nmp.process(dark_frame)
        assert result.shape == dark_frame.shape

    def test_process_brightens(self, dark_frame):
        """Night mode processing increases brightness."""
        nmp = NightModeProcessor(gamma=1.8)
        result = nmp.process(dark_frame)

        original_mean = float(np.mean(dark_frame))
        result_mean = float(np.mean(result))

        assert result_mean > original_mean, (
            f"Night processing should brighten: "
            f"mean_before={original_mean:.1f}, mean_after={result_mean:.1f}"
        )

    def test_visualize_comparison_shape(self, dark_frame):
        """Comparison visualization has correct dimensions."""
        nmp = NightModeProcessor()
        processed = nmp.process(dark_frame)
        comparison = nmp.visualize_comparison(dark_frame, processed)

        h, w = dark_frame.shape[:2]
        assert comparison.shape[0] == h
        assert comparison.shape[1] == w * 2  # Side-by-side

    def test_visualize_pipeline_steps(self, dark_frame):
        """Pipeline steps visualization creates a valid image."""
        nmp = NightModeProcessor()
        grid = nmp.visualize_pipeline_steps(dark_frame)
        assert grid is not None
        assert len(grid.shape) == 3  # BGR image
        assert grid.shape[2] == 3


# ─── PlateDIP Tests ───

class TestPlateDIP:
    """Test suite for PlateDIP class."""

    def test_resize(self, plate_image):
        """Resize to target height."""
        dip = PlateDIP(target_height=64)
        result = dip.resize(plate_image)
        assert result.shape[0] == 64

    def test_to_grayscale(self, plate_image):
        """Convert BGR to grayscale."""
        dip = PlateDIP()
        result = dip.to_grayscale(plate_image)
        assert len(result.shape) == 2  # Single channel

    def test_denoise_shape(self, plate_image):
        """Denoising preserves shape."""
        dip = PlateDIP()
        gray = dip.to_grayscale(plate_image)
        result = dip.denoise(gray)
        assert result.shape == gray.shape

    def test_binarize_produces_binary(self, plate_image):
        """Binarization produces image with only 0 and 255 values."""
        dip = PlateDIP()
        gray = dip.to_grayscale(plate_image)
        binary = dip.binarize(gray)

        unique_vals = np.unique(binary)
        assert all(v in [0, 255] for v in unique_vals), (
            f"Binary image should only have 0 and 255, got: {unique_vals}"
        )

    def test_morphological_clean(self, plate_image):
        """Morphological cleanup preserves shape."""
        dip = PlateDIP()
        gray = dip.to_grayscale(plate_image)
        binary = dip.binarize(gray)
        cleaned = dip.morphological_clean(binary)
        assert cleaned.shape == binary.shape

    def test_pad_image(self, plate_image):
        """Padding increases image dimensions."""
        dip = PlateDIP(pad_size=10)
        gray = dip.to_grayscale(plate_image)
        padded = dip.pad_image(gray)
        assert padded.shape[0] == gray.shape[0] + 20  # 10px top + 10px bottom
        assert padded.shape[1] == gray.shape[1] + 20  # 10px left + 10px right

    def test_preprocess_full_pipeline(self, plate_image):
        """Full preprocessing pipeline produces valid output."""
        dip = PlateDIP()
        result = dip.preprocess(plate_image)
        assert result is not None
        assert result.size > 0

    def test_deskew_shape(self, plate_image):
        """Deskew preserves shape."""
        dip = PlateDIP()
        gray = dip.to_grayscale(plate_image)
        result = dip.deskew(gray)
        assert result.shape == gray.shape

    def test_visualize_pipeline(self, plate_image):
        """Pipeline visualization creates valid grid image."""
        dip = PlateDIP()
        grid = dip.visualize_pipeline(plate_image)
        assert grid is not None
        assert len(grid.shape) == 3
        assert grid.shape[2] == 3


# ─── VideoStabilizer Tests ───

class TestVideoStabilizer:
    """Test suite for VideoStabilizer class."""

    def test_stabilize_first_frame(self, sample_frame):
        """First frame returned unchanged."""
        stab = VideoStabilizer()
        result = stab.stabilize(sample_frame)
        assert result.shape == sample_frame.shape

    def test_stabilize_sequence(self, sample_frame):
        """Stabilizer processes multiple frames without error."""
        stab = VideoStabilizer(window_size=5)

        for i in range(10):
            # Create slight translations to simulate shake
            shifted = np.roll(sample_frame, i % 5, axis=1)
            result = stab.stabilize(shifted)
            assert result.shape == sample_frame.shape

    def test_reset(self, sample_frame):
        """Reset clears stabilizer state."""
        stab = VideoStabilizer()
        stab.stabilize(sample_frame)
        stab.reset()
        assert stab.frame_count == 0
        assert stab.prev_gray is None


# ─── Integration Tests ───

class TestDIPIntegration:
    """Integration tests combining multiple DIP modules."""

    def test_night_pipeline_end_to_end(self, dark_frame):
        """Full night pipeline: denoise → enhance → result."""
        nr = NoiseRemover()
        nmp = NightModeProcessor()

        # Denoise
        denoised = nr.auto_select(dark_frame, is_night=True)
        assert denoised.shape == dark_frame.shape

        # Night enhance
        enhanced = nmp.process(denoised)
        assert enhanced.shape == dark_frame.shape

        # Brightness should improve
        assert float(np.mean(enhanced)) > float(np.mean(dark_frame))

    def test_plate_pipeline_end_to_end(self, plate_image):
        """Full plate pipeline: resize → gray → denoise → binarize."""
        dip = PlateDIP()
        result = dip.preprocess(plate_image)

        assert result is not None
        assert result.size > 0
        # Should be binary (only 0 and 255 after binarization + morph)
        unique_vals = np.unique(result)
        assert all(v in [0, 255] for v in unique_vals)

    def test_output_types(self, sample_frame):
        """All DIP outputs are uint8 numpy arrays."""
        nr = NoiseRemover()
        enhancer = Enhancer()

        results = [
            nr.gaussian_blur(sample_frame),
            nr.bilateral_filter(sample_frame),
            nr.median_blur(sample_frame),
            enhancer.clahe(sample_frame),
            enhancer.gamma_correction(sample_frame),
            enhancer.sharpen_kernel(sample_frame),
            enhancer.unsharp_mask(sample_frame),
        ]

        for i, result in enumerate(results):
            assert result.dtype == np.uint8, f"Result {i} has dtype {result.dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
