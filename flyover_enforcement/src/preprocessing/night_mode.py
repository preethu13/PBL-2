"""
Night Mode Processing Module for Flyover Enforcement System.

Implements a specialized DIP pipeline for low-light CCTV footage.
Night conditions are the primary challenge — sensor noise increases
dramatically, contrast drops, and headlight glare creates artifacts.

DIP Pipeline:
    1. Gamma correction (brighten)
    2. Bilateral filter (edge-preserving denoise)
    3. CLAHE on L channel (local contrast)
    4. Unsharp mask (recover details)
    5. Optional dark frame subtraction
"""

import cv2
import numpy as np
import logging

from .enhancement import Enhancer
from .noise_removal import NoiseRemover

logger = logging.getLogger(__name__)


class NightModeProcessor:
    """
    Specialized processor for night/low-light CCTV frames.

    Combines brightening, denoising, contrast enhancement, and
    sharpening in an optimized pipeline order that maximizes
    information recovery while minimizing noise amplification.
    """

    def __init__(
        self,
        brightness_threshold: int = 80,
        gamma: float = 1.8,
        background_model: np.ndarray = None,
    ):
        """
        Initialize NightModeProcessor.

        Args:
            brightness_threshold: Mean brightness below which a frame
                is classified as night. Default 80 (of 255).
            gamma: Gamma correction value for brightening. Values < 1
                brighten; 1/1.8 ≈ 0.56 provides strong brightening.
            background_model: Optional static dark frame for subtraction
                (captures fixed-pattern sensor noise).
        """
        self.brightness_threshold = brightness_threshold
        self.gamma = gamma
        self.background_model = background_model

        self.enhancer = Enhancer()
        self.denoiser = NoiseRemover()

        logger.info(
            f"NightModeProcessor initialized "
            f"(threshold={brightness_threshold}, gamma={gamma})"
        )

    def detect_night(self, frame: np.ndarray) -> bool:
        """
        Determine if a frame is captured in night/low-light conditions.

        Uses mean brightness of the grayscale frame as the metric.
        A mean below the threshold indicates insufficient lighting.

        Args:
            frame: Input BGR frame.

        Returns:
            True if the frame is a night frame (mean brightness < threshold).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = float(np.mean(gray))
        is_night = mean_brightness < self.brightness_threshold

        logger.debug(
            f"Night detection: brightness={mean_brightness:.1f}, "
            f"threshold={self.brightness_threshold}, is_night={is_night}"
        )
        return is_night

    def _dark_frame_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Subtract a dark reference frame to remove fixed-pattern noise.

        CCTV sensors exhibit consistent hot/dead pixels and readout
        noise. Subtracting a dark frame (captured with lens cap on)
        removes this systematic noise component.

        Args:
            frame: Input BGR frame.

        Returns:
            Frame with fixed-pattern noise removed.
        """
        if self.background_model is None:
            return frame

        # Ensure same dimensions
        if frame.shape != self.background_model.shape:
            logger.warning("Dark frame size mismatch, skipping subtraction")
            return frame

        # Subtract in int16 to handle negative values, then clip
        result = cv2.subtract(frame, self.background_model)
        logger.debug("Dark frame subtraction applied")
        return result

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Full night mode DIP pipeline.

        Pipeline order is critical:
        1. Gamma correction FIRST to lift shadows before denoising
           (denoising dark pixels loses too much information).
        2. Bilateral filter to remove noise while keeping edges
           (plate boundaries, vehicle contours).
        3. CLAHE on L-channel for local contrast recovery
           (enhances plate characters, lane markings).
        4. Unsharp mask to recover fine details that bilateral
           may have slightly softened.
        5. Dark frame subtraction if available (removes sensor artifacts).

        Args:
            frame: Input BGR night frame.

        Returns:
            Enhanced frame optimized for vehicle/plate detection.
        """
        logger.info("Processing night mode pipeline")

        # Step 1: Brighten dark regions via gamma correction
        # Using 1/gamma as the power since gamma > 1 means we want brightening
        result = self.enhancer.gamma_correction(frame, gamma=1.0 / self.gamma)

        # Step 2: Edge-preserving denoising
        result = self.denoiser.bilateral_filter(
            result, d=9, sigma_color=75, sigma_space=75
        )

        # Step 3: Local contrast enhancement via CLAHE
        result = self.enhancer.clahe(result, clip_limit=3.0, tile_size=(8, 8))

        # Step 4: Recover detail with unsharp mask
        result = self.enhancer.unsharp_mask(
            result, kernel_size=5, sigma=1.0, strength=1.5
        )

        # Step 5: Optional dark frame subtraction
        if self.background_model is not None:
            result = self._dark_frame_subtraction(result)

        logger.info("Night mode pipeline complete")
        return result

    def visualize_comparison(
        self, original: np.ndarray, processed: np.ndarray
    ) -> np.ndarray:
        """
        Create a side-by-side comparison of original vs processed frame.

        Useful for DIP project demo — shows the dramatic improvement
        that the night mode pipeline achieves on dark footage.

        Args:
            original: Original dark frame.
            processed: Night-mode processed frame.

        Returns:
            Side-by-side comparison image with labels.
        """
        h, w = original.shape[:2]

        # Resize both to same height if needed
        if original.shape[:2] != processed.shape[:2]:
            processed = cv2.resize(processed, (w, h))

        # Create side-by-side canvas
        comparison = np.hstack([original, processed])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)

        # "ORIGINAL" label on left
        cv2.putText(
            comparison, "ORIGINAL", (10, 30),
            font, font_scale, color, thickness, cv2.LINE_AA
        )

        # "ENHANCED (Night Mode)" label on right
        cv2.putText(
            comparison, "ENHANCED (Night Mode)", (w + 10, 30),
            font, font_scale, color, thickness, cv2.LINE_AA
        )

        # Divider line
        cv2.line(comparison, (w, 0), (w, h), (0, 255, 255), 2)

        # Add brightness stats
        orig_bright = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        proc_bright = np.mean(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))

        cv2.putText(
            comparison, f"Brightness: {orig_bright:.0f}", (10, h - 10),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            comparison, f"Brightness: {proc_bright:.0f}", (w + 10, h - 10),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        logger.debug(
            f"Comparison visualization created "
            f"(brightness: {orig_bright:.0f} → {proc_bright:.0f})"
        )
        return comparison

    def visualize_pipeline_steps(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize each step of the night mode pipeline.

        Creates a 2x3 grid showing the transformation at each stage.
        Essential for DIP project demonstration.

        Args:
            frame: Input BGR night frame.

        Returns:
            Grid image showing all pipeline stages.
        """
        h, w = frame.shape[:2]
        # Resize for display
        display_w, display_h = 400, 300
        font = cv2.FONT_HERSHEY_SIMPLEX

        steps = []

        # Step 0: Original
        step0 = cv2.resize(frame, (display_w, display_h))
        cv2.putText(step0, "0: Original", (5, 20), font, 0.5, (0, 255, 0), 1)
        steps.append(step0)

        # Step 1: Gamma correction
        step1_img = self.enhancer.gamma_correction(frame, gamma=1.0 / self.gamma)
        step1 = cv2.resize(step1_img, (display_w, display_h))
        cv2.putText(step1, "1: Gamma Correction", (5, 20), font, 0.5, (0, 255, 0), 1)
        steps.append(step1)

        # Step 2: Bilateral filter
        step2_img = self.denoiser.bilateral_filter(step1_img, d=9, sigma_color=75, sigma_space=75)
        step2 = cv2.resize(step2_img, (display_w, display_h))
        cv2.putText(step2, "2: Bilateral Filter", (5, 20), font, 0.5, (0, 255, 0), 1)
        steps.append(step2)

        # Step 3: CLAHE
        step3_img = self.enhancer.clahe(step2_img, clip_limit=3.0, tile_size=(8, 8))
        step3 = cv2.resize(step3_img, (display_w, display_h))
        cv2.putText(step3, "3: CLAHE", (5, 20), font, 0.5, (0, 255, 0), 1)
        steps.append(step3)

        # Step 4: Unsharp mask
        step4_img = self.enhancer.unsharp_mask(step3_img, kernel_size=5, sigma=1.0, strength=1.5)
        step4 = cv2.resize(step4_img, (display_w, display_h))
        cv2.putText(step4, "4: Unsharp Mask", (5, 20), font, 0.5, (0, 255, 0), 1)
        steps.append(step4)

        # Step 5: Final result
        step5 = cv2.resize(step4_img, (display_w, display_h))
        cv2.putText(step5, "5: Final Result", (5, 20), font, 0.5, (0, 255, 0), 1)
        steps.append(step5)

        # Arrange in 2x3 grid
        row1 = np.hstack(steps[:3])
        row2 = np.hstack(steps[3:])
        grid = np.vstack([row1, row2])

        logger.debug("Pipeline visualization grid created")
        return grid
