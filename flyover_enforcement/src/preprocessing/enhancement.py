"""
Image Enhancement Module for Flyover Enforcement System.

Implements contrast enhancement, sharpening, gamma correction, and
dehazing techniques. The auto_enhance pipeline orchestrates these
for optimal frame quality before detection and ANPR.

DIP Techniques:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Full Histogram Equalization
    - Gamma Correction (power-law transform via LUT)
    - Unsharp Masking (detail recovery)
    - Kernel-based Sharpening
    - Dark Channel Prior Dehazing (He et al.)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Enhancer:
    """
    Multi-technique image enhancer for CCTV frames.

    Provides individual enhancement methods and an orchestrated
    auto_enhance pipeline that adapts to day/night conditions.
    """

    def __init__(self):
        """Initialize Enhancer with default parameters."""
        logger.info("Enhancer initialized")

    def clahe(
        self,
        frame: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: tuple = (8, 8),
    ) -> np.ndarray:
        """
        Apply CLAHE on L-channel of LAB colorspace.

        CLAHE prevents over-amplification of noise (unlike standard HE)
        by clipping the histogram at a defined limit. Applied to the
        luminance channel to preserve color fidelity.

        Args:
            frame: Input BGR frame.
            clip_limit: Threshold for contrast limiting.
            tile_size: Size of grid for histogram equalization.

        Returns:
            Enhanced BGR frame with improved local contrast.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe_obj.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        logger.debug(f"CLAHE applied (clip_limit={clip_limit}, tile_size={tile_size})")
        return result

    def histogram_equalization(self, frame: np.ndarray) -> np.ndarray:
        """
        Full histogram equalization for severe low-contrast frames.

        Spreads intensity values across the full 0-255 range. More
        aggressive than CLAHE — use only when contrast is extremely poor.

        Args:
            frame: Input BGR frame.

        Returns:
            Histogram-equalized BGR frame.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(ycrcb)

        y_equalized = cv2.equalizeHist(y_channel)

        ycrcb_equalized = cv2.merge([y_equalized, cr, cb])
        result = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

        logger.debug("Full histogram equalization applied")
        return result

    def gamma_correction(self, frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        Power-law (gamma) transform using a lookup table.

        Gamma < 1: brightens dark regions (night mode).
        Gamma > 1: darkens bright regions (overexposed day).
        LUT implementation makes this O(1) per pixel — very fast.

        Args:
            frame: Input BGR frame.
            gamma: Gamma value. Use 0.4-0.7 to brighten (gamma < 1 lifts
            shadows; gamma > 1 darkens). Standard convention:
            output = 255 * (input/255)^gamma

        Returns:
            Gamma-corrected frame.
        """
        # Build LUT using standard power-law: output = 255 * (input/255)^gamma
        # gamma < 1 → brightens (e.g. 0.5 → sqrt → lifts dark pixels)
        # gamma > 1 → darkens
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype("uint8")

        result = cv2.LUT(frame, lut)
        logger.debug(f"Gamma correction applied (gamma={gamma})")
        return result

    def unsharp_mask(
        self,
        frame: np.ndarray,
        kernel_size: int = 5,
        sigma: float = 1.0,
        strength: float = 1.5,
    ) -> np.ndarray:
        """
        Unsharp masking to recover blurred edge details.

        Critical for night mode frames where denoising may over-smooth.
        Subtracts a blurred version from the original to isolate high
        frequencies, then adds them back with amplification.

        Formula: sharpened = original + strength * (original - blurred)

        Args:
            frame: Input BGR frame.
            kernel_size: Gaussian kernel size for blur estimation.
            sigma: Gaussian sigma.
            strength: Amplification factor for high-frequency details.

        Returns:
            Sharpened frame.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)

        # Compute in float to prevent overflow/clipping artefacts
        sharpened = cv2.addWeighted(
            frame, 1.0 + strength, blurred, -strength, 0
        )

        logger.debug(
            f"Unsharp mask applied (ksize={kernel_size}, σ={sigma}, strength={strength})"
        )
        return sharpened

    def sharpen_kernel(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply a 3x3 sharpening convolution kernel.

        Classic high-pass sharpening: centre pixel gets 5x weight,
        neighbors get -1. Simple and fast for mild sharpening.

        Kernel:
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]]

        Args:
            frame: Input BGR frame.

        Returns:
            Sharpened frame.
        """
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(frame, -1, kernel)
        logger.debug("3x3 sharpening kernel applied")
        return result

    def dehaze(self, frame: np.ndarray, omega: float = 0.95, t_min: float = 0.1) -> np.ndarray:
        """
        Dark Channel Prior dehazing (He et al., CVPR 2009).

        Removes fog/haze/mist that degrades visibility in outdoor CCTV.
        Kerala's humid climate frequently causes haze on flyover cameras.

        Algorithm:
            1. Compute dark channel: min over color channels in local patch.
            2. Estimate atmospheric light A from brightest dark channel pixels.
            3. Estimate transmission map t(x) from dark channel.
            4. Recover scene: J(x) = (I(x) - A) / max(t(x), t_min) + A

        Args:
            frame: Input BGR hazy frame.
            omega: Haze removal strength (0.0-1.0). 0.95 = strong removal.
            t_min: Minimum transmission to avoid division artifacts.

        Returns:
            Dehazed BGR frame.
        """
        img = frame.astype(np.float64) / 255.0
        h, w, c = img.shape

        # Step 1: Dark channel — minimum across RGB in 15x15 local patch
        patch_size = 15
        pad = patch_size // 2
        dark_channel = np.min(img, axis=2)
        # Use erosion as local minimum (equivalent to min filter)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(dark_channel, kernel)

        # Step 2: Atmospheric light — top 0.1% brightest pixels in dark channel
        num_pixels = h * w
        num_brightest = max(int(num_pixels * 0.001), 1)
        dark_flat = dark_channel.ravel()
        indices = np.argsort(dark_flat)[-num_brightest:]

        # Get corresponding pixels from original image
        img_flat = img.reshape(num_pixels, c)
        atmospheric_light = np.mean(img_flat[indices], axis=0)
        atmospheric_light = np.clip(atmospheric_light, 0.1, 1.0)

        # Step 3: Transmission map
        normalized = img / atmospheric_light
        dark_normalized = np.min(normalized, axis=2)
        dark_normalized = cv2.erode(dark_normalized, kernel)
        transmission = 1.0 - omega * dark_normalized
        transmission = np.clip(transmission, t_min, 1.0)

        # Guided filter refinement (approximate with bilateral)
        transmission_refined = cv2.bilateralFilter(
            transmission.astype(np.float32), 60, 0.5, 20
        ).astype(np.float64)

        # Step 4: Scene recovery
        transmission_3ch = np.stack([transmission_refined] * 3, axis=2)
        recovered = (img - atmospheric_light) / transmission_3ch + atmospheric_light
        recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)

        logger.debug(f"Dark channel prior dehazing applied (ω={omega}, t_min={t_min})")
        return recovered

    def auto_enhance(self, frame: np.ndarray, is_night: bool = False) -> np.ndarray:
        """
        Orchestrate the full enhancement pipeline.

        Day pipeline:
            1. CLAHE (mild contrast boost)
            2. Dehaze (if contrast ratio is poor)
            3. Light sharpening kernel

        Night pipeline:
            1. Gamma correction (brighten, gamma=0.5)
            2. CLAHE (aggressive, clip_limit=3.0)
            3. Unsharp mask (recover detail after denoising)

        Args:
            frame: Input BGR frame.
            is_night: Whether frame is night footage.

        Returns:
            Enhanced frame optimized for detection.
        """
        if is_night:
            logger.info("Running night enhancement pipeline")
            # Brighten dark regions aggressively
            result = self.gamma_correction(frame, gamma=0.5)
            # Aggressive local contrast
            result = self.clahe(result, clip_limit=3.0, tile_size=(8, 8))
            # Recover edges blurred by denoising
            result = self.unsharp_mask(result, kernel_size=5, sigma=1.0, strength=1.5)
        else:
            logger.info("Running day enhancement pipeline")
            # Mild contrast improvement
            result = self.clahe(frame, clip_limit=2.0, tile_size=(8, 8))

            # Check if hazy: low contrast ratio indicates haze
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            contrast_ratio = float(gray.std())
            if contrast_ratio < 40:
                logger.info(f"Low contrast detected ({contrast_ratio:.1f}), applying dehaze")
                result = self.dehaze(result)

            # Light sharpening
            result = self.sharpen_kernel(result)

        return result
