"""
Noise Removal Module for Flyover Enforcement System.

Implements multiple denoising strategies optimized for CCTV footage,
with automatic selection based on noise estimation. Critical for
night-mode preprocessing where sensor noise degrades detection quality.

DIP Techniques:
    - Gaussian Blur (isotropic smoothing)
    - Bilateral Filter (edge-preserving denoising)
    - Non-Local Means Denoising (patch-based, best for heavy grain)
    - Median Blur (salt-and-pepper noise removal)
    - Automatic noise estimation via Laplacian variance
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NoiseRemover:
    """
    Adaptive noise removal engine for CCTV frames.
    
    Provides multiple denoising algorithms and an automatic selector
    that estimates noise level using Laplacian variance and picks the
    optimal filter based on conditions (day/night, noise intensity).
    """

    def __init__(self, laplacian_threshold: float = 100.0):
        """
        Initialize NoiseRemover.

        Args:
            laplacian_threshold: Laplacian variance below this value
                indicates heavy noise requiring aggressive denoising.
        """
        self.laplacian_threshold = laplacian_threshold
        logger.info(
            f"NoiseRemover initialized (laplacian_threshold={laplacian_threshold})"
        )

    def gaussian_blur(self, frame: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Apply standard Gaussian smoothing.

        Simple isotropic filter suitable for mild noise. Fast but
        blurs edges — avoid for plate regions.

        Args:
            frame: Input BGR frame.
            ksize: Kernel size (must be odd).

        Returns:
            Smoothed frame.
        """
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        logger.debug(f"Gaussian blur applied (ksize={ksize})")
        return result

    def bilateral_filter(
        self,
        frame: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> np.ndarray:
        """
        Edge-preserving bilateral filter.

        Primary denoising method for night mode — smooths flat regions
        while preserving edges (important for number plate boundaries).

        Args:
            frame: Input BGR frame.
            d: Diameter of pixel neighborhood.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.

        Returns:
            Edge-preserved denoised frame.
        """
        result = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
        logger.debug(
            f"Bilateral filter applied (d={d}, σ_color={sigma_color}, σ_space={sigma_space})"
        )
        return result

    def nlmeans_denoising(self, frame: np.ndarray, h: float = 10.0) -> np.ndarray:
        """
        Non-Local Means denoising for heavy grain.

        Best quality denoising for extremely noisy night footage, but
        computationally expensive (~100ms per frame). Uses patch-based
        matching to find similar regions across the entire image.

        Args:
            frame: Input BGR frame.
            h: Filter strength. Higher h removes more noise but also
               removes detail. 10 is good for moderate noise.

        Returns:
            Denoised frame.
        """
        result = cv2.fastNlMeansDenoisingColored(
            frame, None, h, h, templateWindowSize=7, searchWindowSize=21
        )
        logger.debug(f"Non-local means denoising applied (h={h})")
        return result

    def median_blur(self, frame: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Median blur for salt-and-pepper noise.

        Effective for impulse noise common in cheap CCTV sensors.
        Preserves edges better than Gaussian for this noise type.

        Args:
            frame: Input BGR frame.
            ksize: Aperture size (must be odd and > 1).

        Returns:
            Filtered frame.
        """
        if ksize % 2 == 0:
            ksize += 1
        if ksize < 3:
            ksize = 3
        result = cv2.medianBlur(frame, ksize)
        logger.debug(f"Median blur applied (ksize={ksize})")
        return result

    def estimate_noise(self, frame: np.ndarray) -> float:
        """
        Estimate noise level using Laplacian variance.

        The Laplacian highlights rapid intensity changes. In a noisy
        image, the variance of the Laplacian is high due to noise pixels.
        Conversely, a blurry/clean image has low variance.

        NOTE: This is inverted for noise estimation — low variance means
        the image is already smooth (possibly over-denoised or blurry),
        high variance can mean sharp OR noisy. We use the threshold to
        distinguish.

        Args:
            frame: Input BGR frame.

        Returns:
            Laplacian variance (float). Lower = smoother/noisier after
            denoising, Higher = sharper/noisier before denoising.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        logger.debug(f"Noise estimate (Laplacian variance): {variance:.2f}")
        return variance

    def auto_select(self, frame: np.ndarray, is_night: bool = False) -> np.ndarray:
        """
        Automatically select and apply the best denoising strategy.

        Decision logic:
        1. Estimate noise via Laplacian variance.
        2. If night mode AND heavy noise → NL-Means (strongest).
        3. If night mode AND moderate noise → Bilateral (edge-preserving).
        4. If day AND heavy noise → Median + Gaussian.
        5. If day AND low noise → light Gaussian only.

        Args:
            frame: Input BGR frame.
            is_night: Whether the frame is from night footage.

        Returns:
            Best-filtered frame based on noise estimation.
        """
        noise_level = self.estimate_noise(frame)
        heavy_noise = noise_level < self.laplacian_threshold

        if is_night:
            if heavy_noise:
                # Night + heavy noise → aggressive NL-Means
                logger.info(
                    f"Night + heavy noise (var={noise_level:.1f}): applying NL-Means"
                )
                result = self.nlmeans_denoising(frame, h=12)
            else:
                # Night + moderate noise → bilateral (preserve plate edges)
                logger.info(
                    f"Night + moderate noise (var={noise_level:.1f}): applying bilateral"
                )
                result = self.bilateral_filter(frame, d=9, sigma_color=75, sigma_space=75)
        else:
            if heavy_noise:
                # Day + heavy noise → median then Gaussian
                logger.info(
                    f"Day + heavy noise (var={noise_level:.1f}): applying median + gaussian"
                )
                result = self.median_blur(frame, ksize=3)
                result = self.gaussian_blur(result, ksize=3)
            else:
                # Day + clean → light Gaussian
                logger.info(
                    f"Day + clean (var={noise_level:.1f}): applying light gaussian"
                )
                result = self.gaussian_blur(frame, ksize=3)

        return result
