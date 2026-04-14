"""
Plate DIP (Digital Image Processing) Module for Flyover Enforcement System.

CRITICAL DIP-heavy module that prepares cropped plate images for OCR.
Implements a comprehensive preprocessing pipeline including resizing,
grayscale conversion, denoising, deskewing, binarization, morphological
cleanup, and padding.

DIP Techniques:
    - Aspect-ratio-preserving resize
    - Bilateral filtering for denoising
    - Hough Line Transform for skew detection
    - Affine rotation for deskewing
    - Otsu's thresholding + adaptive thresholding fallback
    - Morphological operations (erosion, dilation)
    - Border padding for OCR
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PlateDIP:
    """
    Digital Image Processing pipeline for number plate preparation.

    Takes a raw cropped plate image and applies a sequence of DIP
    operations to maximize OCR accuracy. Each step can be visualized
    individually for project demonstration.
    """

    def __init__(self, target_height: int = 64, pad_size: int = 10):
        """
        Initialize PlateDIP.

        Args:
            target_height: Target height for plate resize (maintains aspect ratio).
            pad_size: White border padding around plate for OCR.
        """
        self.target_height = target_height
        self.pad_size = pad_size
        logger.info(
            f"PlateDIP initialized (target_h={target_height}, pad={pad_size})"
        )

    def resize(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Resize plate to target height while maintaining aspect ratio.

        Consistent sizing improves OCR model performance.

        Args:
            plate_img: Input plate image.

        Returns:
            Resized plate image.
        """
        h, w = plate_img.shape[:2]
        if h == 0 or w == 0:
            return plate_img

        scale = self.target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(
            plate_img, (new_w, self.target_height), interpolation=cv2.INTER_LINEAR
        )
        logger.debug(f"Plate resized: ({w}×{h}) → ({new_w}×{self.target_height})")
        return resized

    def to_grayscale(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale if needed.

        Args:
            plate_img: Input plate image (BGR or gray).

        Returns:
            Grayscale image.
        """
        if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted to grayscale")
            return gray
        return plate_img

    def denoise(self, plate_img: np.ndarray, d: int = 5) -> np.ndarray:
        """
        Bilateral filter denoising for edge-preserved smoothing.

        Removes sensor noise while keeping character edges sharp.

        Args:
            plate_img: Grayscale plate image.
            d: Filter diameter.

        Returns:
            Denoised plate image.
        """
        if len(plate_img.shape) == 3:
            denoised = cv2.bilateralFilter(plate_img, d, 75, 75)
        else:
            denoised = cv2.bilateralFilter(plate_img, d, 75, 75)
        logger.debug(f"Bilateral denoise applied (d={d})")
        return denoised

    def deskew(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Correct rotational skew of the plate using Hough Line Transform.

        Pipeline:
        1. Detect edges with Canny
        2. Find lines with HoughLinesP
        3. Compute dominant angle from line orientations
        4. Rotate image by negative of that angle (affine transform)

        Args:
            plate_img: Grayscale plate image.

        Returns:
            Deskewed plate image.
        """
        h, w = plate_img.shape[:2]

        # Edge detection for line finding
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=w * 0.3,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for deskew, returning original")
            return plate_img

        # Compute angles of all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (plate text is horizontal)
            if abs(angle) < 30:
                angles.append(angle)

        if not angles:
            logger.debug("No suitable angles found for deskew")
            return plate_img

        # Dominant angle = median of detected angles
        dominant_angle = np.median(angles)

        if abs(dominant_angle) < 0.5:
            logger.debug(f"Skew angle {dominant_angle:.2f}° negligible, skipping")
            return plate_img

        # Rotate to correct skew
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
        deskewed = cv2.warpAffine(
            plate_img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        logger.info(f"Plate deskewed by {dominant_angle:.2f}°")
        return deskewed

    def binarize(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Binarize plate image using Otsu's method with adaptive fallback.

        Pipeline:
        1. Try Otsu's threshold (global optimal threshold).
        2. If result has poor contrast (too few black or white pixels),
           fall back to adaptive Gaussian thresholding.

        Args:
            plate_img: Grayscale plate image.

        Returns:
            Binary image (0 and 255 only).
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        # Try Otsu's thresholding
        otsu_thresh, binary_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Evaluate quality: check if the result is reasonable
        white_ratio = np.count_nonzero(binary_otsu) / binary_otsu.size
        good_range = 0.2 < white_ratio < 0.85

        if good_range:
            logger.debug(
                f"Otsu binarization (threshold={otsu_thresh:.0f}, "
                f"white_ratio={white_ratio:.2f})"
            )
            return binary_otsu
        else:
            # Fallback: Adaptive Gaussian threshold
            binary_adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )
            logger.info(
                f"Otsu failed (white_ratio={white_ratio:.2f}), "
                f"using adaptive threshold"
            )
            return binary_adaptive

    def morphological_clean(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Clean binary plate image with morphological operations.

        1. Erosion: removes tiny noise dots between characters.
        2. Dilation: restores character thickness after erosion.

        Args:
            plate_img: Binary plate image.

        Returns:
            Cleaned binary plate image.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Erosion: remove noise
        cleaned = cv2.erode(plate_img, kernel, iterations=1)

        # Dilation: recover character body
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        logger.debug("Morphological cleanup applied to plate")
        return cleaned

    def pad_image(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Add white border padding around the plate for OCR compatibility.

        Many OCR models perform better when characters don't touch the
        image border. Pads to approximately 128x32 target size.

        Args:
            plate_img: Processed plate image.

        Returns:
            Padded plate image.
        """
        padded = cv2.copyMakeBorder(
            plate_img,
            self.pad_size,
            self.pad_size,
            self.pad_size,
            self.pad_size,
            cv2.BORDER_CONSTANT,
            value=255 if len(plate_img.shape) == 2 else (255, 255, 255),
        )
        logger.debug(f"Padding added ({self.pad_size}px)")
        return padded

    def preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Full DIP preprocessing pipeline for a plate crop.

        Pipeline:
        1. Resize to standard height (64px)
        2. Convert to grayscale
        3. Bilateral filter denoising
        4. Deskew (Hough line-based rotation correction)
        5. Binarize (Otsu with adaptive fallback)
        6. Morphological cleanup
        7. Pad with white border

        Args:
            plate_img: Raw cropped plate image (BGR).

        Returns:
            Cleaned, binarized, padded plate image ready for OCR.
        """
        if plate_img is None or plate_img.size == 0:
            logger.warning("Empty plate image received")
            return plate_img

        logger.info("Starting plate DIP preprocessing pipeline")

        # Step 1: Resize
        result = self.resize(plate_img)

        # Step 2: Grayscale
        result = self.to_grayscale(result)

        # Step 3: Denoise
        result = self.denoise(result, d=5)

        # Step 4: Deskew
        result = self.deskew(result)

        # Step 5: Binarize
        result = self.binarize(result)

        # Step 6: Morphological cleanup
        result = self.morphological_clean(result)

        # Step 7: Pad
        result = self.pad_image(result)

        logger.info(f"Plate preprocessing complete: {result.shape}")
        return result

    def visualize_pipeline(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Visualize each step of the DIP pipeline for demonstration.

        Creates a grid showing the transformation at each stage.
        Essential for DIP project presentation.

        Args:
            plate_img: Raw cropped plate image (BGR).

        Returns:
            Grid image showing all 7 DIP steps.
        """
        if plate_img is None or plate_img.size == 0:
            return np.zeros((200, 800, 3), dtype=np.uint8)

        display_w = 200
        font = cv2.FONT_HERSHEY_SIMPLEX
        steps = []

        def to_display(img, label):
            """Convert image to displayable BGR with label."""
            if len(img.shape) == 2:
                disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                disp = img.copy()
            h, w = disp.shape[:2]
            scale = display_w / max(w, 1)
            new_h = max(int(h * scale), 1)
            disp = cv2.resize(disp, (display_w, new_h))
            cv2.putText(disp, label, (2, 12), font, 0.35, (0, 255, 0), 1)
            return disp

        # Step 0: Original
        steps.append(to_display(plate_img, "0:Original"))

        # Step 1: Resize
        s1 = self.resize(plate_img)
        steps.append(to_display(s1, "1:Resize"))

        # Step 2: Grayscale
        s2 = self.to_grayscale(s1)
        steps.append(to_display(s2, "2:Grayscale"))

        # Step 3: Denoise
        s3 = self.denoise(s2, d=5)
        steps.append(to_display(s3, "3:Denoise"))

        # Step 4: Deskew
        s4 = self.deskew(s3)
        steps.append(to_display(s4, "4:Deskew"))

        # Step 5: Binarize
        s5 = self.binarize(s4)
        steps.append(to_display(s5, "5:Binarize"))

        # Step 6: Morphological cleanup
        s6 = self.morphological_clean(s5)
        steps.append(to_display(s6, "6:MorphClean"))

        # Step 7: Padding
        s7 = self.pad_image(s6)
        steps.append(to_display(s7, "7:Padded"))

        # Make all same height
        max_h = max(s.shape[0] for s in steps)
        padded_steps = []
        for s in steps:
            h_diff = max_h - s.shape[0]
            if h_diff > 0:
                pad = np.ones((h_diff, s.shape[1], 3), dtype=np.uint8) * 255
                s = np.vstack([s, pad])
            padded_steps.append(s)

        # Arrange in 2 rows of 4
        row1 = np.hstack(padded_steps[:4])
        row2 = np.hstack(padded_steps[4:])

        # Make rows same width
        w1 = row1.shape[1]
        w2 = row2.shape[1]
        if w1 > w2:
            pad = np.ones((row2.shape[0], w1 - w2, 3), dtype=np.uint8) * 255
            row2 = np.hstack([row2, pad])
        elif w2 > w1:
            pad = np.ones((row1.shape[0], w2 - w1, 3), dtype=np.uint8) * 255
            row1 = np.hstack([row1, pad])

        grid = np.vstack([row1, row2])
        logger.debug("Pipeline visualization grid created")
        return grid
