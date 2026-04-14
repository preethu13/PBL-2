"""
OCR Engine Module for Flyover Enforcement System.

Uses EasyOCR for number plate text recognition with Kerala-specific
plate format validation and common OCR error correction.

Kerala Plate Format: KL-XX-XX-XXXX
    - KL: State code (Kerala)
    - XX: District code (01-99)
    - XX: Series code (1-2 letters)
    - XXXX: Number (1-4 digits)

Regex: r'^KL\\d{2}[A-Z]{1,2}\\d{4}$'

NOTE: PaddlePaddle has no Python 3.14 wheel; EasyOCR provides
      equivalent accuracy for plate recognition on modern Python.
"""

import re
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. Install with: pip install easyocr")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class OCREngine:
    """
    OCR engine for reading Indian number plates.

    Uses EasyOCR with Kerala-specific plate validation,
    OCR error correction, and confidence filtering.
    """

    def __init__(self, min_confidence: float = 0.7, use_gpu: bool = False):
        """
        Initialize OCR Engine.

        Args:
            min_confidence: Minimum OCR confidence to accept a reading.
            use_gpu: Whether to use GPU for OCR inference.
        """
        self.min_confidence = min_confidence
        self.ocr_model = None

        # Kerala plate regex: KL + 2 digits + 1-2 letters + 4 digits
        self.kerala_pattern = re.compile(r'^KL\d{2}[A-Z]{1,2}\d{4}$')

        # Extended pattern for partial matches
        self.partial_pattern = re.compile(r'KL\d{2}[A-Z]{1,2}\d{1,4}')

        # Common OCR character confusion map
        self.ocr_corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'i': '1', 'l': '1', '|': '1',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'B': '8',
            'G': '6',
            'T': '7',
            'A': '4',
            'D': '0',
            'Q': '0',
        }

        if EASYOCR_AVAILABLE:
            try:
                # EasyOCR reader - English language for plates
                # verbose=False suppresses download progress spam
                self.ocr_model = easyocr.Reader(
                    ['en'],
                    gpu=use_gpu,
                    verbose=False,
                )
                logger.info(f"EasyOCR initialized (gpu={use_gpu})")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
        else:
            logger.warning("EasyOCR not available — install with: pip install easyocr")

    def read_plate(self, plate_img) -> Optional[str]:
        """
        Run OCR on a preprocessed plate image.

        Args:
            plate_img: Preprocessed plate image (ideally binarized).

        Returns:
            Recognized plate text (cleaned), or None if unreadable.
        """
        if plate_img is None:
            return None

        if NUMPY_AVAILABLE:
            import numpy as np_local
            if isinstance(plate_img, np_local.ndarray) and plate_img.size == 0:
                return None

        if self.ocr_model is not None:
            return self._read_easyocr(plate_img)
        else:
            return self._read_fallback(plate_img)

    def _read_easyocr(self, plate_img) -> Optional[str]:
        """
        Read plate text using EasyOCR.

        Args:
            plate_img: Plate image (numpy array, BGR or grayscale).

        Returns:
            Cleaned plate text or None.
        """
        try:
            # EasyOCR accepts numpy arrays directly
            # allowlist restricts to alphanumeric characters only
            results = self.ocr_model.readtext(
                plate_img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,         # Return with bounding box and confidence
                paragraph=False,  # Treat each word separately
            )

            if not results:
                logger.debug("EasyOCR returned no results")
                return None

            # results format: [(bbox, text, confidence), ...]
            texts = [(text, conf) for (_, text, conf) in results]

            # Filter by confidence
            filtered = self.confidence_filter(texts, self.min_confidence)

            if not filtered:
                # If nothing passes threshold, take best result anyway
                texts_sorted = sorted(texts, key=lambda x: x[1], reverse=True)
                logger.debug(
                    f"All results below confidence threshold, using best: "
                    f"{texts_sorted[0] if texts_sorted else 'none'}"
                )
                if texts_sorted and texts_sorted[0][1] > 0.3:
                    filtered = [texts_sorted[0]]
                else:
                    return None

            # Concatenate all text fragments
            raw_text = ''.join([t[0] for t in filtered])
            cleaned = self.clean_text(raw_text)

            logger.info(f"EasyOCR raw='{raw_text}' → cleaned='{cleaned}'")
            return cleaned if cleaned else None

        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return None

    def _read_fallback(self, plate_img) -> Optional[str]:
        """
        Fallback when EasyOCR is unavailable.

        Args:
            plate_img: Plate image.

        Returns:
            None (no OCR capability without engine).
        """
        logger.warning("No OCR engine available, returning None")
        return None

    def clean_text(self, text: str) -> str:
        """
        Clean raw OCR text.

        Operations:
        1. Remove spaces, dashes, special characters
        2. Convert to uppercase
        3. Keep only alphanumeric characters

        Args:
            text: Raw OCR output string.

        Returns:
            Cleaned alphanumeric uppercase string.
        """
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
        cleaned = cleaned.upper()
        return cleaned

    def validate_kerala_plate(self, text: str) -> bool:
        """
        Validate if text matches Kerala number plate format.

        Format: KL + 2 digits + 1-2 letters + 4 digits
        Examples: KL07BJ4545, KL01A1234

        Args:
            text: Cleaned plate text.

        Returns:
            True if text matches Kerala plate pattern.
        """
        return bool(self.kerala_pattern.match(text))

    def fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR character misrecognitions.

        Context-aware corrections based on expected position:
        - Chars 3-4 (district code): should be digits
        - Middle chars (series): should be letters
        - Last 4 chars (number): should be digits

        Args:
            text: Cleaned OCR text.

        Returns:
            Corrected text.
        """
        if len(text) < 6:
            return text

        corrected = list(text)

        # Ensure first two chars are 'KL'
        if len(corrected) >= 1 and corrected[0] not in ('K', 'k'):
            pass  # Can't reliably fix first char
        if len(corrected) >= 2 and corrected[1] in ('1', '|', 'l'):
            corrected[1] = 'L'

        # Chars at positions 2-3 should be digits (district code)
        for i in range(2, min(4, len(corrected))):
            char = corrected[i]
            if not char.isdigit() and char in self.ocr_corrections:
                corrected[i] = self.ocr_corrections[char]

        # Last 4 characters should be digits
        if len(corrected) >= 8:
            for i in range(len(corrected) - 4, len(corrected)):
                char = corrected[i]
                if not char.isdigit() and char in self.ocr_corrections:
                    corrected[i] = self.ocr_corrections[char]

            # Middle characters (position 4 to len-4) should be letters
            for i in range(4, len(corrected) - 4):
                char = corrected[i]
                if char.isdigit():
                    reverse_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
                    if char in reverse_map:
                        corrected[i] = reverse_map[char]

        result = ''.join(corrected)
        if result != text:
            logger.info(f"OCR error correction: '{text}' → '{result}'")
        return result

    def confidence_filter(
        self, results: List[Tuple[str, float]], min_conf: float = None
    ) -> List[Tuple[str, float]]:
        """
        Filter OCR results by confidence threshold.

        Args:
            results: List of (text, confidence) tuples.
            min_conf: Minimum confidence. Uses instance default if None.

        Returns:
            Filtered list above threshold.
        """
        threshold = min_conf if min_conf is not None else self.min_confidence
        filtered = [(text, conf) for text, conf in results if conf >= threshold]
        logger.debug(
            f"Confidence filter: {len(filtered)}/{len(results)} passed "
            f"(threshold={threshold})"
        )
        return filtered

    def process(self, plate_img) -> Tuple[Optional[str], bool, float]:
        """
        Full OCR processing pipeline.

        1. Read raw text via EasyOCR
        2. Clean text (remove non-alphanumeric)
        3. Fix common OCR errors
        4. Validate Kerala plate format

        Args:
            plate_img: Preprocessed plate image.

        Returns:
            Tuple of (plate_text, is_valid, confidence).
        """
        raw_text = self.read_plate(plate_img)

        if raw_text is None:
            return None, False, 0.0

        # Apply error corrections
        corrected = self.fix_ocr_errors(raw_text)

        # Validate full pattern
        is_valid = self.validate_kerala_plate(corrected)

        # Try partial match if full validation fails
        if not is_valid:
            match = self.partial_pattern.search(corrected)
            if match:
                corrected = match.group()
                is_valid = self.validate_kerala_plate(corrected)

        logger.info(f"OCR result: '{corrected}', valid={is_valid}")
        return corrected, is_valid, 0.8 if is_valid else 0.5
