"""
Violation Deduplicator for Flyover Enforcement System.

Prevents the same vehicle from being fined multiple times within
a configurable cooldown window. Uses an in-memory cache keyed by
plate number with a timestamp-based cooldown.
"""

import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ViolationDeduplicator:
    """
    Deduplicates violations based on a per-plate cooldown window.

    A vehicle that has already been flagged within the cooldown
    period (default 300 seconds / 5 minutes) will not be logged again.
    """

    def __init__(self, cooldown_seconds: int = 300):
        """
        Initialize Deduplicator.

        Args:
            cooldown_seconds: Minimum time (seconds) between violations
                for the same plate.
        """
        self.cooldown_seconds = cooldown_seconds
        self.cache: Dict[str, float] = {}  # plate -> last_violation_timestamp
        logger.info(f"Deduplicator initialized (cooldown={cooldown_seconds}s)")

    def is_duplicate(self, plate: str) -> bool:
        """
        Check if a violation for this plate is a duplicate.

        A plate is a duplicate if it was last violated less than
        cooldown_seconds ago.

        Args:
            plate: Number plate text.

        Returns:
            True if this is a duplicate (should be skipped).
        """
        if not plate or plate == "UNKNOWN":
            # Unknown plates are always processed (manual review)
            return False

        current_time = time.time()

        if plate in self.cache:
            elapsed = current_time - self.cache[plate]
            if elapsed < self.cooldown_seconds:
                logger.debug(
                    f"Duplicate plate {plate} "
                    f"(elapsed={elapsed:.0f}s < cooldown={self.cooldown_seconds}s)"
                )
                return True

        return False

    def record(self, plate: str) -> None:
        """
        Record a violation timestamp for a plate.

        Call this AFTER confirming the violation is not a duplicate.

        Args:
            plate: Number plate text.
        """
        if plate and plate != "UNKNOWN":
            self.cache[plate] = time.time()
            logger.debug(f"Plate {plate} recorded in deduplication cache")

    def check_and_record(self, plate: str) -> bool:
        """
        Combined check-and-record in one call.

        Returns True if the violation is NEW (not duplicate).
        Automatically records it if new.

        Args:
            plate: Number plate text.

        Returns:
            True if this is a NEW violation (proceed with logging).
            False if duplicate (skip).
        """
        if self.is_duplicate(plate):
            return False
        self.record(plate)
        return True

    def clear_expired(self) -> int:
        """
        Clear expired entries from the cache.

        Removes plates whose cooldown has already passed.
        Call periodically to prevent memory growth in long runs.

        Returns:
            Number of entries cleared.
        """
        current_time = time.time()
        expired = [
            plate
            for plate, ts in self.cache.items()
            if current_time - ts >= self.cooldown_seconds
        ]
        for plate in expired:
            del self.cache[plate]

        if expired:
            logger.info(f"Cleared {len(expired)} expired deduplication entries")
        return len(expired)

    def get_cache_size(self) -> int:
        """Return current number of plates in the cache."""
        return len(self.cache)

    def reset(self) -> None:
        """Clear all entries from the cache."""
        self.cache.clear()
        logger.info("Deduplication cache cleared")
