"""
Violation Logger Module for Flyover Enforcement System.

Provides persistent storage for violation records using SQLite.
Supports CRUD operations, filtering, and status updates.
"""

import os
import sqlite3
import logging
from typing import List, Optional, Dict
from datetime import datetime

from .logic_engine import Violation

logger = logging.getLogger(__name__)


class ViolationLogger:
    """
    SQLite-backed violation logger for persistent storage.

    Table schema:
        violations(
            id TEXT PRIMARY KEY,
            plate TEXT,
            vehicle_class TEXT,
            timestamp TEXT,
            snapshot_path TEXT,
            fine_amount INTEGER,
            location TEXT,
            rule TEXT,
            status TEXT DEFAULT 'pending',
            pdf_path TEXT,
            confidence REAL,
            created_at TEXT
        )
    """

    def __init__(self, db_path: str = "data/violations/violations.db"):
        """
        Initialize ViolationLogger and create table if needed.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._init_db()
        logger.info(f"ViolationLogger initialized (db={db_path})")

    def _init_db(self) -> None:
        """Create the violations table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                plate TEXT NOT NULL,
                vehicle_class TEXT,
                timestamp TEXT,
                snapshot_path TEXT,
                fine_amount INTEGER DEFAULT 500,
                location TEXT,
                rule TEXT,
                status TEXT DEFAULT 'pending',
                pdf_path TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for faster queries by plate and status
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate ON violations(plate)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON violations(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp)
        """)

        conn.commit()
        conn.close()
        logger.debug("Database table initialized")

    def log_violation(self, violation: Violation) -> bool:
        """
        Log a violation to the database.

        Args:
            violation: Violation object to persist.

        Returns:
            True if successfully logged, False on error.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO violations 
                (id, plate, vehicle_class, timestamp, snapshot_path,
                 fine_amount, location, rule, status, pdf_path, 
                 confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    violation.id,
                    violation.plate,
                    violation.vehicle_class,
                    violation.timestamp,
                    violation.snapshot_path,
                    violation.fine_amount,
                    violation.location,
                    violation.rule,
                    violation.status,
                    violation.pdf_path,
                    violation.confidence,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
            logger.info(f"Violation logged: {violation.id} ({violation.plate})")
            return True

        except Exception as e:
            logger.error(f"Failed to log violation: {e}")
            return False

    def get_all_violations(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        """
        Retrieve all violations, ordered by timestamp descending.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of violation dictionaries.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM violations 
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows

        except Exception as e:
            logger.error(f"Failed to retrieve violations: {e}")
            return []

    def get_by_plate(self, plate: str) -> List[Dict]:
        """
        Retrieve all violations for a specific plate.

        Args:
            plate: Number plate text.

        Returns:
            List of violation dictionaries for that plate.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM violations 
                WHERE plate = ?
                ORDER BY timestamp DESC
                """,
                (plate,),
            )

            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows

        except Exception as e:
            logger.error(f"Failed to retrieve violations for {plate}: {e}")
            return []

    def update_status(self, violation_id: str, status: str) -> bool:
        """
        Update violation status.

        Args:
            violation_id: Violation ID to update.
            status: New status ('pending', 'sent', 'manual_review').

        Returns:
            True if updated successfully.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE violations SET status = ? WHERE id = ?",
                (status, violation_id),
            )

            conn.commit()
            updated = cursor.rowcount > 0
            conn.close()

            if updated:
                logger.info(f"Violation {violation_id} status → {status}")
            return updated

        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False

    def update_pdf_path(self, violation_id: str, pdf_path: str) -> bool:
        """
        Update PDF path for a violation.

        Args:
            violation_id: Violation ID.
            pdf_path: Path to generated PDF report.

        Returns:
            True if updated.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE violations SET pdf_path = ? WHERE id = ?",
                (pdf_path, violation_id),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to update PDF path: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get violation statistics.

        Returns:
            Dictionary with counts for today, this week, total,
            and by status.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM violations")
            total = cursor.fetchone()[0]

            # Today's count
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "SELECT COUNT(*) FROM violations WHERE timestamp LIKE ?",
                (f"{today}%",),
            )
            today_count = cursor.fetchone()[0]

            # This week (last 7 days)
            cursor.execute(
                """
                SELECT COUNT(*) FROM violations 
                WHERE datetime(timestamp) >= datetime('now', '-7 days')
                """
            )
            week_count = cursor.fetchone()[0]

            # By status
            cursor.execute(
                """
                SELECT status, COUNT(*) FROM violations 
                GROUP BY status
                """
            )
            status_counts = dict(cursor.fetchall())

            conn.close()

            stats = {
                "total": total,
                "today": today_count,
                "this_week": week_count,
                "pending": status_counts.get("pending", 0),
                "sent": status_counts.get("sent", 0),
                "manual_review": status_counts.get("manual_review", 0),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total": 0, "today": 0, "this_week": 0,
                "pending": 0, "sent": 0, "manual_review": 0,
            }

    def export_csv(self, output_path: str) -> bool:
        """
        Export all violations to a CSV file.

        Args:
            output_path: Path for the CSV output.

        Returns:
            True if exported successfully.
        """
        import csv

        try:
            violations = self.get_all_violations(limit=10000)
            if not violations:
                logger.warning("No violations to export")
                return False

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=violations[0].keys())
                writer.writeheader()
                writer.writerows(violations)

            logger.info(f"Exported {len(violations)} violations to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False
