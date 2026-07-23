from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def require_file(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")


def test_runtime_files() -> None:
    required_files = [
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / ".env.example",
        PROJECT_DIR / "telegram_config.example.json",
        PROJECT_DIR / "models_lstm" / "lstm_runtime_config.example.json",
        PROJECT_DIR / "scripts" / "migrate_database.py",
    ]
    for path in required_files:
        require_file(path)

    with open(PROJECT_DIR / "models_lstm" / "lstm_runtime_config.example.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    if int(config["sequence_len"]) <= 0:
        raise AssertionError("sequence_len must be positive")
    if not 0 <= float(config["threshold"]) <= 1:
        raise AssertionError("threshold must be between 0 and 1")


def test_sequence_comparison() -> None:
    path = PROJECT_DIR / "models_4_clean" / "sequence_comparison_30_35_40.csv"
    require_file(path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise AssertionError("sequence comparison CSV is empty")

    required_columns = {"seq_len", "test_accuracy_05", "test_precision_05", "test_recall_05", "test_f1_05"}
    missing = required_columns.difference(rows[0])
    if missing:
        raise AssertionError(f"Missing columns in sequence comparison CSV: {sorted(missing)}")


def test_feature_shape() -> None:
    from fall_features import FEATURE_DIM

    sample = np.zeros((FEATURE_DIM,), dtype=np.float32)
    if sample.shape != (52,):
        raise AssertionError(f"Expected 52-D feature vector, got {sample.shape}")


def test_event_schema() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fall_events.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE fall_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fall_counter INTEGER NOT NULL,
                    track_id INTEGER,
                    image_path TEXT,
                    status TEXT NOT NULL DEFAULT 'new',
                    review_note TEXT,
                    reviewed_at TEXT,
                    telegram_sent INTEGER NOT NULL DEFAULT 0,
                    telegram_response TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO fall_events (
                    created_at, camera_name, label, confidence,
                    fall_counter, track_id, image_path, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("2026-07-12T10:00:00", "smoke-test", "fall", 0.9, 3, 1, "captures/test.jpg", "new"),
            )
            conn.commit()
            conn.execute(
                """
                UPDATE fall_events
                SET status = ?, review_note = ?, reviewed_at = ?
                WHERE id = 1
                """,
                ("confirmed", "Smoke-test review", "2026-07-12T10:01:00"),
            )
            conn.commit()
            row = conn.execute("SELECT status, review_note, reviewed_at, telegram_sent FROM fall_events").fetchone()
        finally:
            conn.close()

        if row != ("confirmed", "Smoke-test review", "2026-07-12T10:01:00", 0):
            raise AssertionError(f"Unexpected event review state: {row}")


def main() -> None:
    tests = [
        test_runtime_files,
        test_sequence_comparison,
        test_feature_shape,
        test_event_schema,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")

    print("[OK] Smoke test completed.")


if __name__ == "__main__":
    main()
