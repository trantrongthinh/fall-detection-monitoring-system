from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def migrate_database(db_path: Path) -> list[str]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[str] = []

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fall_events (
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

        columns = {row[1] for row in conn.execute("PRAGMA table_info(fall_events)")}
        migrations = {
            "track_id": "ALTER TABLE fall_events ADD COLUMN track_id INTEGER",
            "status": "ALTER TABLE fall_events ADD COLUMN status TEXT NOT NULL DEFAULT 'new'",
            "review_note": "ALTER TABLE fall_events ADD COLUMN review_note TEXT",
            "reviewed_at": "ALTER TABLE fall_events ADD COLUMN reviewed_at TEXT",
            "telegram_sent": "ALTER TABLE fall_events ADD COLUMN telegram_sent INTEGER NOT NULL DEFAULT 0",
            "telegram_response": "ALTER TABLE fall_events ADD COLUMN telegram_response TEXT",
        }

        for column, sql in migrations.items():
            if column not in columns:
                conn.execute(sql)
                applied.append(column)

        conn.commit()

    return applied


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate the fall event SQLite database schema.")
    parser.add_argument("--db", default=str(PROJECT_DIR / "fall_events.db"), help="Path to fall_events.db")
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_DIR / db_path

    applied = migrate_database(db_path)
    if applied:
        print(f"[OK] Migrated {db_path}: {', '.join(applied)}")
    else:
        print(f"[OK] Database already up to date: {db_path}")


if __name__ == "__main__":
    main()
