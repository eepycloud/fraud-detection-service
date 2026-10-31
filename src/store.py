"""Bounded SQLite journal for engineered features and model outputs."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS scores (
                id TEXT PRIMARY KEY, event_time TEXT NOT NULL, amount REAL NOT NULL,
                hour INTEGER NOT NULL, category TEXT NOT NULL, probability REAL NOT NULL,
                prediction INTEGER NOT NULL, label INTEGER, source TEXT NOT NULL,
                latency_ms REAL, features TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("CREATE INDEX IF NOT EXISTS scores_source ON scores(source, event_time)")
            db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value INTEGER)")
    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()
    def add(self, rows):
        with self.connection() as db:
            db.executemany("""INSERT OR IGNORE INTO scores
                (id,event_time,amount,hour,category,probability,prediction,label,source,latency_ms,features)
                VALUES (:id,:event_time,:amount,:hour,:category,:probability,:prediction,:label,:source,:latency_ms,:features)""", rows)
            for source in {r["source"] for r in rows}:
                db.execute("""DELETE FROM scores WHERE source=? AND id NOT IN
                    (SELECT id FROM scores WHERE source=? ORDER BY event_time DESC, id DESC LIMIT 5000)""", (source,source))
    def recent(self, source, limit=5000):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM scores WHERE source=? ORDER BY event_time DESC,id DESC LIMIT ?", (source,limit)).fetchall()
        result = [dict(r) for r in reversed(rows)]
        for row in result:
            row["features"] = json.loads(row["features"])
        return result
    def reserve_demo(self, size, batch):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM state WHERE key='demo_cursor'").fetchone()
            start = row[0] if row else 0
            end = min(size, start + batch)
            db.execute("INSERT OR REPLACE INTO state VALUES ('demo_cursor',?)", (end,))
        return start,end
