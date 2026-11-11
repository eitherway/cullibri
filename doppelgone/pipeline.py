"""Scan a date range: analyze each photo once (cached in SQLite), then group and rank."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import analyze
from .grouping import find_groups, rank_group

DB = Path(__file__).resolve().parent.parent / "data" / "cache.sqlite"

_analyzer: analyze.Analyzer | None = None
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS analysis (key TEXT PRIMARY KEY, version INT, metrics TEXT, emb BLOB)")
    return con


def scan(source, after: datetime, before: datetime, choice: str, progress: Callable[[int, int], None] = lambda done, total: None) -> list[dict]:
    """Returns groups, each a list of photos sorted by time, with the recommended one flagged as 'pick'."""
    global _analyzer
    assets = source.list(after, before, choice)
    con = _db()
    metrics, embs = [], []
    with _lock:
        for n, a in enumerate(assets):
            progress(n, len(assets))
            row = con.execute("SELECT metrics, emb FROM analysis WHERE key=? AND version=?", (a.key, analyze.VERSION)).fetchone()
            if row:
                m, e = json.loads(row[0]), np.frombuffer(row[1], dtype=np.float32)
            else:
                if _analyzer is None:
                    _analyzer = analyze.Analyzer()
                m, e = _analyzer.analyze(source.image(a.id, "full")[0])
                con.execute("INSERT OR REPLACE INTO analysis VALUES (?,?,?,?)", (a.key, analyze.VERSION, json.dumps(m), e.tobytes()))
                con.commit()
            metrics.append(m)
            embs.append(e)
    progress(len(assets), len(assets))
    if not assets:
        return []

    groups = []
    for idx in find_groups([a.taken for a in assets], np.stack(embs)):
        ranked = rank_group([metrics[i] for i in idx])
        best = max(range(len(idx)), key=lambda k: ranked[k]["score"])
        groups.append(
            [
                {"id": assets[i].id, "name": assets[i].name, "taken": assets[i].taken, "pick": k == best, **ranked[k]}
                for k, i in enumerate(idx)
            ]
        )
    groups.sort(key=lambda g: g[0]["taken"])
    return groups
