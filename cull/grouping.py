"""Group similar photos and rank the photos inside each group."""

from __future__ import annotations

import numpy as np

# Two photos belong together when taken within MAX_GAP seconds and their embeddings are at least MIN_SIM alike.
MAX_GAP = 120.0
MIN_SIM = 0.85
BLINK_CLOSED = 0.5


def find_groups(times: list[float], embs: np.ndarray, max_gap: float = MAX_GAP, min_sim: float = MIN_SIM) -> list[list[int]]:
    """Single-linkage clustering over photos sorted by time. Returns groups of 2+ indices."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    parent = list(range(len(times)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1 :]:
            if times[b] - times[a] > max_gap:
                break
            if float(embs[a] @ embs[b]) >= min_sim:
                parent[find(a)] = find(b)

    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def rank_group(metrics: list[dict]) -> list[dict]:
    """Sharpness of each photo relative to the sharpest in its group (0 to 100), with plain-language notes."""
    all_faces = all(m["faces"] for m in metrics)

    def sharp(m: dict) -> float:
        # With faces in every shot, the sharpness of the largest face is what matters.
        if all_faces:
            return max(m["faces"], key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))["sharpness"]
        return m["sharpness"]

    sharps = [sharp(m) for m in metrics]
    best_sharp = max(sharps) or 1.0
    min_blown = min(m["blown"] for m in metrics)

    out = []
    for m, s in zip(metrics, sharps):
        notes = []
        s_rel = s / best_sharp
        score = 100 * s_rel  # the recommendation is the sharpest photo; the notes below are information only

        if s_rel >= 0.999:
            notes.append("sharpest")
        elif s_rel < 0.8:
            notes.append(f"blurrier ({round((1 - s_rel) * 100)}% less detail)")

        closed = sum(1 for f in m["faces"] if f["blink"] > BLINK_CLOSED)
        if closed:
            notes.append(f"eyes closed ({closed} of {len(m['faces'])} faces)" if len(m["faces"]) > 1 else "eyes closed")

        extra_blown = m["blown"] - min_blown
        if extra_blown > 0.02:
            notes.append("more blown highlights")

        out.append(
            {
                "score": round(max(score, 0.0), 1),
                "sharpness_rel": round(s_rel, 3),
                "faces": len(m["faces"]),
                "eyes_closed": closed,
                "notes": notes,
            }
        )
    return out
