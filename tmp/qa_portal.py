"""Masque les portails : tout ce qui se voit DANS une porte n'appartient pas à cette salle.

Le VLM compte sinon les photos / objets du voisin. Ici on peint l'intérieur du
portail en noir (le cadre de la porte reste visible pour le statut open/locked).
"""
from __future__ import annotations

import numpy as np

from agent.vision import _HUD_H
from qa_db import WALL_COLOR_NAMES

_COS_THRESH = 0.99
_SAT_THRESH = 20


def _wall_match(frame: np.ndarray, target_color: str) -> np.ndarray:
    from agent.palette import WALL_PALETTE

    target_rgb = dict(WALL_PALETTE).get(target_color)
    if target_rgb is None:
        return np.zeros(frame[_HUD_H:, :, 0].shape, dtype=bool)
    view = frame[_HUD_H:, :, :].astype(np.float64)
    target = np.array(target_rgb, dtype=np.float64)
    sat = view.max(axis=-1) - view.min(axis=-1)
    colorful = sat > _SAT_THRESH
    pn = np.linalg.norm(view, axis=-1, keepdims=True)
    pn = np.where(pn < 1e-6, 1.0, pn)
    cos = ((view / pn) * (target / np.linalg.norm(target))).sum(axis=-1)
    # cosine seul confond slate (sombre) et lavender (clair) — même teinte.
    # l'éclairage du jeu scale RGB d'un facteur ~0.65, jamais x2.
    scale = pn.squeeze(-1) / (np.linalg.norm(target) + 1e-6)
    return colorful & (cos > _COS_THRESH) & (scale > 0.50) & (scale < 1.18)


def wall_match_fraction(frame: np.ndarray, target_color: str) -> float:
    m = _wall_match(frame, target_color)
    return float(m.mean()) if m.size else 0.0


def best_wall_color(frames: list[np.ndarray]) -> str | None:
    """Couleur palette qui colle le mieux (luminosité + teinte)."""
    scores: dict[str, float] = {c: 0.0 for c in WALL_COLOR_NAMES}
    for fr in frames:
        best_c, best_v = None, 0.0
        for c in WALL_COLOR_NAMES:
            v = wall_match_fraction(fr, c)
            if v > best_v:
                best_c, best_v = c, v
        if best_c and best_v >= 0.12:
            scores[best_c] += best_v
    if not any(scores.values()):
        return None
    return max(scores, key=scores.get)


def _close_small_gaps(is_hole: np.ndarray, max_gap: int = 18) -> np.ndarray:
    hole = is_hole.copy()
    w = hole.size
    i = 0
    while i < w:
        if hole[i]:
            i += 1
            continue
        j = i
        while j < w and not hole[j]:
            j += 1
        if i > 0 and j < w and (j - i) <= max_gap:
            hole[i:j] = True
        i = j
    return hole


def detect_lock_panel(frame: np.ndarray) -> bool:
    """Plaque jaune + cadenas : CV, pas le VLM (qui la prend pour une photo)."""
    from agent.geometry import lock_visible
    return bool(lock_visible(frame, min_warm_frac=0.62, min_dark_frac=0.25))


def mask_portals(
    frame: np.ndarray,
    room_color: str,
    *,
    fill: tuple[int, int, int] = (8, 8, 8),
    floor_keep: float = 0.0,
) -> np.ndarray:
    """Peint en sombre l'intérieur des portails, sol inclus.

    Les ennemis du voisin étaient visibles sur le damier du portail.
    La plaque cadenas est aussi masquée: le VLM la prenait pour une photo.
    """
    if not room_color:
        return frame
    match = _wall_match(frame, room_color)
    h, w = match.shape
    col = match.mean(axis=0)
    k = max(3, w // 80)
    sm = np.convolve(col, np.ones(k) / k, mode="same")
    is_wall = sm > 0.18
    is_hole = _close_small_gaps(sm < 0.14, 18)
    y1 = int(h * (1.0 - floor_keep)) if floor_keep > 0 else h

    out = frame.copy()
    i = 0
    while i < w:
        if not is_hole[i]:
            i += 1
            continue
        j = i
        while j < w and is_hole[j]:
            j += 1
        left_ok = i > 8 and is_wall[max(0, i - 16) : i].mean() > 0.4
        right_ok = j < w - 8 and is_wall[j : min(w, j + 16)].mean() > 0.4
        width = j - i
        centered = i > w * 0.12 and j < w * 0.88
        flanked = (left_ok and right_ok) or (centered and (left_ok or right_ok) and width >= 20)
        if flanked and 8 < width < w * 0.72:
            # petit pad: on masque presque tout le vide (plaque cadenas comprise).
            # le CV a déjà noté le verrou sur la frame non masquée.
            pad = max(1, int(width * 0.02))
            a, b = i + pad, j - pad
            if b > a:
                out[_HUD_H:_HUD_H + y1, a:b, :] = fill
        i = j
    return out


def prepare_survey_frames(
    frames: list[np.ndarray],
    room_color: str | None,
    *,
    mask: bool = True,
) -> list[np.ndarray]:
    if not mask or not room_color:
        return list(frames)
    return [mask_portals(fr, room_color) for fr in frames]
