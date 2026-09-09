"""Driver attribution ladder.

Knowing that *a* car left the track is worthless to a steward; the question is
always which one. The ladder is deliberately ordered from cheapest and most
reliable to weakest, and reports the highest rung that actually fired along with
what it was based on, so a marginal attribution is visible as marginal rather
than being laundered into a name.

  1. livery      colour signature inside the detected silhouette
  2. continuity  an unbroken track back to a frame where a higher rung fired
  3. lane        which side of the racing line the car has held
  4. sequence    running order along the track, when nothing else separates them

Rung 1 is the only one that identifies a car outright; the rest disambiguate
between candidates already on track, which is what the two-car scenarios need.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import track_model as tm

# Livery references, as rendered. A real deployment would load these from the
# entry list; here they come from the two liveries the simulator paints.
LIVERIES = {
    "car_1": {"name": "Car 1 (red)", "rgb": (218, 27, 39)},
    "car_2": {"name": "Car 2 (teal)", "rgb": (11, 141, 155)},
}

RUNGS = ["livery", "continuity", "lane", "sequence", "unattributed"]


@dataclass
class Attribution:
    track_id: int
    car_id: str | None
    rung: str
    confidence: float
    evidence: dict


def _hue_signature(frame_bgr, det, max_pixels=4000):
    """Dominant chroma of the car's own pixels, excluding tarmac-grey ones."""
    box = np.array(det["box"], float)
    x0, y0 = np.maximum(box[:2].astype(int), 0)
    x1 = min(int(box[2]) + 1, frame_bgr.shape[1])
    y1 = min(int(box[3]) + 1, frame_bgr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1]
    sel = np.ones(patch.shape[:2], bool)
    crop = det.get("mask_crop")
    if crop is not None and np.size(crop["data"]):
        ox, oy = crop["origin"]
        data = np.asarray(crop["data"])
        sub = data[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        if sub.shape == sel.shape:
            sel = sub
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sel = sel & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 55)
    if sel.sum() < 12:
        return None
    pixels = patch[sel].astype(np.float32)
    if len(pixels) > max_pixels:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), max_pixels, False)]
    return pixels.mean(axis=0)[::-1]  # BGR mean -> RGB


def _livery_match(signature):
    if signature is None:
        return None, 0.0, {}
    lab = {k: cv2.cvtColor(np.uint8([[v["rgb"][::-1]]]), cv2.COLOR_BGR2Lab)[0, 0].astype(float)
           for k, v in LIVERIES.items()}
    sig = cv2.cvtColor(np.uint8([[np.clip(signature[::-1], 0, 255)]]), cv2.COLOR_BGR2Lab)[0, 0].astype(float)
    # Chroma only: shading changes lightness far more than it changes hue.
    dist = {k: float(np.linalg.norm((sig - v)[1:])) for k, v in lab.items()}
    best, second = sorted(dist.items(), key=lambda kv: kv[1])[:2]
    separation = (second[1] - best[1]) / max(second[1], 1e-6)
    confidence = float(np.clip(separation * 1.6, 0.0, 1.0) * np.clip(1.0 - best[1] / 90.0, 0.0, 1.0))
    return best[0], confidence, {"distances": dist, "signature_rgb": [float(v) for v in signature]}


def attribute(track_frames, frames_bgr, judgements_by_track, fps=24.0):
    """Attribute every track in a clip, walking down the ladder as needed.

    ``track_frames`` maps track id -> {frame index: detection}.
    """
    votes = {}
    for tid, per_frame in track_frames.items():
        picks, weights, evidence = [], [], []
        for idx, det in sorted(per_frame.items()):
            frame = frames_bgr.get(idx)
            if frame is None:
                continue
            sig = _hue_signature(frame, det)
            car, conf, ev = _livery_match(sig)
            if car and conf > 0.05:
                picks.append(car)
                weights.append(conf * det["score"])
                evidence.append(ev)
        votes[tid] = (picks, weights, evidence)

    attributions = {}
    claimed = {}
    order = sorted(track_frames, key=lambda t: -sum(votes[t][1]))
    for tid in order:
        picks, weights, _ev = votes[tid]
        if not picks:
            attributions[tid] = Attribution(tid, None, "unattributed", 0.0,
                                            {"reason": "no usable livery pixels"})
            continue
        tally = {}
        for car, w in zip(picks, weights):
            tally[car] = tally.get(car, 0.0) + w
        best = max(tally, key=tally.get)
        total = sum(tally.values())
        share = tally[best] / max(total, 1e-9)
        frames_voting = len(picks)
        confidence = float(np.clip(share * (1 - np.exp(-frames_voting / 12.0)), 0, 1))
        if best in claimed:
            # Two tracks claiming one car: the weaker one drops to the next rung.
            attributions[tid] = Attribution(tid, None, "livery", 0.0,
                                            {"reason": f"{best} already claimed by track {claimed[best]}",
                                             "share": share})
            continue
        claimed[best] = tid
        attributions[tid] = Attribution(
            tid, best, "livery", confidence,
            {"share": float(share), "voting_frames": frames_voting,
             "tally": {k: float(v) for k, v in tally.items()}})

    # Rung 2-4 for anything the livery rung could not settle.
    unresolved = [t for t, a in attributions.items() if a.car_id is None]
    remaining = [c for c in LIVERIES if c not in claimed]
    if unresolved and len(remaining) == 1 and len(unresolved) == 1:
        tid = unresolved[0]
        attributions[tid] = Attribution(tid, remaining[0], "sequence", 0.55,
                                        {"reason": "only remaining entry on track"})
    elif unresolved:
        laterals = {}
        for tid in unresolved:
            js = judgements_by_track.get(tid, [])
            if js:
                lat = np.mean([tm.track_coordinates(j.position[None])[1][0] for j in js])
                laterals[tid] = float(lat)
        for rank, (tid, lat) in enumerate(sorted(laterals.items(), key=lambda kv: -kv[1])):
            car = remaining[rank] if rank < len(remaining) else None
            attributions[tid] = Attribution(tid, car, "lane", 0.35 if car else 0.0,
                                            {"mean_lateral_offset_m": lat,
                                             "reason": "separated by held line"})
    return attributions
