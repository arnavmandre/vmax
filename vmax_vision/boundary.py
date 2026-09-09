"""Track-limit judgement from detected geometry.

Everything here runs on what the pipeline saw: contact points in pixels, a
ground homography, and a track identity. Nothing reads a per-frame label.

The rule being applied is the one the dataset defines: a frame is an offence
only when *all four* tyre contact points are beyond the outer edge of the white
line, and the margin of that frame is the smallest of the four excesses -- the
corner that is least far out. An event is a sustained run of such frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import track_model as tm
from .calib import apply_h


@dataclass
class FrameJudgement:
    frame_idx: int
    time_s: float
    contacts_world: np.ndarray          # (4,2) raw back-projection
    fitted_contacts: np.ndarray         # (4,2) after rigid car-geometry fit
    position: np.ndarray                # (2,) fitted car centre
    heading_rad: float
    excess: np.ndarray                  # (4,) per-corner, from fitted contacts
    raw_excess: np.ndarray              # (4,) per-corner, no rigid fit
    margin_m: float                     # min excess: the offence margin
    max_excess_m: float
    is_violation: bool
    score: float


@dataclass
class Event:
    start_frame: int
    end_frame_inclusive: int
    frame_count: int
    start_time_s: float
    end_time_exclusive_s: float
    peak_margin_m: float
    peak_frame: int
    mean_score: float
    confidence: float = 0.0
    confidence_terms: dict = field(default_factory=dict)


def back_project(contacts_uv, H_inv):
    return apply_h(H_inv, np.asarray(contacts_uv, float).reshape(-1, 2))


def rigid_fit(contacts_world):
    """Snap four noisy contacts onto the car's known contact rectangle.

    The wheelbase and track width are fixed and surveyed, so the four points are
    not free: fitting the rigid rectangle removes the component of the detector's
    error that violates the car's own geometry, which is most of it.
    """
    centre, heading = tm.pose_from_contacts(contacts_world[None])
    fitted = tm.car_contacts(centre[0], heading[0])
    return fitted, centre[0], float(heading[0])


def smooth_track(positions, headings, window=5, poly=2):
    """Light Savitzky-Golay smoothing of the fitted pose along the clip.

    A car's path is smooth at 24 fps; per-frame detector noise is not. The window
    is deliberately short -- two frames either side -- so that a genuine two-frame
    excursion survives it, which the ``sustained_vs_blip`` case exists to check.
    """
    from scipy.signal import savgol_filter
    positions = np.asarray(positions, float)
    headings = np.unwrap(np.asarray(headings, float))
    if len(positions) < window:
        return positions, headings
    sp = np.column_stack([savgol_filter(positions[:, i], window, poly) for i in (0, 1)])
    sh = savgol_filter(headings, window, poly)
    return sp, sh


def judge_clip(tracked, H_inv, fps=24.0, smooth=True):
    """Per-frame judgements for one track identity over a clip."""
    frames = sorted(tracked)
    raw_contacts, positions, headings, scores = [], [], [], []
    for idx in frames:
        det = tracked[idx]
        world = back_project(det["contacts_uv"], H_inv)
        fitted, centre, heading = rigid_fit(world)
        raw_contacts.append(world)
        positions.append(centre)
        headings.append(heading)
        scores.append(det["score"])
    positions = np.array(positions)
    headings = np.array(headings)
    if smooth and len(frames) >= 5:
        positions, headings = smooth_track(positions, headings)

    out = []
    for i, idx in enumerate(frames):
        fitted = tm.car_contacts(positions[i], headings[i])
        excess = tm.signed_excess(fitted)
        raw_excess = tm.signed_excess(raw_contacts[i])
        out.append(FrameJudgement(
            frame_idx=idx, time_s=idx / fps,
            contacts_world=raw_contacts[i], fitted_contacts=fitted,
            position=positions[i], heading_rad=float(headings[i]),
            excess=excess, raw_excess=raw_excess,
            margin_m=float(excess.min()), max_excess_m=float(excess.max()),
            is_violation=bool((excess > 0).all()), score=float(scores[i])))
    return out


def lateral_noise(judgements):
    """Online estimate of per-frame lateral uncertainty, from the data itself.

    The car's lateral offset is smooth, so the high-frequency part of the
    measured offset is measurement noise. Differencing twice and taking a robust
    scale gives a per-frame sigma without consulting any ground truth.
    """
    if len(judgements) < 5:
        return 0.10
    lat = np.array([tm.track_coordinates(j.position[None])[1][0] for j in judgements])
    second = np.diff(lat, n=2)
    mad = np.median(np.abs(second - np.median(second)))
    sigma = 1.4826 * mad / np.sqrt(6.0)   # var(2nd difference) = 6 sigma^2
    return float(np.clip(sigma, 0.005, 1.0))


def find_events(judgements, min_frames=3, bridge=2, fps=24.0):
    """Sustained runs of offence frames, bridging brief detector dropouts."""
    flags = {j.frame_idx: j for j in judgements}
    idxs = sorted(flags)
    events, run = [], []
    gap = 0
    for i in idxs:
        if flags[i].is_violation:
            run.append(i)
            gap = 0
        elif run:
            gap += 1
            if gap > bridge:
                events.append(run)
                run, gap = [], 0
    if run:
        events.append(run)

    out = []
    for run in events:
        if len(run) < min_frames:
            continue
        members = [flags[i] for i in range(run[0], run[-1] + 1) if i in flags]
        offence = [m for m in members if m.is_violation]
        peak = max(offence, key=lambda m: m.margin_m)
        out.append(Event(
            start_frame=run[0], end_frame_inclusive=run[-1],
            frame_count=len(offence),
            start_time_s=run[0] / fps, end_time_exclusive_s=(run[-1] + 1) / fps,
            peak_margin_m=float(peak.margin_m), peak_frame=int(peak.frame_idx),
            mean_score=float(np.mean([m.score for m in members]))))
    return out


def _phi(z):
    """Standard normal CDF without pulling in scipy for one call."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def score_event(event, judgements, sigma_lateral, calibration_sigma=0.0,
                fps=24.0, min_frames=3):
    """Confidence that the offence is real, and why.

    The dominant term is statistical: how many standard deviations the peak
    margin sits above zero, given the measurement noise the clip itself reveals
    and the stated uncertainty of the calibration. The remaining terms are
    penalties a steward would apply anyway -- a weak detection, a barely
    sustained run, a track that kept dropping out.
    """
    members = [j for j in judgements if event.start_frame <= j.frame_idx <= event.end_frame_inclusive]
    offence = [j for j in members if j.is_violation]
    n_eff = max(min(len(offence), 8), 1)          # correlated frames: cap the gain
    sigma = float(np.hypot(sigma_lateral / np.sqrt(n_eff), calibration_sigma))
    z = event.peak_margin_m / max(sigma, 1e-4)
    geometric = _phi(z)

    duration = float(np.clip(len(offence) / (min_frames * 2.0), 0.35, 1.0))
    detection = float(np.clip((event.mean_score - 0.25) / 0.45, 0.3, 1.0))
    coverage = float(np.clip(len(offence) / max(len(members), 1), 0.5, 1.0))
    confidence = geometric * (0.55 + 0.45 * duration) * (0.6 + 0.4 * detection) * coverage

    event.confidence = float(np.clip(confidence, 0.0, 1.0))
    event.confidence_terms = {
        "geometric": float(geometric),
        "z_score": float(z),
        "sigma_total_m": sigma,
        "sigma_lateral_m": float(sigma_lateral),
        "sigma_calibration_m": float(calibration_sigma),
        "effective_frames": int(n_eff),
        "duration": duration,
        "detection": detection,
        "coverage": coverage,
    }
    return event
