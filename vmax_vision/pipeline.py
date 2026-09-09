"""End-to-end steward pipeline for one clip.

Detect -> track -> back-project -> judge -> attribute. The only inputs are the
video, the circuit map and a ground homography; which homography is used is the
one deployment choice that matters, so it is explicit:

``surveyed``  the camera calibration an installed system is given by survey.
``self``      the homography this pipeline recovered from the video alone.

Ground truth is never read here. :mod:`vmax_vision.evaluate` scores the output
afterwards.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict

import numpy as np

from . import boundary, clips as clipmod, driver_id, tracker as tracking
from .calib import Camera

SELF_CALIBRATION = pathlib.Path("pipeline_out/self_calibration.json")


def load_homography(camera_name, mode="surveyed", cameras=None):
    """The ground homography and the lateral uncertainty that comes with it."""
    cameras = cameras or Camera.load_all()
    if mode == "surveyed":
        return np.array(cameras[camera_name].H, float), 0.0, "surveyed calibration"
    data = json.loads(SELF_CALIBRATION.read_text())
    entry = data[camera_name]
    sigma = float(entry["track_frame_error"]["mean_abs_lateral_error_m"])
    return np.array(entry["homography"], float), sigma, "video-only self-calibration"


def run_clip(clip, detector, mode="surveyed", cameras=None, min_frames=3,
             bridge=2, keep_frames=False, progress=None):
    """Run the whole stack over one clip and return a plain-dict result."""
    cameras = cameras or Camera.load_all()
    if clip.moving_camera:
        raise ValueError("run_clip handles fixed cameras; the moving rig is training data")
    H, sigma_calibration, calibration_label = load_homography(clip.camera, mode, cameras)
    H_inv = np.linalg.inv(H)

    per_frame, frames_bgr = [], {}
    for idx, frame in clip.frames():
        dets = detector(frame)
        per_frame.append(dets)
        if keep_frames:
            frames_bgr[idx] = frame
        else:
            frames_bgr[idx] = frame if idx % 4 == 0 else None
        if progress:
            progress(idx, dets)
    frames_bgr = {k: v for k, v in frames_bgr.items() if v is not None}

    tracked, _tracker = tracking.run(per_frame)
    by_track: dict[int, dict] = {}
    for idx, live in enumerate(tracked):
        for tid, det, _vel in live:
            if det.get("contacts_uv") is None:
                continue
            by_track.setdefault(tid, {})[idx] = det

    judgements, events = {}, {}
    for tid, frames in by_track.items():
        js = boundary.judge_clip(frames, H_inv, fps=clip.fps)
        sigma_lateral = boundary.lateral_noise(js)
        evs = boundary.find_events(js, min_frames=min_frames, bridge=bridge, fps=clip.fps)
        for ev in evs:
            boundary.score_event(ev, js, sigma_lateral, sigma_calibration,
                                 fps=clip.fps, min_frames=min_frames)
        judgements[tid] = js
        events[tid] = evs

    attributions = driver_id.attribute(by_track, frames_bgr, judgements, fps=clip.fps)

    return {
        "clip": clip.key,
        "scenario": clip.scenario,
        "camera": clip.camera,
        "fps": clip.fps,
        "detector": getattr(detector, "name", type(detector).__name__),
        "calibration_mode": mode,
        "calibration_label": calibration_label,
        "calibration_sigma_m": sigma_calibration,
        "frames": len(per_frame),
        "detections_per_frame": [len(d) for d in per_frame],
        "tracks": {
            str(tid): {
                "attribution": asdict(attributions[tid]) if tid in attributions else None,
                "lateral_noise_m": boundary.lateral_noise(judgements[tid]),
                "frames": [
                    {
                        "frame_idx": j.frame_idx,
                        "time_s": j.time_s,
                        "score": j.score,
                        "position_m": j.position.tolist(),
                        "heading_rad": j.heading_rad,
                        "excess_m": j.excess.tolist(),
                        "raw_excess_m": j.raw_excess.tolist(),
                        "margin_m": j.margin_m,
                        "max_excess_m": j.max_excess_m,
                        "is_violation": j.is_violation,
                        "contacts_world": j.fitted_contacts.tolist(),
                    }
                    for j in judgements[tid]
                ],
                "events": [asdict(e) for e in events[tid]],
            }
            for tid in judgements
        },
    }


def run_all(detector, mode="surveyed", scenarios=None, cameras_wanted=None,
            out_dir="pipeline_out/runs", verbose=True):
    """Run every fixed-camera clip and write one JSON per clip."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = Camera.load_all()
    results = []
    for clip in clipmod.discover(include_race_pace=False):
        if scenarios and clip.scenario not in scenarios:
            continue
        if cameras_wanted and clip.camera not in cameras_wanted:
            continue
        result = run_clip(clip, detector, mode=mode, cameras=cameras)
        name = f"{clip.scenario}__{clip.camera}__{mode}.json"
        (out_dir / name).write_text(json.dumps(result))
        results.append(result)
        if verbose:
            n_events = sum(len(t["events"]) for t in result["tracks"].values())
            print(f"  {clip.key:34s} tracks {len(result['tracks'])} events {n_events}", flush=True)
    return results
