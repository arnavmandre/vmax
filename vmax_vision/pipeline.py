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

import hashlib
import json
import pathlib
import pickle
from dataclasses import asdict

import cv2
import numpy as np

from . import boundary, clips as clipmod, driver_id, tracker as tracking
from .calib import Camera

SELF_CALIBRATION = pathlib.Path("pipeline_out/self_calibration.json")
DETECTION_CACHE = pathlib.Path("pipeline_out/detections")


def _weights_fingerprint(path):
    """Content hash of the detector's weights, or the name if it has no file."""
    if not path:
        return "none"
    file = pathlib.Path(path)
    if not file.exists():
        return str(path)
    digest = hashlib.sha256()
    with file.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return f"{file.name}:{digest.hexdigest()[:16]}"


def detect_clip(clip, detector, cache=True, progress=None):
    """Detections for every frame, cached so a clip is judged under more than
    one calibration without paying for inference twice."""
    name = getattr(detector, "name", type(detector).__name__)
    path = DETECTION_CACHE / f"{clip.scenario}__{clip.camera}__{name}.pkl"
    fingerprint = _weights_fingerprint(getattr(detector, "weights", None))
    if cache and path.exists():
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        # Keyed on the weights' contents, not their path: retraining to the same
        # filename must invalidate the cache rather than silently reuse it.
        if blob.get("frames") and blob.get("fingerprint") == fingerprint:
            frames = {k: cv2.imdecode(v, cv2.IMREAD_COLOR) for k, v in blob["frames"].items()}
            return blob["detections"], frames

    per_frame, frames_bgr = [], {}
    for idx, frame in clip.frames():
        per_frame.append(detector(frame))
        if idx % 4 == 0:
            frames_bgr[idx] = frame
        if progress:
            progress(idx)
    if cache:
        DETECTION_CACHE.mkdir(parents=True, exist_ok=True)
        # The kept frames are only for livery sampling, so they cache as JPEG.
        encoded = {k: cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
                   for k, v in frames_bgr.items()}
        with path.open("wb") as fh:
            pickle.dump({"detections": per_frame, "frames": encoded,
                         "fingerprint": fingerprint}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return per_frame, frames_bgr


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
             bridge=4, cache=True, progress=None):
    """Run the whole stack over one clip and return a plain-dict result."""
    cameras = cameras or Camera.load_all()
    if clip.moving_camera:
        raise ValueError("run_clip handles fixed cameras; the moving rig is training data")
    H, sigma_calibration, calibration_label = load_homography(clip.camera, mode, cameras)
    H_inv = np.linalg.inv(H)

    per_frame, frames_bgr = detect_clip(clip, detector, cache=cache, progress=progress)

    tracked, _tracker = tracking.run(per_frame)
    by_track: dict[int, dict] = {}
    for idx, live in enumerate(tracked):
        for tid, det, _vel in live:
            if det.get("contacts_uv") is None:
                continue
            by_track.setdefault(tid, {})[idx] = det

    # Judge each track once, so the attribution ladder has a trajectory to
    # reason about, then attribute.
    per_track = {tid: boundary.judge_clip(frames, H_inv, fps=clip.fps)
                 for tid, frames in by_track.items()}
    attributions = driver_id.attribute(by_track, frames_bgr, per_track, fps=clip.fps)

    # An incident belongs to a car, not to a track id. If the tracker split one
    # car into two ids, judging each separately reports one excursion twice and
    # understates both. Regroup the detections under the attributed car and
    # judge the car's whole trajectory in one pass.
    grouped: dict[str, dict] = {}
    members: dict[str, list] = {}
    for tid, frames in by_track.items():
        attribution = attributions.get(tid)
        # Group by what the paint says, then let the ladder decide the name.
        # Two overlapping tracks whose livery vote agrees are one car, however
        # the ladder resolved the conflict between them.
        car = attribution.car_id if attribution else None
        if car is None and attribution is not None:
            car = (attribution.evidence or {}).get("livery_vote")
        key = car or f"track {tid}"
        target = grouped.setdefault(key, {})
        members.setdefault(key, []).append(tid)
        for idx, det in frames.items():
            if idx not in target or det["score"] > target[idx]["score"]:
                target[idx] = det

    grouped, members = _merge_coincident(grouped, members, H_inv)

    judgements, events = {}, {}
    for key, frames in grouped.items():
        js = boundary.judge_clip(frames, H_inv, fps=clip.fps)
        sigma_lateral = boundary.lateral_noise(js)
        evs = boundary.find_events(js, min_frames=min_frames, bridge=bridge, fps=clip.fps)
        attribution = _merged_attribution(key, members[key], attributions)
        driver_confidence = (attribution or {}).get("confidence", 0.0)
        for ev in evs:
            boundary.score_event(ev, js, sigma_lateral, sigma_calibration,
                                 fps=clip.fps, min_frames=min_frames)
            boundary.apply_driver_confidence(ev, driver_confidence)
        judgements[key] = js
        events[key] = evs

    return {
        "clip": clip.key,
        "scenario": clip.scenario,
        "camera": clip.camera,
        "fps": clip.fps,
        "detector": getattr(detector, "name", type(detector).__name__),
        "calibration_mode": mode,
        "calibration_label": calibration_label,
        "calibration_sigma_m": sigma_calibration,
        "sustained_min_frames": min_frames,
        "sustained_bridge_frames": bridge,
        "frames": len(per_frame),
        "detections_per_frame": [len(d) for d in per_frame],
        "tracks": {
            str(key): {
                "attribution": _merged_attribution(key, members[key], attributions),
                "track_ids": members[key],
                "lateral_noise_m": boundary.lateral_noise(judgements[key]),
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
                    for j in judgements[key]
                ],
                "events": [asdict(e) for e in events[key]],
            }
            for key in judgements
        },
    }


def _merge_coincident(grouped, members, H_inv, radius_m=2.5, share=0.6):
    """Fold an unnamed group into a named one when they are the same car.

    Image-space suppression cannot do this: two cars racing side by side overlap
    heavily in the picture. On the ground they never do -- a formula car is
    under 2 m wide -- so world-space coincidence is an unambiguous test for one
    car tracked twice.
    """
    named = [k for k in grouped if not k.startswith("track ")]
    unnamed = [k for k in grouped if k.startswith("track ")]
    if not named or not unnamed:
        return grouped, members

    def centres(frames):
        out = {}
        for idx, det in frames.items():
            world = boundary.back_project(det["contacts_uv"], H_inv)
            out[idx] = world.mean(axis=0)
        return out

    named_centres = {k: centres(grouped[k]) for k in named}
    for key in unnamed:
        mine = centres(grouped[key])
        for other in named:
            shared = set(mine) & set(named_centres[other])
            if not shared:
                continue
            close = sum(np.linalg.norm(mine[i] - named_centres[other][i]) < radius_m
                        for i in shared)
            if close >= share * len(shared):
                for idx, det in grouped[key].items():
                    if idx not in grouped[other] or det["score"] > grouped[other][idx]["score"]:
                        grouped[other][idx] = det
                members[other].extend(members.pop(key))
                grouped.pop(key)
                break
    return grouped, members


def _merged_attribution(key, track_ids, attributions):
    """The attribution for a merged group: the strongest of its tracks'."""
    claims = [attributions[t] for t in track_ids if t in attributions]
    named = [a for a in claims if a.car_id]
    best = max(named, key=lambda a: a.confidence) if named else (claims[0] if claims else None)
    if best is None:
        return None
    out = asdict(best)
    if len(track_ids) > 1:
        out["evidence"] = dict(out.get("evidence") or {},
                               merged_from_tracks=sorted(track_ids))
    return out


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
