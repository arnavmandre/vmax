"""Command line for the VMAX steward pipeline."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path("pipeline_out")


def cmd_selfcal(args):
    import time
    from . import clips, selfcalib
    from .calib import Camera, back_projection_error, track_frame_error
    cams = Camera.load_all()
    out = {}
    for name in clips.FIXED_CAMERAS:
        best = None
        for seed in args.seeds:
            t0 = time.time()
            bg = clips.static_background(clips.OUTPUT / "clean_lap" / f"{name}.mp4")
            est = selfcalib.estimate_homography(bg, seed=seed)
            est["seed"] = seed
            est["seconds"] = round(time.time() - t0, 1)
            print(f"  {name} seed {seed}: agreement {est['paint_agreement']:.3f}", flush=True)
            if best is None or est["paint_agreement"] > best["paint_agreement"]:
                best = est
        H = np.asarray(best.pop("homography"))
        best["homography"] = H.tolist()
        best["source_clip"] = f"clean_lap/{name}.mp4"
        best["ground_frame_error"] = back_projection_error(H, cams[name])
        best["track_frame_error"] = track_frame_error(H, cams[name])
        best["truth"] = {"position_m": cams[name].position.tolist(),
                         "look_at_m": cams[name].spec["look_at_m"],
                         "vertical_fov_deg": cams[name].spec["vertical_fov_deg"]}
        best["position_error_m"] = float(
            np.linalg.norm(np.array(best["position_m"]) - cams[name].position))
        out[name] = best
        print(f"{name}: position error {best['position_error_m']:.2f} m, "
              f"lateral {best['track_frame_error']['mean_abs_lateral_error_m']:.3f} m", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "self_calibration.json").write_text(json.dumps(out, indent=2))


def cmd_train(args):
    from .train import main as train_main
    argv = ["--iterations", str(args.iterations), "--batch", str(args.batch)]
    if args.resume:
        argv += ["--resume", args.resume]
    train_main(argv)


def _detector(args):
    if args.detector == "yolo-coco":
        from .detector import CocoYoloDetector
        return CocoYoloDetector(threshold=args.threshold)
    from .detector import VmaxDetector
    return VmaxDetector(weights=args.weights, threshold=args.threshold)


def cmd_run(args):
    from .pipeline import run_all
    detector = _detector(args)
    run_all(detector, mode=args.mode, scenarios=args.scenarios or None,
            cameras_wanted=args.cameras or None, out_dir=args.out)


def cmd_evaluate(args):
    from .evaluate import score_run
    report = score_run(run_dir=args.runs, out=args.out)
    print(json.dumps(report["summary"], indent=2))


def cmd_overlay(args):
    from . import clips as clipmod
    from .overlay import render_clip
    from .pipeline import load_homography
    by_key = {c.key: c for c in clipmod.discover(include_race_pace=False)}
    for path in sorted(pathlib.Path(args.runs).glob("*.json")):
        result = json.loads(path.read_text())
        key = f"{result['scenario']}/{result['camera']}"
        if args.only and key not in args.only:
            continue
        H, _sigma, _label = load_homography(result["camera"], result["calibration_mode"])
        out = pathlib.Path(args.out) / f"{result['scenario']}__{result['camera']}.mp4"
        render_clip(by_key[key], result, out, H)
        print("wrote", out, flush=True)


def cmd_baseline(args):
    from .baseline import run
    run(weights=args.weights, stride=args.stride, conf=args.threshold)


def cmd_report(args):
    from .report import build
    payload = build()
    print(f"{len(payload['cases'])} cases written to pipeline_out/dashboard.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vmax-steward", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("selfcal", help="estimate each camera's homography from video alone")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 13])
    p.set_defaults(func=cmd_selfcal)

    p = sub.add_parser("train", help="train VMAX-Net on the held-in clips")
    p.add_argument("--iterations", type=int, default=900)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="run the pipeline over the clips")
    p.add_argument("--detector", choices=["vmaxnet", "yolo-coco"], default="vmaxnet")
    p.add_argument("--weights", default="pipeline_out/vmaxnet.pt")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--mode", choices=["surveyed", "self"], default="surveyed")
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--cameras", nargs="*", default=None)
    p.add_argument("--out", default="pipeline_out/runs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("evaluate", help="score pipeline output against the labels")
    p.add_argument("--runs", default="pipeline_out/runs")
    p.add_argument("--out", default="pipeline_out/evaluation.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("overlay", help="render steward review footage")
    p.add_argument("--runs", default="pipeline_out/runs")
    p.add_argument("--out", default="pipeline_out/overlays")
    p.add_argument("--only", nargs="*", default=None)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("baseline", help="zero-shot COCO YOLOv8-seg baseline")
    p.add_argument("--weights", default="yolov8m-seg.pt")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.02)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("report", help="assemble the dashboard JSON")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
