"""Run video-only self-calibration for every fixed camera and score it."""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, '.')
from vmax_vision import clips, selfcalib
from vmax_vision.calib import Camera, back_projection_error, track_frame_error

def main():
    cams = Camera.load_all()
    out = {}
    for name in clips.FIXED_CAMERAS:
        video = clips.OUTPUT / 'clean_lap' / f'{name}.mp4'
        t0 = time.time()
        bg = clips.static_background(video)
        est = selfcalib.estimate_homography(bg, seed=7)
        est['seconds'] = round(time.time() - t0, 1)
        est['source_clip'] = f'clean_lap/{name}.mp4'
        H = est.pop('homography')
        est['homography'] = np.asarray(H).tolist()
        est['ground_frame_error'] = back_projection_error(H, cams[name])
        est['track_frame_error'] = track_frame_error(H, cams[name])
        est['truth'] = {'position_m': cams[name].position.tolist(),
                        'look_at_m': cams[name].spec['look_at_m'],
                        'vertical_fov_deg': cams[name].spec['vertical_fov_deg']}
        est['position_error_m'] = float(np.linalg.norm(np.array(est['position_m']) - cams[name].position))
        out[name] = est
        print(name, 'pos err %.2f m' % est['position_error_m'],
              '| lateral %.3f m' % est['track_frame_error']['mean_abs_lateral_error_m'],
              '| gauge %.2f m' % est['track_frame_error']['along_track_gauge_m'], flush=True)
    p = pathlib.Path('pipeline_out/self_calibration.json')
    p.write_text(json.dumps(out, indent=2))
    print('wrote', p)

if __name__ == '__main__':
    main()
