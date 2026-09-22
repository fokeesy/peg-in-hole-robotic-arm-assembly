"""Runs one demo trial and saves an animated GIF of the whole sequence
(approach -> vision -> spiral search -> compliant insertion) plus the
overhead detection frame, to outputs/.

    python main.py --clearance-mm 2 --offset-mm 12

Pass --gui to watch it happen live in a real-time PyBullet window instead
(no GIF is saved in that mode - the whole point is to just watch):

    python main.py --clearance-mm 2 --offset-mm 12 --gui
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p

from src.domain_randomization import sample_trial
from src.trial import run_single_trial

OUT_DIR = Path(__file__).resolve().parent / "outputs"

VIEW_EYE = (0.95, -0.65, 0.55)
VIEW_TARGET = (0.5, 0.0, 0.05)


def _scene_frame(client):
    view = p.computeViewMatrix(VIEW_EYE, VIEW_TARGET, (0, 0, 1))
    proj = p.computeProjectionMatrixFOV(45.0, 1.3, 0.1, 3.0)
    _, _, rgba, _, _ = p.getCameraImage(
        480, 360, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER, physicsClientId=client,
    )
    return np.reshape(rgba, (360, 480, 4))[:, :, :3].astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument("--offset-mm", type=float, default=None,
                         help="if omitted, a random placement offset is sampled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                         help="watch the trial live in a real-time window instead of saving a GIF")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = sample_trial(args.clearance_mm / 1000.0, rng, placement_jitter=0.015, pixel_noise_std=1.0)
    if args.offset_mm is not None:
        # override the sampled offset with a specific, reproducible one for the demo
        angle = rng.uniform(0, 2 * np.pi)
        dx = (args.offset_mm / 1000.0) * np.cos(angle)
        dy = (args.offset_mm / 1000.0) * np.sin(angle)
        from src.domain_randomization import NOMINAL_HOLE_CENTER
        cfg["hole_center"] = (NOMINAL_HOLE_CENTER[0] + dx, NOMINAL_HOLE_CENTER[1] + dy)
        cfg["true_offset_mm"] = args.offset_mm

    print("Trial config:", {k: v for k, v in cfg.items()})

    if args.gui:
        print("Opening a live PyBullet window - it should pop up now (check your "
              "taskbar/dock if you don't see it appear on top).")
        result, _ = run_single_trial(cfg, capture_frames=False, seed=args.seed,
                                      gui=True, hold_open=True)
        print("Result:", result)
        return

    print("Running headless (no window) - this mode saves a GIF instead of "
          "showing anything live. Pass --gui if you want to watch it happen.")

    frames = []
    tick_count = [0]

    def on_tick(scene):
        tick_count[0] += 1
        if tick_count[0] % 2 == 0:  # keep the gif a reasonable size
            frames.append(_scene_frame(scene.client))

    result, capture = run_single_trial(cfg, capture_frames=True, seed=args.seed, on_tick=on_tick)
    print("Result:", result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if frames:
        # hold the final frame briefly so the outcome is visible on loop
        frames = frames + [frames[-1]] * 8
        imageio.mimsave(OUT_DIR / "demo_animation.gif", frames, duration=0.06)
        print(f"Saved animation: {OUT_DIR / 'demo_animation.gif'} ({len(frames)} frames)")

    if capture:
        import cv2
        cv2.imwrite(str(OUT_DIR / "overhead_detection.png"),
                     cv2.cvtColor(capture[0], cv2.COLOR_RGB2BGR))
        print(f"Saved overhead frame: {OUT_DIR / 'overhead_detection.png'}")


if __name__ == "__main__":
    main()
