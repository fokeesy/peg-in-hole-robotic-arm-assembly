"""Randomizes per-trial conditions: hole clearance, fixture placement, and camera noise."""
import numpy as np

NOMINAL_HOLE_CENTER = (0.5, 0.0)


def sample_trial(clearance, rng, placement_jitter=0.012, pixel_noise_std=0.6):
    """`clearance` is one-sided (m); `placement_jitter` is the max |x|,|y| offset (m)
    applied to the true hole position, standing in for fixture placement tolerance."""
    dx = rng.uniform(-placement_jitter, placement_jitter)
    dy = rng.uniform(-placement_jitter, placement_jitter)
    hole_center = (NOMINAL_HOLE_CENTER[0] + dx, NOMINAL_HOLE_CENTER[1] + dy)
    return {
        "clearance": clearance,
        "hole_center": hole_center,
        "true_offset_mm": 1000.0 * float(np.hypot(dx, dy)),
        "pixel_noise_std": pixel_noise_std,
    }
