import numpy as np
import pytest

from src.trial import run_single_trial
from src.domain_randomization import sample_trial


def test_easy_case_succeeds():
    """Generous clearance, no misalignment: the pipeline should reliably insert."""
    config = {"clearance": 0.003, "hole_center": (0.5, 0.0),
              "true_offset_mm": 0.0, "pixel_noise_std": 0.0}
    result, _ = run_single_trial(config, seed=1)
    assert result["success"], result
    assert result["insertion_depth"] > 0.025


def test_peg_tilt_treats_the_180_degree_grasp_as_upright():
    """Regression test: the peg is grasped pointing down (a 180-degree flip
    about X), which used to be mis-scored as a ~180-degree tilt."""
    config = {"clearance": 0.003, "hole_center": (0.5, 0.0),
              "true_offset_mm": 0.0, "pixel_noise_std": 0.0}
    result, _ = run_single_trial(config, seed=1)
    assert result["tilt_deg"] < 20.0, "an upright peg should not be reported as tilted"


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_random_trials_produce_consistent_result_shape(seed):
    rng = np.random.default_rng(seed)
    cfg = sample_trial(0.003, rng, placement_jitter=0.01, pixel_noise_std=0.6)
    result, _ = run_single_trial(cfg, seed=seed)
    assert "success" in result
    assert "insertion_depth" in result
    assert "tilt_deg" in result
    assert result["clearance_mm"] == pytest.approx(3.0)


def test_success_requires_actual_xy_alignment_with_the_hole():
    """Regression test: depth + tilt alone can't tell 'seated in the hole'
    apart from 'the vision estimate was so far off the peg free-fell onto
    open ground nowhere near the fixture' - both look identical without an
    explicit XY-proximity check against the true hole (see README)."""
    config = {"clearance": 0.003, "hole_center": (0.5, 0.0),
              "true_offset_mm": 0.0, "pixel_noise_std": 0.0}
    result, _ = run_single_trial(config, seed=1)
    assert result["success"]
    assert result["xy_error_mm"] < 20.0


def test_vision_error_scales_with_pixel_noise():
    """More camera noise should on average produce larger vision error - a
    basic sanity check that the noise injection actually reaches detection."""
    rng = np.random.default_rng(0)
    errs_low, errs_high = [], []
    for i in range(4):
        cfg_low = sample_trial(0.004, rng, placement_jitter=0.005, pixel_noise_std=0.0)
        r, _ = run_single_trial(cfg_low, seed=100 + i)
        if r.get("vision_error_mm") is not None:
            errs_low.append(r["vision_error_mm"])
    for i in range(4):
        cfg_high = sample_trial(0.004, rng, placement_jitter=0.005, pixel_noise_std=3.0)
        r, _ = run_single_trial(cfg_high, seed=200 + i)
        if r.get("vision_error_mm") is not None:
            errs_high.append(r["vision_error_mm"])
    assert np.mean(errs_high) > np.mean(errs_low)
