"""Sweeps trial conditions across many randomized runs and produces the
robustness curves that are the real deliverable of this project.

Early exploration (see README) found that physical hole clearance and raw
placement jitter barely matter on their own: the spiral search's lead-in
absorbs a wide range of *placement* offset automatically, because the camera
always looks at wherever the fixture actually is. What actually drives
success/failure is the *residual vision localization error* (pixel_noise_std)
combined with clearance - vision error sets how far the spiral has to search
before the peg is even near the opening, and clearance sets how much slack
the final compliant-insertion phase has once it's there. So the primary sweep
here is vision noise, at a few representative clearances.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .domain_randomization import sample_trial
from .trial import run_single_trial

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def sweep(pixel_noise_values, clearances_mm, trials_per_point=15, placement_jitter=0.015, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    total = len(clearances_mm) * len(pixel_noise_values) * trials_per_point
    done = 0
    for clearance_mm in clearances_mm:
        clearance = clearance_mm / 1000.0
        for pixel_noise_std in pixel_noise_values:
            for trial_idx in range(trials_per_point):
                cfg = sample_trial(clearance, rng, placement_jitter=placement_jitter,
                                    pixel_noise_std=pixel_noise_std)
                result, _ = run_single_trial(cfg, seed=int(rng.integers(0, 2**31 - 1)))
                row = {
                    "clearance_mm": clearance_mm,
                    "pixel_noise_std": pixel_noise_std,
                    "trial_idx": trial_idx,
                    "success": result["success"],
                    "reason": result.get("reason"),
                    "true_offset_mm": cfg["true_offset_mm"],
                    "vision_error_mm": result.get("vision_error_mm"),
                    "spiral_iters": result.get("spiral_iters"),
                    "insert_iters": result.get("insert_iters"),
                    "insertion_depth_mm": 1000.0 * result.get("insertion_depth", 0.0),
                    "tilt_deg": result.get("tilt_deg"),
                    "xy_error_mm": result.get("xy_error_mm"),
                    "max_contact_force": result.get("max_contact_force"),
                }
                rows.append(row)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{total}] clearance={clearance_mm:.2f}mm noise={pixel_noise_std} "
                      f"vision_err={row['vision_error_mm']} success={result['success']} "
                      f"reason={result.get('reason')} ({elapsed:.0f}s elapsed)")
    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clearances = sorted(set(r["clearance_mm"] for r in rows))
    noise_levels = sorted(set(r["pixel_noise_std"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(clearances):
        rates, cis = [], []
        for pn in noise_levels:
            subset = [r for r in rows if r["clearance_mm"] == c and r["pixel_noise_std"] == pn]
            successes = [r["success"] for r in subset]
            p_hat = np.mean(successes)
            n = len(successes)
            se = np.sqrt(p_hat * (1 - p_hat) / n) if n > 0 else 0.0
            rates.append(100 * p_hat)
            cis.append(100 * 1.96 * se)
        color = cmap(i / max(1, len(clearances) - 1))
        ax.errorbar(noise_levels, rates, yerr=cis, marker="o", capsize=3,
                     label=f"{c:g} mm clearance", color=color)
    ax.set_xlabel("Camera pixel-localization noise (std, px)")
    ax.set_ylabel("Insertion success rate (%)")
    ax.set_title("Success rate vs. vision noise, by clearance")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_vision_noise.png", dpi=150)
    plt.close(fig)

    # vision error -> search effort, for trials that succeeded
    succ_rows = [r for r in rows if r["success"]]
    if succ_rows:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        errs = [r["vision_error_mm"] for r in succ_rows]
        spirals = [r["spiral_iters"] for r in succ_rows]
        sc = ax.scatter(errs, spirals, c=[r["clearance_mm"] for r in succ_rows],
                         cmap="viridis", alpha=0.75)
        ax.set_xlabel("Vision localization error (mm)")
        ax.set_ylabel("Spiral-search iterations needed")
        ax.set_title("Search effort vs. vision error (successful trials)")
        cbar = fig.colorbar(sc)
        cbar.set_label("Clearance (mm)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "search_effort.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    reasons = sorted(set(r["reason"] for r in rows if not r["success"]))
    if reasons:
        counts = [sum(1 for r in rows if r["reason"] == reason) for reason in reasons]
        ax.bar(reasons, counts, color="#c53030")
        ax.set_ylabel("Count")
        ax.set_title("Failure modes across all trials")
        fig.tight_layout()
        fig.savefig(out_dir / "failure_modes.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-noise-values", type=float, nargs="+",
                         default=[0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0])
    parser.add_argument("--clearances-mm", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    parser.add_argument("--trials-per-point", type=int, default=12)
    parser.add_argument("--placement-jitter-mm", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = sweep(
        args.pixel_noise_values,
        args.clearances_mm,
        trials_per_point=args.trials_per_point,
        placement_jitter=args.placement_jitter_mm / 1000.0,
        seed=args.seed,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(rows, OUT_DIR / "results.csv")
    make_plots(rows, OUT_DIR)

    overall = np.mean([r["success"] for r in rows])
    print(f"\nOverall success rate: {100 * overall:.1f}% over {len(rows)} trials")
    print(f"Results written to {OUT_DIR / 'results.csv'}")
    print(f"Plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
