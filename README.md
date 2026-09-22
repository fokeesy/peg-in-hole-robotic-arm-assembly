# Vision-Guided Compliant Peg-in-Hole Assembly

A from-scratch, fully-simulated precision assembly pipeline: a Franka Panda
arm perceives a hole opening with a synthetic camera + classical computer
vision, plans an approach, and inserts a peg using a **spiral search +
force-guided compliant insertion controller** — the standard algorithm
industrial robot makers (FANUC, Yaskawa, DENSO, Mitsubishi Electric) use for
precision assembly. Everything runs in [PyBullet](https://pybullet.org/) on a
laptop; no real hardware required.

This mirrors the actual task those companies' robots perform on a factory
floor, end to end: **perception** (find the hole from a camera image),
**planning** (approach trajectory), and **control** (compliant, force-guided
insertion) in one pipeline, with quantitative, reproducible results rather
than a single anecdotal demo.

## What it does

1. **Builds a randomized scene**: a square peg gripped by a Panda arm, and a
   square-hole fixture (with a lead-in chamfer, like a real fixture) whose
   position and clearance are randomized per trial.
2. **Perceives the hole** from a synthetic overhead camera image using
   classical CV (HSV color segmentation + contour hierarchy to find the hole
   as a "hole" in the frame's binary mask, then unprojects the pixel to world
   coordinates via the camera's view/projection matrices) — the camera is
   fixed in the world, so the hole's position is genuinely recovered from the
   image every trial, not read from the simulator's ground truth.
3. **Approaches** the estimated hole location with a scripted, collision-safe
   trajectory (lift → transit → descend).
4. **Spiral-searches** for the actual opening (an outward Archimedean spiral)
   to absorb the residual vision/placement error, since the estimate is never
   exact.
5. **Inserts compliantly**: an admittance-control law reads simulated contact
   forces (standing in for a wrist force/torque sensor) and yields laterally
   in the direction the fixture is pushing back, instead of grinding the peg
   into whatever it's touching — this is what lets it recover from
   misalignment that the spiral search alone doesn't fully resolve.
6. **Evaluates it properly**: sweeps hole clearance (and placement/vision
   noise) across many randomized trials and produces robustness curves
   (success rate vs. clearance, search effort vs. placement error, failure
   mode breakdown) — see `outputs/`.

## Quickstart

```bash
pip install -r requirements.txt

# run one trial and save a GIF of the whole sequence + the detection frame
python main.py --clearance-mm 2 --offset-mm 12

# watch a trial happen live in a real-time window instead
python main.py --clearance-mm 2 --offset-mm 12 --gui

# open a manual playground: drag joint sliders, or click-and-drag the arm
# directly in the 3D view (native PyBullet mouse control)
python interactive.py

# run the full evaluation sweep (robustness curves)
python -m src.evaluate --trials-per-point 15

# run the test suite
python -m pytest tests/ -v
```

Everything runs headless (no window, full speed) by default, since the
evaluation sweep needs hundreds of trials to run fast. `--gui` and
`interactive.py` open a real PyBullet window and pace the simulation to real
time so it's actually watchable.

## Results

See `outputs/success_vs_vision_noise.png`, `outputs/search_effort.png`,
`outputs/failure_modes.png`, and `outputs/results.csv` for the full sweep.
`outputs/demo_animation.gif` and `outputs/overhead_detection.png` show a
single trial end to end.

**The finding that shaped the evaluation design**: physical hole clearance
and raw placement jitter turned out *not* to be the binding constraint. The
camera always looks at wherever the fixture actually is, so the spiral
search's lead-in absorbs a wide range of placement offset (tens of mm) with
no trouble at all, regardless of clearance. What actually drives
success/failure is **residual vision localization error** (`pixel_noise_std`
in the code) — how far the *estimate* is from the *true* hole center — which
is what the spiral has to blindly search around. So the primary sweep here is
vision noise, at a few representative clearances, rather than clearance
alone.

**Headline numbers from the full sweep** (288 trials: 3 clearances × 8 noise
levels × 12 trials each):

- **63.2% overall success rate**, dropping cleanly from ~100% at zero vision
  noise to ~15–20% at the highest noise level tested (20px std), with looser
  clearance consistently more forgiving.
- **Spiral-search effort scales with vision error** in the classic
  Archimedean-spiral way: near-zero extra iterations below ~20mm of error,
  then roughly quadratic growth beyond it (see `search_effort.png`).
- **The dominant failure mode is `spiral_exhausted`** (88 of 106 failures) —
  the search ran out of budget before finding the opening — not jamming
  during insertion (only 2 failures). In other words: once the compliant
  insertion controller actually finds the hole, it reliably finishes the
  job; the real bottleneck is bounding how far off the vision estimate can be
  before the search radius can't recover it in time.

## Architecture

```
src/
  robot.py        Panda arm: joint discovery, IK, macro moves, and a
                   closed-loop Jacobian (resolved-rate) servo for the small,
                   contact-sensitive spiral/insertion steps
  sim_env.py       Scene construction: ground, peg, chamfered hole fixture
  perception.py    Synthetic camera + classical CV hole detection
  insertion.py     Spiral search + force-guided compliant insertion
  domain_randomization.py   Per-trial clearance/placement/noise sampling
  trial.py         Wires one full trial together
  evaluate.py      Sweeps trials, saves CSV + plots
  sim_step.py      Shared simulation-step wrapper: real-time pacing and the
                   wrist-camera refresh for any GUI session, so every phase
                   of a run (grasp, transit, search, insertion) gets both
                   automatically, with no call site needing to remember to
  wrist_camera.py  Camera mounted on the gripper's wrist, looking down its
                   approach axis - feeds PyBullet's live preview pane
  gui_style.py     Visual polish for the PyBullet window (shadows, dark
                   background, decluttered preview panels)
main.py            Single-trial demo -> GIF + detection frame (or --gui to watch live)
interactive.py     Manual playground: joint sliders + mouse-drag control,
                   with the live wrist-camera feed
tests/             pytest suite (perception, kinematics, insertion)
```

## Simplifications (stated up front)

- **Grasping is assumed, not planned.** The arm visibly hovers over the peg
  with the gripper open, descends so the peg sits between the fingers, and
  closes them — but what actually *holds* the peg is a rigid constraint
  created at that moment, not finger-contact friction (contact-only grasping
  is prone to slipping, which would make the rest of the pipeline's physics
  unreliable). Grasp planning itself (deciding *where* and *how* to grasp) is
  a separate, large problem; this project is scoped to perception + approach
  + compliant insertion, which is where the real precision-assembly
  difficulty lives.
- **The peg and hole are square**, not round, so the fixture geometry could
  be built from primitive box shapes (no mesh authoring needed) while keeping
  the physics (contact, friction, jamming) genuine. Square-peg-in-square-hole
  is a real, standard NIST-style assembly benchmark, not a toy simplification
  of the control problem.
- **One fixed overhead camera**, not eye-in-hand. This is simpler to reason
  about but means the robot's own arm can occlude the view for some
  configurations — a real limitation of single fixed-camera systems, not
  something the code special-cases around (see `tests/test_perception.py`).

## Engineering pitfalls hit along the way (and why they matter)

Getting a physically simulated compliant-insertion controller to actually
work — not just look plausible — surfaced several non-obvious bugs, each a
useful lesson in its own right:

- **A "180° tilt" that wasn't.** The peg is a symmetric box grasped pointing
  down (rotated 180° about the approach axis). A naive `arccos(dot(local_z,
  world_z))` tilt metric scored a perfectly upright peg as maximally tilted.
  Fix: use the *absolute* value of the dot product — a symmetric part doesn't
  care which end is "up."
- **An inverted admittance sign.** The compliant-insertion law initially
  corrected *opposite* the contact-force direction, actively grinding the peg
  into whatever it touched instead of yielding away from it. The correct law
  moves *with* the push (`correction = +k · F`), which is what actually
  relieves contact force.
- **Position-control micro-steps collapse into gravity sag.** Converting a
  desired joint *velocity* into a tiny per-physics-step position delta
  (`target = current + dq·dt`) produced commands smaller than the position
  controller's own steady-state gravity error — the arm would visibly stall
  even with a large, correct velocity command. Fix: drive the joints with
  actual `VELOCITY_CONTROL`, not position deltas disguised as velocity.
- **Un-modeled fingers silently blocking descent.** The gripper fingers are
  left open (grasping is simplified to a rigid attach), and their collision
  geometry was still active — they were hitting the ground/fixture with
  hundreds of newtons of force at low approach heights, with no visible
  symptom except "the arm won't go any lower." Disabling finger collision
  against the ground/fixture (since they're not functionally grasping
  anyway) fixed it.
- **The grasp point wasn't where the fingers actually are.** The peg was
  initially attached exactly at the hand link's own coordinate frame origin
  — which is *inside* the hand's physical body. A real grasp holds an object
  ~116mm below that origin, out at the fingertips. Attaching the peg there
  instead is what let the hand clear the fixture while the peg still reached
  the hole (and included a sign bug: the hand's local `+Z` maps to world
  `-Z` under the "pointing down" orientation, so the offset had to be
  positive, not negative, to land below the hand).
- **Joint-space interpolation isn't Cartesian-straight.** A single IK move
  from a high "grasp" pose to a low "approach" pose let the elbow swing
  through a path that clipped the fixture mid-flight, even though both
  endpoints were individually collision-free. Fix: explicit lift → transit →
  descend waypoints (standard pick-and-place practice), not an implicit
  trust that joint-space interpolation stays out of the way.
- **A depth+tilt-only success check can't tell "inserted" from "missed
  entirely."** When the vision estimate is far enough off, the peg can
  descend over open ground with no fixture there at all — nothing stops it,
  so it reaches full depth with near-zero tilt just like a real insertion. An
  early, wider evaluation sweep reported a flat 100% success rate because of
  exactly this: successes at absurd vision errors (100mm+) were mostly false
  positives. Fix: success also requires the peg's XY position to actually be
  within the hole's tolerance of the *true* hole center
  (`Scene.peg_xy_error`), not just "something happened to stop it."
- **A camera pointed at its own arm.** The wrist camera's "eye" offset had
  the same class of sign error as the grasp offset above: it placed the eye
  *behind* the hand (into the forearm) instead of forward past the fingers,
  so it was rendering the inside of the arm's own geometry - like a selfie
  with a hand over the lens. Caught by actually saving and looking at a
  rendered frame, not just checking the call didn't crash.
- **A blocking `input()` looks identical to a crash.** Waiting on `input()`
  before closing the GUI window is fragile: some terminals/launchers don't
  attach an interactive stdin, and `input()` then raises `EOFError`
  immediately - which unwinds straight to disconnect, so the window opens
  and closes almost instantly with no visible error. Fixed by waiting on the
  window's own connection state instead (`p.isConnected`), which has no
  dependency on the terminal at all.

## Possible extensions

- Swap the classical-CV hole detector for a learned pose estimator, and
  measure whether/how much of the accuracy gap the compliant controller
  still needs to absorb.
- Replace the rigid-attach "grasp" with an actual closed-finger grasp
  (contact-based, with a grasp-quality check).
- Add rotational (not just lateral) admittance to handle a peg that jams at
  an angle rather than just off-center.
