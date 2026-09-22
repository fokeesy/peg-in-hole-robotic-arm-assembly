"""Spiral search + force-guided compliant insertion controller.

Strategy (the classic industrial peg-in-hole algorithm):
  1. Hover above the vision-estimated hole center and descend to the fixture surface.
  2. Spiral search: walk an outward Archimedean spiral while pressing lightly down,
     until the peg tip drops past the fixture's top surface (i.e. it found the opening).
  3. Compliant insertion: continue descending while an admittance law uses the
     simulated contact-force feedback (from PyBullet contact points, standing in
     for a wrist force/torque sensor) to correct lateral position and avoid jamming.

Both phases drive the arm with `robot.track_to`, a closed-loop Jacobian servo
toward a persistent, monotonically-advancing setpoint. That matters: an
open-loop "move by a tiny delta from wherever you currently are" command stalls
under gravity sag and contact disturbance (each call re-bases off an
already-lagging actual position), whereas a setpoint that keeps advancing and
is chased by an error-proportional controller keeps making progress until it
is genuinely blocked.
"""
import numpy as np
import pybullet as p

DOWN_ORN = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
GRASP_OFFSET_Z = 0.116  # hand-origin-to-fingertip distance on this URDF


def contact_force(scene):
    fx = fy = fz = 0.0
    for c in scene.peg_contacts_frame():
        normal_force = c[9]
        nx, ny, nz = c[7]
        fx += normal_force * nx
        fy += normal_force * ny
        fz += normal_force * nz
    return np.array([fx, fy, fz])


def run_insertion(robot, scene, hole_estimate_xy, config, on_tick=None):
    peg_len = scene.peg_len
    frame_top = scene.table_z + scene.frame_height
    # these are hand (end-effector) heights, not peg heights: the peg hangs
    # GRASP_OFFSET_Z + peg_len/2 below the hand when pointing straight down
    hand_drop = GRASP_OFFSET_Z + peg_len / 2.0
    hover_z = frame_top + hand_drop + 0.06
    approach_z = frame_top + hand_drop + 0.008

    hx, hy = hole_estimate_xy

    # 1. approach & descend to the fixture surface, directly above the *estimated* hole
    robot.move_to_pose((hx, hy, hover_z), DOWN_ORN, sim_steps=90)
    robot.move_to_pose((hx, hy, approach_z), DOWN_ORN, sim_steps=90)

    # 2. spiral search
    pitch = config.get("spiral_pitch", 0.006)
    dtheta = config.get("spiral_dtheta", 0.45)
    max_spiral_iters = config.get("max_spiral_iters", 220)
    press_step = config.get("press_step", 0.0012)
    drop_margin = config.get("drop_margin", 0.004)
    floor_z = scene.table_z + hand_drop  # hand height when the peg is fully seated

    target = np.array([hx, hy, approach_z])
    found = False
    spiral_iters_used = 0

    for n in range(1, max_spiral_iters + 1):
        theta = n * dtheta
        r = pitch * theta / (2.0 * np.pi)
        target[0] = hx + r * np.cos(theta)
        target[1] = hy + r * np.sin(theta)
        target[2] = max(floor_z, target[2] - press_step)

        robot.track_to(target, DOWN_ORN, sim_steps=8)
        spiral_iters_used = n
        if on_tick is not None:
            on_tick(scene)

        peg_pos, _ = scene.peg_state()
        peg_tip_z = peg_pos[2] - peg_len / 2.0
        if peg_tip_z < frame_top - drop_margin:
            found = True
            break

    if not found:
        return {
            "success": False,
            "reason": "spiral_exhausted",
            "spiral_iters": spiral_iters_used,
            "insertion_depth": scene.insertion_depth(),
            "tilt_deg": scene.peg_tilt_deg(),
        }

    # 3. compliant, force-guided insertion
    k_force = config.get("admittance_gain", 0.00035)
    max_nudge = config.get("max_lateral_nudge", 0.0015)
    insert_step = config.get("insert_step", 0.0015)
    max_insert_iters = config.get("max_insert_iters", 220)
    success_depth = config.get("success_depth", scene.frame_height * 0.9)
    tilt_limit = config.get("tilt_limit_deg", 14.0)
    stuck_patience = config.get("stuck_patience", 25)
    # peg center must actually be within the hole opening, not just "some
    # depth was reached" - a peg that free-fell onto open ground far from the
    # fixture reads identically to a real insertion on depth/tilt alone
    xy_tolerance = scene.peg_side / 2.0 + scene.clearance + 0.002

    ee_pos, _ = robot.ee_pose()
    target = np.array(ee_pos, dtype=float)

    stuck_counter = 0
    last_depth = scene.insertion_depth()
    max_force_seen = 0.0

    for i in range(max_insert_iters):
        force = contact_force(scene)
        lateral = force[:2]
        max_force_seen = max(max_force_seen, float(np.linalg.norm(force)))
        # admittance law: yield in the direction the wall is pushing (contact
        # normal points wall -> peg), which relieves the contact instead of
        # grinding the peg further into whatever it is touching
        correction = np.clip(k_force * lateral, -max_nudge, max_nudge)
        target[0] += correction[0]
        target[1] += correction[1]
        target[2] = max(floor_z, target[2] - insert_step)

        robot.track_to(target, DOWN_ORN, sim_steps=8)
        if on_tick is not None:
            on_tick(scene)

        depth = scene.insertion_depth()
        tilt = scene.peg_tilt_deg()
        xy_error = scene.peg_xy_error()

        if depth >= success_depth and tilt <= tilt_limit and xy_error <= xy_tolerance:
            return {
                "success": True,
                "reason": "inserted",
                "spiral_iters": spiral_iters_used,
                "insert_iters": i + 1,
                "insertion_depth": depth,
                "tilt_deg": tilt,
                "xy_error_mm": 1000.0 * xy_error,
                "max_contact_force": max_force_seen,
            }

        if depth <= last_depth + 1e-5:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_depth = depth

        if stuck_counter >= stuck_patience or tilt > tilt_limit * 1.8:
            return {
                "success": False,
                "reason": "jammed",
                "spiral_iters": spiral_iters_used,
                "insert_iters": i + 1,
                "insertion_depth": depth,
                "tilt_deg": tilt,
                "max_contact_force": max_force_seen,
            }

    return {
        "success": False,
        "reason": "timeout",
        "spiral_iters": spiral_iters_used,
        "insert_iters": max_insert_iters,
        "insertion_depth": scene.insertion_depth(),
        "tilt_deg": scene.peg_tilt_deg(),
        "max_contact_force": max_force_seen,
    }
