"""Runs one full pick-agnostic peg-in-hole trial: build scene -> perceive -> insert -> report."""
import time

import numpy as np
import pybullet as p

from .sim_env import Scene
from .perception import estimate_hole_position
from .insertion import run_insertion, DOWN_ORN, GRASP_OFFSET_Z
from .sim_step import step as _step, enable_realtime, enable_wrist_camera
from . import gui_style

GRASP_HEIGHT = 0.32


def run_single_trial(trial_config, gui=False, capture_frames=False, insertion_config=None, seed=None,
                      on_tick=None, realtime=None, hold_open=False):
    """`gui`: open a visible PyBullet window instead of running headless.
    `realtime`: pace the simulation to real time (defaults to `gui`'s value,
    since a headless run has no reason to slow down, but a visible one is
    meant to be watched). `hold_open`: block with a prompt after the trial
    finishes instead of closing the window immediately, so a person watching
    can actually see the final result."""
    rng = np.random.default_rng(seed)
    insertion_config = insertion_config or {}
    if realtime is None:
        realtime = gui

    client = p.connect(p.GUI, options=gui_style.CONNECT_OPTIONS) if gui else p.connect(p.DIRECT)
    if realtime:
        enable_realtime(client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)
    if gui:
        gui_style.apply(client)
        p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=40, cameraPitch=-35,
                                      cameraTargetPosition=[0.5, 0.0, 0.1], physicsClientId=client)

    frames = []
    try:
        scene = Scene(
            client,
            clearance=trial_config["clearance"],
            hole_center=trial_config["hole_center"],
        )
        if gui:
            # feeds PyBullet's live preview pane throughout the whole trial -
            # grasp, transit, spiral search, insertion - since every phase
            # steps the simulation through this same shared `_step` function
            enable_wrist_camera(client, scene.robot)
        for _ in range(60):
            _step(client)

        # the peg is rigidly attached to the gripper (see below) and dangles
        # GRASP_OFFSET_Z below the hand once grasped, so it can swing close to
        # the forearm/wrist links as the arm moves - disable peg<->robot
        # collision entirely rather than just for the hand/fingers
        for link in range(-1, p.getNumJoints(scene.robot.id, physicsClientId=client)):
            p.setCollisionFilterPair(scene.peg_id, scene.robot.id, -1, link, 0, physicsClientId=client)

        # simulate a completed grasp, but *visibly*: hover with the gripper
        # open, descend so the peg sits between the fingers, close the
        # fingers around it, then rigidly attach (grasp planning itself is
        # out of scope for this project - see README). Attaching without this
        # approach-and-close sequence would just teleport the peg from the
        # table straight into the gripper, which looks like it's being
        # sucked up rather than picked up.
        hcx, hcy = trial_config["hole_center"]
        grasp_xy = (hcx, hcy - 0.25)
        scene.robot.set_gripper(0.035)
        scene.robot.move_to_pose((grasp_xy[0], grasp_xy[1], GRASP_HEIGHT), DOWN_ORN, sim_steps=90)

        # descend to the height where the peg (resting on the table) sits
        # between the open fingers - this is the same GRASP_OFFSET_Z
        # relationship the rest of the pipeline assumes, so no teleport is
        # needed to align the peg with the attach point afterward
        pick_z = scene.peg_len / 2.0 + GRASP_OFFSET_Z
        scene.robot.move_to_pose((grasp_xy[0], grasp_xy[1], pick_z), DOWN_ORN, sim_steps=90)

        # close the fingers around it. This is a visual close, not what
        # actually holds the peg - contact-only grasping is prone to slipping,
        # which would make the rest of the pipeline's physics unreliable, so
        # the fixed constraint below is the real attachment.
        scene.robot.set_gripper(scene.peg_side / 2.0 - 0.001)
        for _ in range(40):
            _step(client)

        scene.robot.attach_peg(scene.peg_id)
        for _ in range(20):
            _step(client)

        scene.robot.move_to_pose((grasp_xy[0], grasp_xy[1], GRASP_HEIGHT), DOWN_ORN, sim_steps=90)

        vision_rng = np.random.default_rng(seed if seed is not None else 0)
        hole_estimate, rgb_frame = estimate_hole_position(
            client, pixel_noise_std=trial_config.get("pixel_noise_std", 0.0), rng=vision_rng
        )
        true_hole = trial_config["hole_center"]

        if hole_estimate is None:
            return {
                "success": False,
                "reason": "vision_failed",
                "clearance_mm": 1000.0 * trial_config["clearance"],
                "true_offset_mm": trial_config.get("true_offset_mm", None),
                "vision_error_mm": None,
            }, frames

        vision_error_mm = 1000.0 * float(np.hypot(
            hole_estimate[0] - true_hole[0], hole_estimate[1] - true_hole[1]
        ))

        if capture_frames:
            frames.append(rgb_frame)

        # transit to above the estimated hole at a safe, fixed height *before*
        # descending. Joint-space interpolation between two far-apart poses does
        # not stay on the Cartesian straight line between them, so a single move
        # straight from the grasp pose down to the approach height can clip the
        # fixture mid-flight; a horizontal move at a safe altitude first, then a
        # purely vertical descent, avoids that.
        scene.robot.move_to_pose((hole_estimate[0], hole_estimate[1], GRASP_HEIGHT),
                                  DOWN_ORN, sim_steps=90)

        result = run_insertion(scene.robot, scene, hole_estimate, insertion_config, on_tick=on_tick)
        result["clearance_mm"] = 1000.0 * trial_config["clearance"]
        result["true_offset_mm"] = trial_config.get("true_offset_mm", None)
        result["vision_error_mm"] = vision_error_mm
        if gui and hold_open:
            print(f"\nResult: {result}")
            # Waiting on input() here is fragile: some terminals/launchers
            # don't attach an interactive stdin, and input() then raises
            # EOFError immediately - which looked like "the window flashes
            # and closes" rather than an error, since the exception unwinds
            # straight to the disconnect below. Just close the actual window
            # instead - no dependency on the terminal's stdin at all.
            print("Trial finished - close the window to exit.")
            try:
                while p.isConnected(client):
                    time.sleep(0.1)
            except p.error:
                pass
        return result, frames
    finally:
        p.disconnect(physicsClientId=client)
