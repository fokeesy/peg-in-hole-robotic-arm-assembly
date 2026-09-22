import numpy as np
import pybullet as p
import pytest

from src.sim_env import Scene
from src.insertion import DOWN_ORN


@pytest.mark.parametrize("target", [(0.5, 0.0, 0.15), (0.45, -0.1, 0.2), (0.55, 0.08, 0.12)])
def test_ik_reaches_target_pose(target):
    """A fresh (non-warm-started) IK solve should place the end effector within
    a few millimeters of the requested pose - this is what every macro move in
    the pipeline (approach, hover, grasp) depends on."""
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        scene = Scene(client, clearance=0.003, hole_center=(0.5, 0.0))
        sol = scene.robot.solve_ik(target, DOWN_ORN, warm_start=False)
        for j, val in zip(scene.robot.arm_joints, sol):
            p.resetJointState(scene.robot.id, j, val, physicsClientId=client)
        pos, _ = scene.robot.ee_pose()
        error_mm = 1000.0 * np.linalg.norm(np.array(pos) - np.array(target))
        assert error_mm < 5.0, f"IK residual {error_mm:.2f}mm exceeds 5mm"
    finally:
        p.disconnect(physicsClientId=client)


def test_move_to_pose_converges_without_finger_ground_collision():
    """Regression test for the finger/ground collision bug that silently
    capped how low the arm could reach (see README pitfalls)."""
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        scene = Scene(client, clearance=0.003, hole_center=(0.5, 0.0))
        for _ in range(30):
            p.stepSimulation(physicsClientId=client)
        low_target = (0.5, 0.0, 0.1)
        scene.robot.move_to_pose(low_target, DOWN_ORN, sim_steps=60, settle_steps=60)
        pos, _ = scene.robot.ee_pose()
        error_mm = 1000.0 * abs(pos[2] - low_target[2])
        assert error_mm < 15.0, f"arm did not converge to the low target (z error {error_mm:.1f}mm)"
    finally:
        p.disconnect(physicsClientId=client)
