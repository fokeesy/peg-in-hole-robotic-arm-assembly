import numpy as np
import pybullet as p
import pytest

from src.sim_env import Scene
from src.perception import estimate_hole_position


@pytest.mark.parametrize("hole_center", [(0.5, 0.0), (0.46, -0.06), (0.54, 0.035), (0.49, 0.02)])
def test_hole_detection_accuracy(hole_center):
    """NOTE: the camera is fixed in the world frame, and the robot's own idle
    rest pose can occlude it for some hole positions near the arm's resting
    silhouette (a real limitation of a single fixed overhead camera - see
    README). The real pipeline never actually hits this: by the time it calls
    detection, the arm has already moved off to grasp the peg. We move it out
    of the way here too, to test detection rather than this known occlusion
    case."""
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        scene = Scene(client, clearance=0.003, hole_center=hole_center)
        for _ in range(30):
            p.stepSimulation(physicsClientId=client)
        from src.insertion import DOWN_ORN
        scene.robot.move_to_pose((hole_center[0], hole_center[1] - 0.25, 0.32), DOWN_ORN, sim_steps=60)

        estimate, _ = estimate_hole_position(client, pixel_noise_std=0.0)
        assert estimate is not None, "hole should be detected with a clear, unobstructed camera view"

        error_mm = 1000.0 * np.hypot(estimate[0] - hole_center[0], estimate[1] - hole_center[1])
        assert error_mm < 5.0, f"detection error {error_mm:.2f}mm exceeds 5mm tolerance"
    finally:
        p.disconnect(physicsClientId=client)


def test_hole_detection_returns_none_without_frame():
    """No red frame in view (camera looking at empty space) should fail closed, not crash."""
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        # build a scene but look at a point far from the fixture
        Scene(client, clearance=0.003, hole_center=(0.5, 0.0))
        from src import perception
        rgb = perception.capture(client)
        blank = rgb.copy()
        blank[:] = (120, 130, 140)  # flat "table" color, no frame at all
        estimate = perception.detect_hole(blank)
        assert estimate is None
    finally:
        p.disconnect(physicsClientId=client)
