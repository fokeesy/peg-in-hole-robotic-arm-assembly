"""Camera mounted on the gripper's wrist, looking down its approach axis.
Shared by interactive.py (manual playground) and any GUI trial run, so the
live preview updates during every phase of a run - grasp, transit, spiral
search, insertion - not just when a script happens to remember to call it.
"""
import numpy as np
import pybullet as p

LOCAL_EYE = (0.0, 0.0, 0.02)
LOCAL_TARGET = (0.0, 0.0, 0.4)
LOCAL_UP = (1.0, 0.0, 0.0)


def render(client, robot, width=200, height=200):
    ee_pos, ee_orn = robot.ee_pose()
    eye, _ = p.multiplyTransforms(ee_pos.tolist(), ee_orn.tolist(), LOCAL_EYE, [0, 0, 0, 1])
    target, _ = p.multiplyTransforms(ee_pos.tolist(), ee_orn.tolist(), LOCAL_TARGET, [0, 0, 0, 1])
    up_point, _ = p.multiplyTransforms(ee_pos.tolist(), ee_orn.tolist(), LOCAL_UP, [0, 0, 0, 1])
    up = np.array(up_point) - np.array(eye)

    view = p.computeViewMatrix(eye, target, up.tolist())
    proj = p.computeProjectionMatrixFOV(70.0, 1.0, 0.02, 2.0)
    p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj,
                      renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=client)
