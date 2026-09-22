"""Manual playground: opens a live PyBullet window with the Panda arm and the
peg-in-hole fixture, and gives you two ways to move the arm yourself:

  1. Drag the on-screen sliders (one per joint) to pose it directly.
  2. Click-and-drag any link of the arm with the mouse in the 3D view -
     PyBullet's GUI supports this natively, no extra code needed.

A camera mounted on the wrist streams into PyBullet's built-in preview pane
(top-left of the window) so you can see what the gripper "sees" as you move
it.

    python interactive.py
"""
import pybullet as p

from src.sim_env import Scene
from src import gui_style
from src.sim_step import step as _step, enable_wrist_camera, enable_realtime

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]


def main():
    client = p.connect(p.GUI, options=gui_style.CONNECT_OPTIONS)
    gui_style.apply(client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)
    p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=40, cameraPitch=-35,
                                  cameraTargetPosition=[0.5, 0.0, 0.1], physicsClientId=client)

    scene = Scene(client, clearance=0.004, hole_center=(0.5, 0.0))
    robot = scene.robot
    enable_realtime(client)
    enable_wrist_camera(client, robot)

    sliders = []
    for name, j in zip(JOINT_NAMES, robot.arm_joints):
        idx = robot.movable_joints.index(j)
        lo, hi = robot.lower_limits[idx], robot.upper_limits[idx]
        start = p.getJointState(robot.id, j, physicsClientId=client)[0]
        sliders.append(p.addUserDebugParameter(name, lo, hi, start, physicsClientId=client))

    finger_lo, finger_hi = 0.0, 0.04
    gripper_slider = p.addUserDebugParameter("gripper", finger_lo, finger_hi, 0.02,
                                              physicsClientId=client)

    print(__doc__)
    print("Drag the sliders in the window's side panel, or click-and-drag the arm directly.")
    print("Close the window (or Ctrl+C here) to exit.\n")

    try:
        while p.isConnected(client):
            targets = [p.readUserDebugParameter(s, physicsClientId=client) for s in sliders]
            for j, target in zip(robot.arm_joints, targets):
                p.setJointMotorControl2(robot.id, j, p.POSITION_CONTROL, targetPosition=target,
                                         force=140.0, positionGain=0.2, physicsClientId=client)
            gripper_target = p.readUserDebugParameter(gripper_slider, physicsClientId=client)
            robot.set_gripper(gripper_target)

            _step(client)  # real-time pacing + wrist-camera refresh handled here
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)


if __name__ == "__main__":
    main()
