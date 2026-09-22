"""Panda arm wrapper: joint discovery, inverse kinematics, smooth Cartesian motion."""
import numpy as np
import pybullet as p

from .sim_step import step as _step


class PandaRobot:
    def __init__(self, client, base_position=(0.0, 0.0, 0.0)):
        self.client = client
        self.id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=base_position,
            useFixedBase=True,
            physicsClientId=self.client,
        )

        num_joints = p.getNumJoints(self.id, physicsClientId=self.client)
        self.movable_joints = []
        self.arm_joints = []
        self.finger_joints = []
        self.ee_link = None

        for j in range(num_joints):
            info = p.getJointInfo(self.id, j, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            jtype = info[2]
            if jtype != p.JOINT_FIXED:
                self.movable_joints.append(j)
                if "finger" in name:
                    self.finger_joints.append(j)
                else:
                    self.arm_joints.append(j)
            if name == "panda_hand_joint":
                self.ee_link = j

        if self.ee_link is None:
            self.ee_link = num_joints - 1

        lower, upper, ranges, rest = [], [], [], []
        arm_rest = [0.0, -0.35, 0.0, -2.35, 0.0, 2.0, 0.78]
        for j in self.movable_joints:
            info = p.getJointInfo(self.id, j, physicsClientId=self.client)
            lo, hi = info[8], info[9]
            lower.append(lo)
            upper.append(hi)
            ranges.append(hi - lo)
            if j in self.finger_joints:
                rest.append(0.02)
            else:
                idx = self.arm_joints.index(j)
                rest.append(arm_rest[idx] if idx < len(arm_rest) else 0.0)

        self.lower_limits = lower
        self.upper_limits = upper
        self.joint_ranges = ranges
        self.rest_poses = rest

        for idx, j in enumerate(self.arm_joints):
            p.resetJointState(self.id, j, self.rest_poses[idx], physicsClientId=self.client)
        for j in self.finger_joints:
            p.resetJointState(self.id, j, 0.02, physicsClientId=self.client)
        self.set_gripper(0.02)

    def solve_ik(self, target_pos, target_orn, warm_start=False):
        """`warm_start`: bias the (redundant, 7-DOF) solution toward the robot's
        CURRENT joint configuration instead of the fixed rest pose. Without this,
        repeated small nudges each re-solve against the fixed rest pose and can
        jump to a different elbow configuration every call, producing jerky motion
        instead of a smooth local correction."""
        rest = self.rest_poses
        if warm_start:
            current_arm = self.get_arm_positions()
            rest = list(self.rest_poses)
            for j in self.arm_joints:
                idx = self.movable_joints.index(j)
                rest[idx] = current_arm[self.arm_joints.index(j)]

        solution = p.calculateInverseKinematics(
            self.id,
            self.ee_link,
            target_pos,
            target_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=rest,
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        arm_targets = []
        for j in self.arm_joints:
            idx = self.movable_joints.index(j)
            arm_targets.append(solution[idx])
        return arm_targets

    def get_arm_positions(self):
        return [
            p.getJointState(self.id, j, physicsClientId=self.client)[0]
            for j in self.arm_joints
        ]

    def set_gripper(self, half_width):
        for j in self.finger_joints:
            p.setJointMotorControl2(
                self.id, j, p.POSITION_CONTROL, targetPosition=half_width,
                force=20.0, physicsClientId=self.client,
            )

    def _apply_arm_targets(self, targets, max_force=87.0, position_gain=0.05):
        for j, target in zip(self.arm_joints, targets):
            p.setJointMotorControl2(
                self.id, j, p.POSITION_CONTROL, targetPosition=target,
                force=max_force, positionGain=position_gain, velocityGain=1.0,
                physicsClientId=self.client,
            )

    def move_to_pose(self, target_pos, target_orn, sim_steps=120, settle_steps=120):
        """Interpolate in joint space from the current pose to the IK solution.

        Deliberately does NOT warm-start from the current configuration: for
        this arm/target family, biasing toward the fixed default rest pose
        consistently finds a collision-free elbow configuration, whereas
        warm-starting from wherever the previous macro move happened to land
        can drift into a solution that clips the fixture (see README)."""
        start = np.array(self.get_arm_positions())
        goal = np.array(self.solve_ik(target_pos, target_orn, warm_start=False))
        for i in range(1, sim_steps + 1):
            alpha = i / sim_steps
            waypoint = start + alpha * (goal - start)
            self._apply_arm_targets(waypoint, position_gain=0.2, max_force=140.0)
            _step(self.client)
        for _ in range(settle_steps):
            self._apply_arm_targets(goal, position_gain=0.2, max_force=140.0)
            _step(self.client)
        return goal

    def track_to(self, target_pos, target_orn, sim_steps=10, kp_lin=10.0, kp_ang=6.0,
                 max_lin_vel=0.06, max_ang_vel=1.2, max_force=140.0):
        """Closed-loop resolved-rate (Jacobian pseudo-inverse) servo toward an
        absolute Cartesian target, re-evaluating the position/orientation ERROR
        every physics step and driving the arm with actual joint VELOCITY
        control (not position control toward a barely-shifted target - that
        collapses into gravity-sag noise once the per-step delta is smaller
        than the position controller's own steady-state gravity error)."""
        target_pos = np.array(target_pos, dtype=float)
        for _ in range(sim_steps):
            q = [p.getJointState(self.id, j, physicsClientId=self.client)[0]
                 for j in self.movable_joints]
            zeros = [0.0] * len(self.movable_joints)
            jac_t, jac_r = p.calculateJacobian(
                self.id, self.ee_link, [0.0, 0.0, 0.0], q, zeros, zeros,
                physicsClientId=self.client,
            )
            jac_t = np.array(jac_t)
            jac_r = np.array(jac_r)

            cur_pos, cur_orn = self.ee_pose()
            pos_err = target_pos - cur_pos
            lin_vel = np.clip(kp_lin * pos_err, -max_lin_vel, max_lin_vel)

            _, inv_orn = p.invertTransform([0, 0, 0], cur_orn.tolist())
            _, q_diff = p.multiplyTransforms([0, 0, 0], list(target_orn), [0, 0, 0], inv_orn)
            axis, angle = p.getAxisAngleFromQuaternion(q_diff)
            if angle > np.pi:
                angle -= 2 * np.pi
            ang_vel = np.clip(kp_ang * angle * np.array(axis), -max_ang_vel, max_ang_vel)

            task_vel = np.concatenate([lin_vel, ang_vel])
            jac_full = np.vstack([jac_t, jac_r])
            dq = np.linalg.pinv(jac_full) @ task_vel

            for j in self.arm_joints:
                idx = self.movable_joints.index(j)
                p.setJointMotorControl2(
                    self.id, j, p.VELOCITY_CONTROL, targetVelocity=float(dq[idx]),
                    force=max_force, physicsClientId=self.client,
                )
            _step(self.client)

        # leave the arm holding its final position (position control) once the
        # servo loop ends, rather than continuing to coast at the last velocity
        self._apply_arm_targets(self.get_arm_positions(), position_gain=0.5, max_force=max_force)
        ee_state = p.getLinkState(self.id, self.ee_link, physicsClientId=self.client)
        return np.array(ee_state[4])

    def ee_pose(self):
        state = p.getLinkState(self.id, self.ee_link, computeForwardKinematics=1,
                                physicsClientId=self.client)
        return np.array(state[4]), np.array(state[5])

    def attach_peg(self, peg_id):
        """Rigidly attach a body (already positioned at the desired grasp pose) to the gripper."""
        ee_pos, ee_orn = self.ee_pose()
        peg_pos, peg_orn = p.getBasePositionAndOrientation(peg_id, physicsClientId=self.client)
        inv_pos, inv_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, peg_pos, peg_orn)
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=peg_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=rel_pos,
            parentFrameOrientation=rel_orn,
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0, 1),
            physicsClientId=self.client,
        )
        p.changeConstraint(constraint_id, maxForce=5000.0, physicsClientId=self.client)
        return constraint_id
