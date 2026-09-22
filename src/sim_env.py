"""Scene construction: ground, square peg, square-hole frame fixture, and cameras."""
import numpy as np
import pybullet as p
import pybullet_data

from .robot import PandaRobot

FRAME_COLOR = (0.85, 0.25, 0.15, 1.0)   # distinct red frame, segmented by the vision pipeline
TABLE_COLOR = (0.55, 0.58, 0.62, 1.0)   # grey table top, visible through the hole opening
PEG_COLOR = (0.15, 0.45, 0.85, 1.0)     # blue peg


class Scene:
    def __init__(self, client, peg_side=0.03, peg_len=0.06, clearance=0.003,
                 wall_thickness=0.045, frame_height=0.03, hole_center=(0.5, 0.0)):
        self.client = client
        self.peg_side = peg_side
        self.peg_len = peg_len
        self.clearance = clearance
        self.wall_thickness = wall_thickness
        self.frame_height = frame_height
        self.hole_center = np.array(hole_center, dtype=float)
        self.table_z = 0.0

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeVisualShape(self.ground_id, -1, rgbaColor=TABLE_COLOR, physicsClientId=self.client)
        p.changeDynamics(self.ground_id, -1, lateralFriction=0.6, physicsClientId=self.client)

        self.robot = PandaRobot(self.client, base_position=(0.0, 0.0, 0.0))

        self.frame_id = self._build_frame()
        self.peg_id = self._build_peg()

        # the fingers are left open (the peg is rigidly attached, not actually
        # grasped between them - see trial.py) and would otherwise physically
        # collide with the ground/fixture at low approach heights, silently
        # blocking descent with no signal other than "the arm won't go lower"
        frame_links = [-1] + list(range(p.getNumJoints(self.frame_id, physicsClientId=self.client)))
        for link in self.robot.finger_joints:
            p.setCollisionFilterPair(self.robot.id, self.ground_id, link, -1, 0, physicsClientId=self.client)
            for flink in frame_links:
                p.setCollisionFilterPair(self.robot.id, self.frame_id, link, flink, 0, physicsClientId=self.client)

    # ------------------------------------------------------------------ #
    def _ring_specs(self, gap_half, z_bottom, height):
        """4 wall boxes forming a square ring with a square gap in the middle."""
        hx, hy = self.hole_center
        wt = self.wall_thickness
        outer_half = gap_half + wt
        z_center = z_bottom + height / 2.0
        return [
            (hx, hy + gap_half + wt / 2.0, outer_half, wt / 2.0, z_center, height),
            (hx, hy - gap_half - wt / 2.0, outer_half, wt / 2.0, z_center, height),
            (hx - gap_half - wt / 2.0, hy, wt / 2.0, gap_half, z_center, height),
            (hx + gap_half + wt / 2.0, hy, wt / 2.0, gap_half, z_center, height),
        ]

    def _build_frame(self):
        """Two stacked rings: a wider chamfer-style lead-in on top of the true
        tight-tolerance socket, mirroring the lead-in chamfer every real precision
        assembly fixture uses to avoid the peg wedging on entry."""
        gap_half = self.peg_side / 2.0 + self.clearance
        lead_in_extra = max(0.006, 2.0 * self.clearance)
        lead_in_height = min(0.012, self.frame_height * 0.4)
        precision_height = self.frame_height - lead_in_height

        specs = []
        specs += self._ring_specs(gap_half + lead_in_extra, self.table_z + precision_height, lead_in_height)
        specs += self._ring_specs(gap_half, self.table_z, precision_height)

        link_masses, link_col, link_vis = [], [], []
        link_pos, link_orn = [], []
        for cx, cy, hx_, hy_, cz, height in specs:
            half_extents = [hx_, hy_, height / 2.0]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=FRAME_COLOR,
                                       physicsClientId=self.client)
            link_masses.append(0.0)
            link_col.append(col)
            link_vis.append(vis)
            link_pos.append([cx, cy, cz])
            link_orn.append([0, 0, 0, 1])

        n = len(specs)
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.001, 0.001, 0.001],
                                           physicsClientId=self.client)
        frame_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=base_col,
            basePosition=[0, 0, 0],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=link_orn,
            linkInertialFramePositions=[[0, 0, 0]] * n,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
            linkParentIndices=[0] * n,
            linkJointTypes=[p.JOINT_FIXED] * n,
            linkJointAxis=[[0, 0, 0]] * n,
            physicsClientId=self.client,
        )
        p.changeDynamics(frame_id, -1, lateralFriction=0.3, physicsClientId=self.client)
        for i in range(n):
            p.changeDynamics(frame_id, i, lateralFriction=0.3, physicsClientId=self.client)
        return frame_id

    def _build_peg(self):
        half = self.peg_side / 2.0
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, self.peg_len / 2.0],
                                      physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, self.peg_len / 2.0],
                                   rgbaColor=PEG_COLOR, physicsClientId=self.client)
        start_pos = [self.hole_center[0], self.hole_center[1] - 0.25, 0.35]
        peg_id = p.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=start_pos,
            physicsClientId=self.client,
        )
        p.changeDynamics(peg_id, -1, lateralFriction=0.5, spinningFriction=0.001,
                          rollingFriction=0.001, physicsClientId=self.client)
        return peg_id

    # ------------------------------------------------------------------ #
    def peg_contacts_frame(self):
        return p.getContactPoints(self.peg_id, self.frame_id, physicsClientId=self.client)

    def peg_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.client)
        return np.array(pos), np.array(orn)

    def insertion_depth(self):
        """How far the peg tip has sunk below the top of the frame (top of hole opening)."""
        pos, _ = self.peg_state()
        peg_tip_z = pos[2] - self.peg_len / 2.0
        top_of_frame = self.table_z + self.frame_height
        return max(0.0, top_of_frame - peg_tip_z)

    def peg_xy_error(self):
        """Distance from the peg's center to the TRUE hole center - needed
        because depth+tilt alone can't distinguish "seated in the hole" from
        "the estimate was so far off the peg free-fell onto open ground
        nowhere near the fixture" (nothing there to stop its descent either
        way looks identical in depth/tilt terms)."""
        pos, _ = self.peg_state()
        return float(np.hypot(pos[0] - self.hole_center[0], pos[1] - self.hole_center[1]))

    def peg_tilt_deg(self):
        """Deviation from vertical. The peg is a symmetric box grasped pointing
        down (a 180-degree flip about X), so both +z and -z alignment count as
        upright; only genuine tipping-over increases this angle."""
        _, orn = self.peg_state()
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        local_z = rot[:, 2]
        cos_angle = np.clip(abs(np.dot(local_z, [0, 0, 1])), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
