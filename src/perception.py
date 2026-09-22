"""Synthetic overhead camera + classical CV pipeline that localizes the hole opening.

The camera is fixed in the world frame (it is NOT re-centered on the hole), so the
hole's position must genuinely be recovered from the rendered image every trial.
"""
import numpy as np
import cv2
import pybullet as p

IMG_W, IMG_H = 480, 480
CAM_EYE = (0.5, 0.0, 1.0)
CAM_TARGET = (0.5, 0.0, 0.0)
CAM_UP = (1.0, 0.0, 0.0)
FOV_DEG = 55.0
NEAR, FAR = 0.1, 2.0

# HSV range that isolates the red frame fixture (see sim_env.FRAME_COLOR)
FRAME_HSV_LOW = np.array([0, 90, 60])
FRAME_HSV_HIGH = np.array([12, 255, 255])
FRAME_HSV_LOW2 = np.array([168, 90, 60])
FRAME_HSV_HIGH2 = np.array([180, 255, 255])


def _matrices():
    view = np.array(
        p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)
    ).reshape(4, 4, order="F")
    proj = np.array(
        p.computeProjectionMatrixFOV(FOV_DEG, 1.0, NEAR, FAR)
    ).reshape(4, 4, order="F")
    return view, proj


def capture(client):
    view, proj = _matrices()
    _, _, rgba, depth, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view.flatten(order="F"),
        projectionMatrix=proj.flatten(order="F"),
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    rgb = np.reshape(rgba, (IMG_H, IMG_W, 4))[:, :, :3].astype(np.uint8)
    return rgb


def pixel_to_world(px, py, table_z=0.0):
    """Unproject an image pixel onto the known table plane z = table_z."""
    view, proj = _matrices()
    vp = proj @ view
    inv_vp = np.linalg.inv(vp)

    x_ndc = 2.0 * (px + 0.5) / IMG_W - 1.0
    y_ndc = 1.0 - 2.0 * (py + 0.5) / IMG_H

    near_h = inv_vp @ np.array([x_ndc, y_ndc, -1.0, 1.0])
    far_h = inv_vp @ np.array([x_ndc, y_ndc, 1.0, 1.0])
    near_pt = near_h[:3] / near_h[3]
    far_pt = far_h[:3] / far_h[3]

    direction = far_pt - near_pt
    if abs(direction[2]) < 1e-9:
        return None
    t = (table_z - near_pt[2]) / direction[2]
    world = near_pt + t * direction
    return world[0], world[1]


def detect_hole(rgb, pixel_noise_std=0.0, rng=None):
    """Return (world_x, world_y) of the hole center, or None if not found."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, FRAME_HSV_LOW, FRAME_HSV_HIGH)
    mask2 = cv2.inRange(hsv, FRAME_HSV_LOW2, FRAME_HSV_HIGH2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return None
    hierarchy = hierarchy[0]

    outer_idx, outer_area = None, 0.0
    for i, c in enumerate(contours):
        if hierarchy[i][3] == -1:  # top-level contour (no parent)
            area = cv2.contourArea(c)
            if area > outer_area:
                outer_area = area
                outer_idx = i
    if outer_idx is None:
        return None

    child_idx = hierarchy[outer_idx][2]
    if child_idx == -1:
        return None  # frame detected but no hole visible inside it

    m = cv2.moments(contours[child_idx])
    if m["m00"] == 0:
        return None
    px = m["m10"] / m["m00"]
    py = m["m01"] / m["m00"]

    if pixel_noise_std > 0.0:
        rng = rng or np.random
        px += rng.normal(0.0, pixel_noise_std)
        py += rng.normal(0.0, pixel_noise_std)

    return pixel_to_world(px, py)


def estimate_hole_position(client, pixel_noise_std=0.0, rng=None):
    rgb = capture(client)
    return detect_hole(rgb, pixel_noise_std=pixel_noise_std, rng=rng), rgb
