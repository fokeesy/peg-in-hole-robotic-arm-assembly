"""Visual polish for the PyBullet debug GUI.

PyBullet's built-in window is a physics-debug renderer, not a product UI - it
has a real ceiling on how modern it can look (flat shading, fixed widget
chrome). This applies what's actually available: shadows for real depth
instead of flat cartoon shading, a dark background instead of the default
grey/blue, and turning off preview panels we don't use so the one we do
use (the wrist camera) isn't lost in clutter.
"""
import pybullet as p

BACKGROUND_RGB = (0.05, 0.06, 0.08)

CONNECT_OPTIONS = (
    f"--background_color_red={BACKGROUND_RGB[0]} "
    f"--background_color_green={BACKGROUND_RGB[1]} "
    f"--background_color_blue={BACKGROUND_RGB[2]}"
)


def apply(client):
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=client)
