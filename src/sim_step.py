"""Thin wrapper around p.stepSimulation that adds:
  - a real-time pause when a client has been marked for real-time playback
  - an automatically-throttled wrist-camera render when a client has been
    marked for it (so PyBullet's live preview pane updates during every
    phase of a GUI run - grasp, transit, search, insertion - without every
    call site needing to remember to render a frame itself)
Headless (non-GUI) clients do neither, and run at full speed.
"""
import time
import pybullet as p

from . import wrist_camera

_REALTIME_CLIENTS = set()
_CAMERA_FEEDS = {}  # client -> {"robot", "every", "counter"}
_DT = 1.0 / 240.0


def enable_realtime(client):
    _REALTIME_CLIENTS.add(client)


def disable_realtime(client):
    _REALTIME_CLIENTS.discard(client)


def enable_wrist_camera(client, robot, every_n_steps=12):
    _CAMERA_FEEDS[client] = {"robot": robot, "every": every_n_steps, "counter": 0}


def disable_wrist_camera(client):
    _CAMERA_FEEDS.pop(client, None)


def step(client):
    p.stepSimulation(physicsClientId=client)
    if client in _REALTIME_CLIENTS:
        time.sleep(_DT)
    feed = _CAMERA_FEEDS.get(client)
    if feed is not None:
        feed["counter"] += 1
        if feed["counter"] % feed["every"] == 0:
            wrist_camera.render(client, feed["robot"])
