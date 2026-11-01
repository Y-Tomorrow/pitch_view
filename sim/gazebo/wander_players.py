#!/usr/bin/env python3
"""Smooth wander for each player inside the pitch.

Each player keeps a heading and a steady speed, and only turns gradually.
Yaw of the body stays fixed so the back number keeps facing the sideline camera.
The pitch is 18 m x 10.5 m. A margin keeps their bodies inside the lines.
"""

import json
import math
import os
import random
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

HERE = os.path.dirname(os.path.abspath(__file__))
POSES = os.path.join(HERE, "player_poses.json")
MOTION_FLAG = os.path.join(HERE, "player_motion_enabled")
SERVICE = "/world/soccer_pitch/set_pose"
HALF_L = 8.6
HALF_W = 4.8
DT = 0.05
MAX_TURN = 1.1  # rad/s
WALL = 1.3


def quat_z(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class _Walker:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.heading = random.uniform(-math.pi, math.pi)
        self.speed = random.uniform(0.7, 1.5)

    def step(self, dt):
        inward_x = inward_y = 0.0
        if self.x > HALF_L - WALL:
            inward_x -= 1.0
        elif self.x < -HALF_L + WALL:
            inward_x += 1.0
        if self.y > HALF_W - WALL:
            inward_y -= 1.0
        elif self.y < -HALF_W + WALL:
            inward_y += 1.0
        if inward_x or inward_y:
            desired = math.atan2(inward_y, inward_x)
        else:
            desired = self.heading + random.uniform(-0.35, 0.35) * dt
        turn = max(-MAX_TURN * dt, min(MAX_TURN * dt, _wrap(desired - self.heading)))
        self.heading = _wrap(self.heading + turn)
        self.x = min(HALF_L, max(-HALF_L, self.x + math.cos(self.heading) * self.speed * dt))
        self.y = min(HALF_W, max(-HALF_W, self.y + math.sin(self.heading) * self.speed * dt))


def _motion_enabled():
    try:
        with open(MOTION_FLAG, encoding="utf-8") as f:
            return f.read().strip() not in ("0", "off", "false")
    except OSError:
        return True


def main():
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose
    import gz.transport13 as transport

    with open(POSES, encoding="utf-8") as f:
        homes = json.load(f)
    state = {p["name"]: _Walker(p["x"], p["y"]) for p in homes}
    meta = {p["name"]: p for p in homes}
    node = transport.Node()
    deadline = time.time() + 45
    while time.time() < deadline:
        first = homes[0]
        qz, qw = quat_z(first["yaw"])[2:]
        req = Pose()
        req.name = first["name"]
        req.position.x = first["x"]
        req.position.y = first["y"]
        req.position.z = first["z"]
        req.orientation.z = qz
        req.orientation.w = qw
        ok, _ = node.request(SERVICE, req, Pose, Boolean, 300)
        if ok:
            break
        time.sleep(0.5)

    while True:
        if not _motion_enabled():
            time.sleep(0.2)
            continue
        t0 = time.perf_counter()
        for name, walker in state.items():
            home = meta[name]
            walker.step(DT)
            qz, qw = quat_z(home["yaw"])[2:]
            req = Pose()
            req.name = name
            req.position.x = walker.x
            req.position.y = walker.y
            req.position.z = home["z"]
            req.orientation.z = qz
            req.orientation.w = qw
            node.request(SERVICE, req, Pose, Boolean, 30)
        remain = DT - (time.perf_counter() - t0)
        if remain > 0:
            time.sleep(remain)


if __name__ == "__main__":
    main()
