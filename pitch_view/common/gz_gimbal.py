"""Point the sim telephoto head at a player and keep that player in frame centre.

Open-loop aim uses the pitch position. Each new telephoto frame then measures
how far the colour blob sits from the image centre and trims yaw and pitch.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Beside the wide sideline camera (0, -13, 8). Sensor looks along link +X.
MOUNT = (2.2, -13.0, 8.0)
LOOK_Z = 1.0
SERVICE = "/world/soccer_pitch/set_pose"
MODEL = "tele_camera"
YAW_MODEL = "tele_yaw"
HFOV = 0.32
# Small steps only. A large gain plus the homography aim on the same
# frame makes the head hunt and then lose the player.
SERVO_GAIN = 0.55
DEADZONE = 0.008
# World aim is allowed to keep up with a running player. The image
# correction stays smaller so it recentres without hunting.
FOLLOW_STEP = 0.05
SERVO_STEP = 0.02
SERVO_LIMIT = 0.25


def template_cm_to_world(x_cm: float, y_cm: float, z: float = LOOK_Z):
    """SoccerPitchConfiguration centimetres → metres on the 18 m × 10.5 m pitch."""
    return (
        (float(x_cm) / 100.0 - 60.0) * 0.15,
        (35.0 - float(y_cm) / 100.0) * 0.15,
        float(z),
    )


def _quat(pitch: float, yaw: float):
    """Quaternion for yaw about Z then pitch about Y. Positive pitch looks down."""
    sp, cp = math.sin(pitch * 0.5), math.cos(pitch * 0.5)
    sy, cy = math.sin(yaw * 0.5), math.cos(yaw * 0.5)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


def aim_angles(target):
    dx = target[0] - MOUNT[0]
    dy = target[1] - MOUNT[1]
    dz = target[2] - MOUNT[2]
    horiz = math.hypot(dx, dy)
    if horiz < 1e-3:
        horiz = 1e-3
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, horiz)
    return pitch, yaw


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _approach(current, goal, max_step, wrap=False):
    delta = _wrap(goal - current) if wrap else (goal - current)
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return current + delta


def center_error(frame, prefer=None):
    """Yaw/pitch radians that would move one player onto the centre.

    ``prefer`` is the previous image point. The blob nearest that point is
    kept, so a second player walking through the frame does not steal the aim.
    Image x to the right is negative yaw. Image y downward is positive pitch.
    """
    from pitch_view.common.sim_players import detect_sim_players

    height, width = frame.shape[:2]
    boxes = detect_sim_players(frame)
    if not boxes:
        return None
    cx, cy = width * 0.5, height * 0.5
    anchor = prefer if prefer is not None else (cx, cy)
    best, best_d = None, 1e18
    for _cls, _score, xyxy in boxes:
        u = (float(xyxy[0]) + float(xyxy[2])) * 0.5
        v = (float(xyxy[1]) + float(xyxy[3])) * 0.5
        dist = (u - anchor[0]) ** 2 + (v - anchor[1]) ** 2
        if dist < best_d:
            best, best_d = (u, v), dist
    u, v = best
    fx = (width * 0.5) / math.tan(HFOV * 0.5)
    vfov = 2.0 * math.atan(math.tan(HFOV * 0.5) * height / width)
    fy = (height * 0.5) / math.tan(vfov * 0.5)
    yaw = -math.atan2(u - cx, fx)
    pitch = math.atan2(v - cy, fy)
    return pitch, yaw, (u, v)


class TeleGimbal:
    def __init__(self):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose
        import gz.transport13 as transport

        self._Pose = Pose
        self._Boolean = Boolean
        self._node = transport.Node()
        self._pitch = None
        self._yaw = None
        self._off_pitch = 0.0
        self._off_yaw = 0.0
        self._track = None
        self._blob = None

    def _send_one(self, name, pitch, yaw):
        qx, qy, qz, qw = _quat(pitch, yaw)
        req = self._Pose()
        req.name = name
        req.position.x, req.position.y, req.position.z = MOUNT
        req.orientation.x = qx
        req.orientation.y = qy
        req.orientation.z = qz
        req.orientation.w = qw
        ok, _ = self._node.request(SERVICE, req, self._Pose, self._Boolean, 80)
        return bool(ok)

    def _send(self, pitch, yaw):
        # Yaw stage stays level. Pitch stage is the camera, inside the fork.
        yaw_ok = self._send_one(YAW_MODEL, 0.0, yaw)
        cam_ok = self._send_one(MODEL, pitch, yaw)
        return yaw_ok and cam_ok

    def aim_world(self, target, frame=None, track_id=None) -> bool:
        """Follow the pitch point continuously, then nudge toward image centre."""
        goal_pitch, goal_yaw = aim_angles(target)
        if self._pitch is None or track_id != self._track:
            self._track = track_id
            self._blob = None
            self._off_pitch = 0.0
            self._off_yaw = 0.0
            self._pitch, self._yaw = goal_pitch, goal_yaw
            return self._send(self._pitch, self._yaw)
        self._pitch = _approach(self._pitch, goal_pitch, FOLLOW_STEP)
        self._yaw = _approach(self._yaw, goal_yaw, FOLLOW_STEP, wrap=True)
        if frame is not None and track_id is not None:
            err = center_error(frame, self._blob)
            if err is None:
                self._blob = None
                self._off_pitch = _approach(self._off_pitch, 0.0, SERVO_STEP)
                self._off_yaw = _approach(self._off_yaw, 0.0, SERVO_STEP)
            else:
                self._blob = err[2]
                want_p = 0.0 if abs(err[0]) <= DEADZONE else max(
                    -SERVO_LIMIT, min(SERVO_LIMIT, SERVO_GAIN * err[0]))
                want_y = 0.0 if abs(err[1]) <= DEADZONE else max(
                    -SERVO_LIMIT, min(SERVO_LIMIT, SERVO_GAIN * err[1]))
                self._off_pitch = _approach(self._off_pitch, want_p, SERVO_STEP)
                self._off_yaw = _approach(self._off_yaw, want_y, SERVO_STEP)
        return self._send(self._pitch + self._off_pitch, self._yaw + self._off_yaw)
