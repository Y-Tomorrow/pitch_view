#!/usr/bin/env python3
"""Pitch lines matching SoccerPitchConfiguration, drawn at 18 m x 10.5 m.

The absolute size is 0.15 of the 120 m x 70 m template. Every line keeps the
template's relative position, so a homography onto that template is consistent.
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pitch_view.configs.soccer import SoccerPitchConfiguration

ROOT = os.path.dirname(__file__)
MODEL = os.path.join(ROOT, "models", "robocup_3Dsim_field")
MESH = os.path.join(MODEL, "meshes", "lines.obj")

# Template length 120 m -> 18 m. Width follows 70/120.
SCALE = 18.0 / 120.0
LINE_W = 0.08
LINE_H = 0.012
# Link pose is z=0.01 and the turf top is 0.02 above the link, so lines sit
# just above the grass in the link frame.
Z = 0.028


def to_world(x_cm: float, y_cm: float):
    """Template centimetres -> metres, origin at the centre spot."""
    cfg = SoccerPitchConfiguration()
    x = (x_cm / 100.0 - cfg.length / 200.0) * SCALE
    y = (cfg.width / 200.0 - y_cm / 100.0) * SCALE
    return x, y


class Mesh:
    def __init__(self):
        self.verts = []
        self.faces = []

    def add_box(self, cx, cy, cz, sx, sy, sz, yaw):
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        c, s = math.cos(yaw), math.sin(yaw)
        base = len(self.verts)
        for dz in (-hz, hz):
            for dy in (-hy, hy):
                for dx in (-hx, hx):
                    self.verts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz))
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        for a, b, c, d in quads:
            self.faces.append((base + a, base + b, base + c))
            self.faces.append((base + a, base + c, base + d))

    def add_segment(self, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        self.add_box((x1 + x2) / 2, (y1 + y2) / 2, Z, length + LINE_W, LINE_W, LINE_H, math.atan2(dy, dx))

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for x, y, z in self.verts:
                f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
            for a, b, c in self.faces:
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def add_arc(mesh, cx, cy, radius, keep):
    n = 48
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        if keep(x, y):
            pts.append((x, y))
        elif len(pts) > 1:
            for p, q in zip(pts, pts[1:]):
                mesh.add_segment(*p, *q)
            pts = []
    if len(pts) > 1:
        for p, q in zip(pts, pts[1:]):
            mesh.add_segment(*p, *q)


def main():
    cfg = SoccerPitchConfiguration()
    mesh = Mesh()
    verts = [to_world(x, y) for x, y in cfg.vertices]
    for a, b in cfg.edges:
        mesh.add_segment(*verts[a - 1], *verts[b - 1])

    cx, cy = to_world(cfg.length / 2, cfg.width / 2)
    radius = cfg.centre_circle_radius / 100.0 * SCALE
    add_arc(mesh, cx, cy, radius, lambda _x, _y: True)
    mesh.add_box(cx, cy, Z, LINE_W * 1.4, LINE_W * 1.4, LINE_H, 0)

    for spot_cm in (cfg.penalty_spot_distance, cfg.length - cfg.penalty_spot_distance):
        sx, sy = to_world(spot_cm, cfg.width / 2)
        mesh.add_box(sx, sy, Z, LINE_W * 1.4, LINE_W * 1.4, LINE_H, 0)
        front = to_world(cfg.penalty_box_length, 0)[0]
        far = to_world(cfg.length - cfg.penalty_box_length, 0)[0]
        if spot_cm < cfg.length / 2:
            add_arc(mesh, sx, sy, radius, lambda x, y, front=front: x >= front)
        else:
            add_arc(mesh, sx, sy, radius, lambda x, y, far=far: x <= far)

    mesh.write(MESH)
    length = cfg.length / 100.0 * SCALE
    width = cfg.width / 100.0 * SCALE

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="robocup_3d_field">
    <static>true</static>
    <link name="field">
      <pose>0 0 0.01 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>{length:.4f} {width:.4f}</size>
          </plane>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.2</mu>
              <mu2>1.2</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="turf">
        <pose>0 0 0.01 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <box><size>{length:.4f} {width:.4f} 0.02</size></box>
        </geometry>
        <material>
          <ambient>0.20 0.55 0.16 1</ambient>
          <diffuse>0.20 0.55 0.16 1</diffuse>
          <specular>0.02 0.04 0.02 1</specular>
        </material>
      </visual>
      <visual name="lines">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>model://robocup_3Dsim_field/meshes/lines.obj</uri>
          </mesh>
        </geometry>
        <material>
          <ambient>0.95 0.95 0.95 1</ambient>
          <diffuse>0.95 0.95 0.95 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    with open(os.path.join(MODEL, "model.sdf"), "w", encoding="utf-8") as f:
        f.write(sdf)
    print(f"pitch {length:.2f} m x {width:.2f} m, lines {len(mesh.faces)} tris")


if __name__ == "__main__":
    main()
