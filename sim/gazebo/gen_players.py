#!/usr/bin/env python3
"""Static players as colored primitives. Gazebo Sim paints meshes white."""

import os

OUT = os.path.join(os.path.dirname(__file__), "models", "soccer_players")

SEGS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abged",
    "3": "abgcd",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgecd",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}

# (name, number, x, y, yaw, r, g, b)
# yaw = pi/2: chest faces +Y, back (and the number) faces the camera on the -Y side.
_FACE_CAMERA = 1.5707963
PLAYERS = [
    ("red_1", "1", -7.2, 0.0, _FACE_CAMERA, 0.75, 0.12, 0.12),
    ("red_4", "4", -4.2, -2.4, _FACE_CAMERA, 0.75, 0.12, 0.12),
    ("red_5", "5", -4.2, 2.4, _FACE_CAMERA, 0.75, 0.12, 0.12),
    ("red_7", "7", -1.6, -1.5, _FACE_CAMERA, 0.75, 0.12, 0.12),
    ("red_9", "9", -1.2, 1.8, _FACE_CAMERA, 0.75, 0.12, 0.12),
    ("red_10", "10", 1.4, -0.4, _FACE_CAMERA, 0.75, 0.12, 0.12),
    ("blue_1", "1", 7.2, 0.0, _FACE_CAMERA, 0.12, 0.28, 0.72),
    ("blue_3", "3", 4.2, 2.3, _FACE_CAMERA, 0.12, 0.28, 0.72),
    ("blue_6", "6", 4.0, -2.3, _FACE_CAMERA, 0.12, 0.28, 0.72),
    ("blue_8", "8", 1.5, 1.6, _FACE_CAMERA, 0.12, 0.28, 0.72),
    ("blue_11", "11", -0.6, -1.6, _FACE_CAMERA, 0.12, 0.28, 0.72),
]


def mat(r, g, b):
    return (
        "<material>"
        f"<ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>"
        f"<diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>"
        "<specular>0.04 0.04 0.04 1</specular>"
        "</material>"
    )


def visual(name, x, y, z, yaw, geom, color):
    r, g, b = color
    return (
        f'<visual name="{name}">'
        f"<pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>"
        "<cast_shadows>false</cast_shadows>"
        f"<geometry>{geom}</geometry>"
        f"{mat(r, g, b)}"
        "</visual>"
    )


def number_visuals(prefix, number, color):
    height, width, thick, depth = 0.16, 0.09, 0.016, 0.008
    layout = {
        "a": (0.0, 0.0, height / 2, depth, width, thick),
        "d": (0.0, 0.0, -height / 2, depth, width, thick),
        "g": (0.0, 0.0, 0.0, depth, width, thick),
        "b": (0.0, width / 2, height / 4, depth, thick, height / 2),
        "c": (0.0, width / 2, -height / 4, depth, thick, height / 2),
        "f": (0.0, -width / 2, height / 4, depth, thick, height / 2),
        "e": (0.0, -width / 2, -height / 4, depth, thick, height / 2),
    }
    digits = list(number)
    span = 0.11
    origin = span * (len(digits) - 1) / 2.0
    chunks = []
    for on_back in (False, True):
        sign = -1.0 if on_back else 1.0
        face = -0.10 if on_back else 0.10
        tag = "b" if on_back else "f"
        for i, ch in enumerate(digits):
            cy = origin - i * span
            for seg in SEGS[ch]:
                dx, dy, dz, sx, sy, sz = layout[seg]
                chunks.append(
                    visual(
                        f"{prefix}_{tag}{i}_{seg}",
                        face + dx,
                        sign * (cy + dy),
                        0.80 + dz,
                        0.0,
                        f"<box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>",
                        color,
                    )
                )
    return chunks


def player_model(name, number, x, y, yaw, r, g, b):
    """One model per player so each can be moved on its own."""
    shirt = (r, g, b)
    shorts = (r * 0.55, g * 0.55, b * 0.55)
    skin = (0.86, 0.72, 0.58)
    white = (0.96, 0.96, 0.96)
    parts = [
        visual(f"{name}_leg_l", 0, 0.07, 0.28, 0, "<box><size>0.09 0.09 0.52</size></box>", shorts),
        visual(f"{name}_leg_r", 0, -0.07, 0.28, 0, "<box><size>0.09 0.09 0.52</size></box>", shorts),
        visual(f"{name}_torso", 0, 0, 0.78, 0, "<box><size>0.18 0.34 0.42</size></box>", shirt),
        visual(f"{name}_arm_l", 0, 0.24, 0.78, 0, "<box><size>0.08 0.08 0.36</size></box>", shirt),
        visual(f"{name}_arm_r", 0, -0.24, 0.78, 0, "<box><size>0.08 0.08 0.36</size></box>", shirt),
        visual(f"{name}_head", 0, 0, 1.12, 0, "<sphere><radius>0.11</radius></sphere>", skin),
        *number_visuals(name, number, white),
    ]
    body = "\n      ".join(parts)
    model_name = f"sim_{name}"
    return model_name, f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{model_name}">
    <static>false</static>
    <pose>{x:.3f} {y:.3f} 0.03 0 0 {yaw:.4f}</pose>
    <link name="body">
      <kinematic>true</kinematic>
      <gravity>false</gravity>
      <inertial>
        <mass>1</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="body">
        <pose>0 0 0.6 0 0 0</pose>
        <geometry><box><size>0.3 0.4 1.2</size></box></geometry>
      </collision>
      {body}
    </link>
  </model>
</sdf>
"""


def main():
    import json
    models = os.path.join(os.path.dirname(__file__), "models")
    poses = []
    includes = []
    for row in PLAYERS:
        model_name, sdf = player_model(*row)
        folder = os.path.join(models, model_name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "model.sdf"), "w", encoding="utf-8") as f:
            f.write(sdf)
        with open(os.path.join(folder, "model.config"), "w", encoding="utf-8") as f:
            f.write(
                f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>One colored player. Moved by wander_players.py.</description>
</model>
"""
            )
        name, _number, x, y, yaw = row[:5]
        poses.append({"name": model_name, "x": x, "y": y, "z": 0.03, "yaw": yaw})
        includes.append(
            f"""    <include>
      <uri>model://{model_name}</uri>
      <name>{model_name}</name>
    </include>"""
        )
    with open(os.path.join(os.path.dirname(__file__), "player_poses.json"), "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    world = os.path.join(os.path.dirname(__file__), "worlds", "soccer_pitch.sdf")
    text = open(world, encoding="utf-8").read()
    begin, end = "<!-- PLAYERS_BEGIN -->", "<!-- PLAYERS_END -->"
    block = begin + "\n" + "\n".join(includes) + "\n    " + end
    if begin not in text:
        text = text.replace(
            """    <include>
      <uri>model://soccer_players</uri>
      <name>players</name>
      <pose>0 0 0.03 0 0 0</pose>
    </include>""",
            "    " + block,
        )
    else:
        pre, rest = text.split(begin, 1)
        _old, post = rest.split(end, 1)
        text = pre + block + post
    with open(world, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {len(PLAYERS)} movable players")


if __name__ == "__main__":
    main()
