#!/usr/bin/env python3
"""旧入口。请改用 scripts/run_pitch_view.py。"""
import os
import runpy
import sys

sys.stderr.write("scripts/run_homography.py 已改名为 scripts/run_pitch_view.py\n")
target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_pitch_view.py")
runpy.run_path(target, run_name="__main__")
