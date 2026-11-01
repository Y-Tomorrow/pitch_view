#!/usr/bin/env bash
# Start the Gazebo Sim pitch. Requires gz-sim (Harmonic or newer): `gz sim`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_SIM_RESOURCE_PATH="${HERE}/models${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

# This laptop exposes both the AMD iGPU and the RTX 4060 through WSLg.
# Mesa's default D3D12 adapter is the AMD GPU. Pin Gazebo's OpenGL to the NVIDIA one.
unset LIBGL_ALWAYS_SOFTWARE
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

if ! command -v gz >/dev/null 2>&1; then
  echo "未找到 gz。Ubuntu 22.04 可安装 Gazebo Harmonic：" >&2
  echo "  sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg" >&2
  echo "  echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable \$(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list" >&2
  echo "  sudo apt-get update && sudo apt-get install -y gz-harmonic" >&2
  exit 1
fi

python3 "${HERE}/gen_players.py" >/dev/null

# A previous Ctrl+C used to leave wander_players.py alive. It keeps
# teleporting models after Gazebo is gone, and piles players into a corner.
pkill -f "${HERE}/wander_players.py" >/dev/null 2>&1 || true

# Own process groups so Ctrl+C can kill Gazebo's server and GUI together.
set -m
WANDER_PID=""
GZ_PID=""
cleanup() {
  if [[ -n "${WANDER_PID}" ]]; then
    kill -TERM -- "-${WANDER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GZ_PID}" ]]; then
    kill -TERM -- "-${GZ_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 "${HERE}/wander_players.py" >/dev/null 2>&1 &
WANDER_PID=$!
gz sim -r --gui-config "${HERE}/gui.config" "${HERE}/worlds/soccer_pitch.sdf" "$@" &
GZ_PID=$!
wait "${GZ_PID}" || true
