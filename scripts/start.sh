#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# 启动 qmzs_grasp：注册到 atlas、声明 MCP 工具，并按需懒初始化 rclpy
# （arm pos_command/end_pose + 可选相机 rgb）。
set -eo pipefail
PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG_ROOT"

# 直接 `rbnx boot`（没有经过部署 start.sh）时也能拿到 ROS 2 环境。
if [[ -z "${ROS_DISTRO:-}" && -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

export PYTHONPATH="$(rbnx path robonix-api):$PKG_ROOT:${PYTHONPATH:-}"
# 部署环境里 RBNX_PYTHON 指向 .venv/bin/python（--system-site-packages，含 rclpy/mcp）。
PY="${RBNX_PYTHON:-python3}"
exec "$PY" -m qmzs_grasp_skill.main
