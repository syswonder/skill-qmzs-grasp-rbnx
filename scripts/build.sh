#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# 生成本包 MCP 契约的 proto/gRPC stubs + typed Request/Response 类。
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

FLAGS=(--mcp)
[[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)
rbnx codegen -p "$PKG" "${FLAGS[@]}"
echo "[qmzs_grasp] build done"
