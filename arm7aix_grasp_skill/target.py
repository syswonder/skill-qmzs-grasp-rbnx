# SPDX-License-Identifier: MulanPSL-2.0
"""目标位姿组合：标定 XY + 人工 Z + 人工固定四元数。

优先级（与 config.spec 一致）：

    Z：工具参数 > orientation_path（标定网页保存的目标）> manifest default_z_m
    四元数：工具参数 > orientation_path > manifest default_quaternion_xyzw

orientation_path 指向 handeye-calibration 保存的 config/grasp_orientation.json
（{"target_quaternion_xyzw": [...], "target_z_mm": ...}）；文件存在但内容非法
时明确报错，不静默改用别的姿态。文件按 mtime 自动重载。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .calibration import (
    CalibrationCache,
    CalibrationError,
    Projection,
    read_orientation_file,
)
from .config import ConfigError, SkillConfig, finite, parse_metres, parse_quaternion

log = logging.getLogger("arm7aix_grasp.target")


class Target:
    def __init__(
        self,
        *,
        projection: Projection,
        z: float,
        quaternion,
        sources: dict,
    ) -> None:
        self.projection = projection
        self.z = z
        self.quaternion = quaternion  # tuple[4] | None
        self.sources = sources

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float, quaternion, sources: dict) -> "Target":
        """用显式 XY 构造目标（grab/place 的配置位姿、原地放置等非像素目标）。

        projection 里只填 XY（u/v 为 0），motion/grasp 只使用 x/y/z/quaternion。
        """
        return cls(
            projection=Projection(
                u=0.0, v=0.0, undistorted_u=0.0, undistorted_v=0.0,
                x=float(x), y=float(y),
            ),
            z=float(z),
            quaternion=quaternion,
            sources=dict(sources),
        )

    @property
    def x(self) -> float:
        return self.projection.x

    @property
    def y(self) -> float:
        return self.projection.y

    def with_z(self, z: float) -> "Target":
        """复制一个只有 Z 不同的目标（hover 用），XY/姿态/来源不变。"""
        return Target(
            projection=self.projection,
            z=float(z),
            quaternion=self.quaternion,
            sources=dict(self.sources),
        )

    def note(self) -> str:
        parts = [f"z 来源={self.sources.get('z', '?')}"]
        parts.append(f"四元数来源={self.sources.get('quaternion', '未配置')}")
        if self.sources.get("orientation_file_error"):
            parts.append(f"目标设置文件问题：{self.sources['orientation_file_error']}")
        if self.sources.get("ignored_tool_overrides"):
            parts.append(
                "已忽略工具参数 "
                f"{self.sources['ignored_tool_overrides']}"
                "（allow_tool_overrides=false，以配置目标为准）"
            )
        return "；".join(parts)


class TargetResolver:
    """把一个像素 + 可选人工 Z/四元数组合成 base_link 目标位姿。"""

    def __init__(self, config: SkillConfig, calibration: CalibrationCache) -> None:
        self.config = config
        self.calibration = calibration
        self._orientation: Optional[dict] = None
        self._orientation_mtime_ns: Optional[int] = None
        self._orientation_error: Optional[str] = None

    # ── orientation_path ────────────────────────────────────────────
    def effective_orientation(self) -> dict:
        """返回 {"quaternion", "z_m", "path"}；文件缺失=没配置，文件非法=抛错。"""
        path = self.config.orientation_path
        if path is None:
            return {"quaternion": None, "z_m": None, "path": None}
        if not path.is_file():
            if self._orientation_error is None:
                log.warning(
                    "orientation_path 配置了但文件不存在：%s（回退 manifest 默认值）",
                    path,
                )
                self._orientation_error = f"目标设置文件不存在：{path}"
            return {"quaternion": None, "z_m": None, "path": str(path)}
        mtime = path.stat().st_mtime_ns
        if (
            self._orientation is not None
            and self._orientation_mtime_ns == mtime
            and self._orientation_error is None
        ):
            return self._orientation
        try:
            values = read_orientation_file(path)
        except CalibrationError as exc:
            self._orientation = None
            self._orientation_error = str(exc)
            self._orientation_mtime_ns = mtime
            raise
        self._orientation = {
            "quaternion": values["target_quaternion_xyzw"],
            "z_m": values["target_z_m"],
            "path": str(path),
        }
        self._orientation_error = None
        self._orientation_mtime_ns = mtime
        log.info("loaded target orientation from %s", path)
        return self._orientation

    @property
    def orientation_error(self) -> Optional[str]:
        return self._orientation_error

    # ── 组合 ────────────────────────────────────────────────────────
    def _resolve_z_quaternion(
        self,
        z_text,
        quaternion_text,
        require_quaternion: bool,
    ) -> tuple[float, Optional[tuple[float, ...]], dict]:
        """解析人工 Z + 固定四元数（含工具参数覆盖策略），返回 (z, q, sources)。"""
        orientation = {"quaternion": None, "z_m": None, "path": None}
        orientation_error: Optional[str] = None
        try:
            orientation = self.effective_orientation()
        except CalibrationError as exc:
            orientation_error = str(exc)

        sources: dict = {}

        # 工具参数里的 z/quaternion 默认不生效（allow_tool_overrides=false），
        # 目标以配置的固定值优先；只有当部署显式允许时才用调用参数覆盖。
        call_z = parse_metres(z_text, "z")
        call_quaternion = parse_quaternion(quaternion_text, "quaternion")
        if not self.config.allow_tool_overrides:
            ignored = []
            if call_z is not None:
                ignored.append(f"z={z_text!r}")
            if call_quaternion is not None:
                ignored.append(f"quaternion={quaternion_text!r}")
            if ignored:
                sources["ignored_tool_overrides"] = "，".join(ignored)
                log.info(
                    "忽略工具参数（allow_tool_overrides=false）：%s",
                    sources["ignored_tool_overrides"],
                )
            call_z = None
            call_quaternion = None

        # Z：工具参数（需 allow_tool_overrides）> 标定网页保存文件 > manifest 默认
        z = call_z
        if z is not None:
            sources["z"] = "工具参数"
        elif orientation.get("z_m") is not None:
            z = orientation["z_m"]
            sources["z"] = f"orientation_path ({orientation['path']})"
        elif self.config.default_z_m is not None:
            z = self.config.default_z_m
            sources["z"] = "manifest default_z_m"
        else:
            raise ConfigError(
                "没有可用的目标 Z：请在参数里给 z（如 0.1024 或 102.4mm），"
                "或配置 orientation_path / default_z_m"
            )

        # 四元数：工具参数（需 allow_tool_overrides）> 标定网页保存文件 > manifest 默认
        quaternion = call_quaternion
        if quaternion is not None:
            sources["quaternion"] = "工具参数"
        elif orientation.get("quaternion") is not None and orientation_error is None:
            quaternion = orientation["quaternion"]
            sources["quaternion"] = f"orientation_path ({orientation['path']})"
        elif self.config.default_quaternion_xyzw is not None:
            quaternion = self.config.default_quaternion_xyzw
            sources["quaternion"] = "manifest default_quaternion_xyzw"
        elif orientation_error is not None:
            raise CalibrationError(orientation_error)
        elif require_quaternion:
            raise ConfigError(
                "没有可用的固定四元数：请配置 orientation_path（标定网页保存的目标）"
                "或 default_quaternion_xyzw，或在参数里给 "
                'quaternion="qx,qy,qz,qw"'
            )

        if orientation_error is not None:
            sources["orientation_file_error"] = orientation_error
        return z, quaternion, sources

    def resolve(
        self,
        *,
        u,
        v,
        z_text=None,
        quaternion_text=None,
        require_quaternion: bool,
    ) -> Target:
        calibration = self.calibration.get()
        projection = calibration.project(u, v)
        z, quaternion, sources = self._resolve_z_quaternion(
            z_text, quaternion_text, require_quaternion
        )
        return Target(
            projection=projection,
            z=z,
            quaternion=quaternion,
            sources=sources,
        )

    def configured_pose(
        self,
        *,
        x,
        y,
        z=None,
        require_quaternion: bool = True,
    ) -> Target:
        """配置位姿（grab/place）：XY/Z 由调用方给，姿态仍取固定四元数。"""
        resolved_z, quaternion, sources = self._resolve_z_quaternion(
            None, None, require_quaternion
        )
        if z is not None:
            resolved_z = finite(z, "配置位姿 Z")
            sources["z"] = "配置位姿 (grasp_pose/place_pose)"
        return Target.from_xyz(float(x), float(y), resolved_z, quaternion, sources)

    # ── 状态 ────────────────────────────────────────────────────────
    def status(self) -> dict:
        orientation: dict = {
            "path": str(self.config.orientation_path)
            if self.config.orientation_path
            else None,
            "values": None,
            "error": None,
        }
        try:
            values = self.effective_orientation()
            orientation["values"] = {
                "target_quaternion_xyzw": list(values["quaternion"])
                if values["quaternion"]
                else None,
                "target_z_m": values["z_m"],
            }
            orientation["error"] = self._orientation_error
        except CalibrationError as exc:
            orientation["error"] = str(exc)
        return {
            "default_z_m": self.config.default_z_m,
            "default_quaternion_xyzw": list(self.config.default_quaternion_xyzw)
            if self.config.default_quaternion_xyzw
            else None,
            "orientation_path": orientation,
        }
