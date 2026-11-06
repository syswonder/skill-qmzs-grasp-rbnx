# SPDX-License-Identifier: MulanPSL-2.0
"""手眼标定模型：去畸变像素 → base_link XY（平面单应）。

标定由同机的 handeye-calibration 工具（固定相机 + 桌面标定面）完成，
产物 `artifacts/planar/calibration.json` 的字段约定：

    {
      "homography": 3x3,            # 去畸变像素 -> base_link XY，单位米
      "camera": {width, height, fx, fy, cx, cy, coeffs},
      "workspace_hull": [[u, v], ...],   # 去畸变像素覆盖区域（多边形）
      "table_z_mm": 123.4,               # 采点时读到的参考 Z（仅信息）
      ...
    }

几何前提（与标定工具一致）：固定相机俯视桌面，单应矩阵把“桌面上一点”的
像素映射到该点在 base_link 的 XY。机械臂法兰只要移动到这个 XY 的正上方
（人工指定 Z），XY 就成立 —— 与法兰高度无关（相机有倾角时也一样）。
如果目标是拿同一像素去反推不同高度的物体表面，则会有视差，不在这里处理。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .config import (
    ConfigError,
    SkillConfig,
    finite,
    normalize_quaternion,
)

log = logging.getLogger("qmzs_grasp.calibration")

_MAX_ABS_BASE_XY_M = 5.0  # 明显超出机械臂工作范围的投影视为标定矩阵异常


class CalibrationError(ValueError):
    """标定模型缺失/非法、像素超出覆盖区域等（工具返回 ok=false）。"""


def _matrix3(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(row, (list, tuple)) and len(row) == 3 for row in value
    ):
        value = [v for row in value for v in row]
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise CalibrationError(f"{name} 必须是 3x3 矩阵（9 个数）")
    try:
        return tuple(finite(v, f"{name}[{i}]") for i, v in enumerate(value))
    except ConfigError as exc:
        raise CalibrationError(str(exc)) from exc


def _camera_from_model(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CalibrationError("calibration.camera 必须是对象")
    try:
        width = int(finite(value.get("width", 0), "camera.width") or 0)
        height = int(finite(value.get("height", 0), "camera.height") or 0)
        fx = finite(value.get("fx"), "camera.fx")
        fy = finite(value.get("fy"), "camera.fy")
        cx = finite(value.get("cx"), "camera.cx")
        cy = finite(value.get("cy"), "camera.cy")
    except ConfigError as exc:  # finite() 抛的是 ConfigError
        raise CalibrationError(str(exc)) from exc
    coeffs_raw = value.get("coeffs", [])
    if coeffs_raw is None:
        coeffs_raw = []
    if not isinstance(coeffs_raw, (list, tuple)):
        raise CalibrationError("calibration.camera.coeffs 必须是数值列表")
    coeffs = [finite(c, f"camera.coeffs[{i}]") for i, c in enumerate(coeffs_raw)]
    if fx <= 0 or fy <= 0:
        raise CalibrationError("calibration.camera.fx/fy 必须大于 0")
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "coeffs": coeffs,
    }


def _hull_from_model(value: Any) -> Optional[list[tuple[float, float]]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise CalibrationError("calibration.workspace_hull 必须是 [[u,v], ...]")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CalibrationError(f"calibration.workspace_hull[{index}] 必须是 [u,v]")
        points.append(
            (
                finite(point[0], f"workspace_hull[{index}].u"),
                finite(point[1], f"workspace_hull[{index}].v"),
            )
        )
    if len(points) < 3:
        return None
    return points


@dataclass(frozen=True)
class Projection:
    u: float
    v: float
    undistorted_u: float
    undistorted_v: float
    x: float
    y: float


@dataclass(frozen=True)
class Calibration:
    """一个已加载的标定模型。homography 输入是去畸变像素，输出米。"""

    homography: tuple[float, ...]
    source: str
    path: Optional[Path] = None
    camera: Optional[dict] = None
    workspace_hull: Optional[list[tuple[float, float]]] = None
    table_z_mm: Optional[float] = None

    # ── 构造 ────────────────────────────────────────────────────────
    @classmethod
    def from_model(
        cls, model: Mapping, *, source: str, path: Optional[Path] = None
    ) -> "Calibration":
        if not isinstance(model, Mapping):
            raise CalibrationError("calibration.json 顶层必须是对象")
        raw_matrix = model.get("homography")
        if raw_matrix is None:
            raw_matrix = model.get("H")
        if raw_matrix is None:
            raise CalibrationError("calibration.json 缺少 homography 字段")
        table_z_raw = model.get("table_z_mm")
        table_z_mm = None
        if table_z_raw is not None and table_z_raw != "":
            try:
                table_z_mm = finite(table_z_raw, "table_z_mm")
            except ConfigError as exc:
                raise CalibrationError(str(exc)) from exc
        return cls(
            homography=_matrix3(raw_matrix, "homography"),
            source=source,
            path=path,
            camera=_camera_from_model(model.get("camera")),
            workspace_hull=_hull_from_model(model.get("workspace_hull")),
            table_z_mm=table_z_mm,
        )

    @classmethod
    def from_inline(
        cls,
        homography: Sequence[float],
        camera: Optional[Mapping] = None,
        *,
        source: str = "inline config",
    ) -> "Calibration":
        return cls(
            homography=_matrix3(homography, "homography"),
            source=source,
            path=None,
            camera=_camera_from_model(dict(camera)) if camera else None,
            workspace_hull=None,
            table_z_mm=None,
        )

    # ── 投影 ────────────────────────────────────────────────────────
    @property
    def pixel_size(self) -> Optional[tuple[int, int]]:
        if not self.camera:
            return None
        width = int(self.camera.get("width") or 0)
        height = int(self.camera.get("height") or 0)
        return (width, height) if width > 0 and height > 0 else None

    def _undistort(self, u: float, v: float) -> tuple[float, float]:
        camera = self.camera
        if not camera:
            return u, v  # 没有内参时按“已去畸变像素”处理
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # noqa: BLE001
            raise CalibrationError(
                f"去畸变需要 opencv/numpy，当前环境不可用：{exc}"
            ) from exc
        k = np.array(
            [
                [camera["fx"], 0.0, camera["cx"]],
                [0.0, camera["fy"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        d = np.array(camera["coeffs"] or [0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        point = np.array([[[float(u), float(v)]]], dtype=float)
        result = cv2.undistortPoints(point, k, d, P=k).reshape(2)
        return float(result[0]), float(result[1])

    def _check_hull(self, u: float, v: float) -> None:
        hull = self.workspace_hull
        if not hull:
            return
        try:
            import cv2
            import numpy as np
        except Exception:  # noqa: BLE001 - 没有 cv2 时跳过覆盖检查
            return
        polygon = np.array(hull, dtype=np.float32).reshape(-1, 1, 2)
        if cv2.pointPolygonTest(polygon, (float(u), float(v)), False) < 0:
            raise CalibrationError(
                "该像素超出标定点覆盖区域（workspace_hull 之外），"
                "请补采周边标定点或确认像素坐标/分辨率是否正确"
            )

    def project(self, u: Any, v: Any) -> Projection:
        """像素 -> base_link XY。u/v 必须是标定分辨率下的像素坐标。"""
        try:
            uu = finite(u, "u")
            vv = finite(v, "v")
        except ConfigError as exc:
            raise CalibrationError(str(exc)) from exc
        undistorted_u, undistorted_v = self._undistort(uu, vv)
        self._check_hull(undistorted_u, undistorted_v)

        h = self.homography
        denom = h[6] * undistorted_u + h[7] * undistorted_v + h[8]
        if abs(denom) < 1e-12:
            raise CalibrationError("单应投影分母接近 0（像素或标定矩阵异常）")
        x = (h[0] * undistorted_u + h[1] * undistorted_v + h[2]) / denom
        y = (h[3] * undistorted_u + h[4] * undistorted_v + h[5]) / denom
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CalibrationError("单应投影结果不是有限数值")
        if abs(x) > _MAX_ABS_BASE_XY_M or abs(y) > _MAX_ABS_BASE_XY_M:
            raise CalibrationError(
                f"单应投影到基座的 XY=({x:.3f}, {y:.3f}) m 明显超出机械臂范围，"
                "请检查标定矩阵/像素坐标是否一致"
            )
        return Projection(
            u=uu,
            v=vv,
            undistorted_u=undistorted_u,
            undistorted_v=undistorted_v,
            x=x,
            y=y,
        )

    def info(self) -> dict:
        return {
            "source": self.source,
            "path": str(self.path) if self.path else None,
            "homography": list(self.homography),
            "camera": self.camera,
            "pixel_size": self.pixel_size,
            "workspace_hull_points": len(self.workspace_hull or []),
            "workspace_hull": [[u, v] for u, v in (self.workspace_hull or [])],
            "table_z_mm": self.table_z_mm,
        }


class CalibrationCache:
    """按 mtime 自动重载标定文件；inline 矩阵作为文件缺失时的回退。"""

    def __init__(self, cfg: SkillConfig) -> None:
        self._cfg = cfg
        self._calibration: Optional[Calibration] = None
        self._mtime_ns: Optional[int] = None
        self._error: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _load(self) -> Optional[Calibration]:
        path = self._cfg.calibration_path
        if path is not None and path.is_file():
            mtime = path.stat().st_mtime_ns
            if self._calibration is not None and self._mtime_ns == mtime:
                return self._calibration
            try:
                model = json.loads(path.read_text(encoding="utf-8"))
                calibration = Calibration.from_model(
                    model, source="calibration file", path=path
                )
            except (OSError, json.JSONDecodeError, CalibrationError) as exc:
                self._error = f"标定文件无法使用（{path}）：{exc}"
                log.error("%s", self._error)
                self._calibration = None
                self._mtime_ns = mtime
                return None
            self._calibration = calibration
            self._mtime_ns = mtime
            self._error = None
            log.info(
                "loaded calibration %s (pixel_size=%s, hull=%d pts)",
                path,
                calibration.pixel_size,
                len(calibration.workspace_hull or []),
            )
            return calibration

        if self._cfg.homography is not None:
            if self._calibration is None or self._calibration.source != "inline config":
                try:
                    self._calibration = Calibration.from_inline(
                        self._cfg.homography, self._cfg.camera_intrinsics
                    )
                except CalibrationError as exc:
                    self._error = f"inline homography 不可用：{exc}"
                    self._calibration = None
                    return None
                self._error = None
                log.info("using inline homography from manifest config")
            return self._calibration

        if self._cfg.calibration_path is not None:
            self._error = (
                f"标定文件不存在：{self._cfg.calibration_path}；"
                "请先在 handeye-calibration 里完成桌面标定，或配置 inline homography"
            )
        else:
            self._error = "未配置标定：请设置 calibration_path 或 homography"
        self._calibration = None
        return None

    def get(self) -> Calibration:
        calibration = self._load()
        if calibration is None:
            raise CalibrationError(self._error or "标定模型不可用")
        return calibration


# ── 标定网页保存的目标（固定 Z + 四元数）────────────────────────────

def read_orientation_file(path: Path) -> dict:
    """读取 handeye-calibration 保存的 config/grasp_orientation.json。

    格式：{"target_quaternion_xyzw": [qx,qy,qz,qw], "target_z_mm": 102.4}
    返回 {"target_quaternion_xyzw": tuple|None, "target_z_m": float|None}。
    文件存在但内容非法时抛 CalibrationError（不静默回退到别的姿态）。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(f"无法读取目标设置文件 {path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"目标设置文件不是合法 JSON（{path}）：{exc}") from exc
    if not isinstance(data, Mapping):
        raise CalibrationError(f"目标设置文件顶层必须是对象：{path}")
    quat = None
    if data.get("target_quaternion_xyzw") is not None:
        try:
            quat = normalize_quaternion(
                data["target_quaternion_xyzw"], "target_quaternion_xyzw"
            )
        except ConfigError as exc:
            raise CalibrationError(f"{path}：{exc}") from exc
    z_m = None
    if data.get("target_z_mm") not in (None, ""):
        try:
            z_m = finite(data["target_z_mm"], "target_z_mm") / 1000.0
        except ConfigError as exc:
            raise CalibrationError(f"{path}：{exc}") from exc
    return {"target_quaternion_xyzw": quat, "target_z_m": z_m}
