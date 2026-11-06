# SPDX-License-Identifier: MulanPSL-2.0
"""arm7aix_grasp 的配置解析（manifest 的 skill config → SkillConfig）。

约定：
- 配置错误不直接让 on_init 失败（技能会停在 INACTIVE 等 executor 激活），
  而是收集成 error 文本，之后每个工具调用都返回同一条明确错误（便于现场改
  manifest 后重启）。
- 路径类配置可以是绝对路径，也可以是相对部署根目录的路径。
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

log = logging.getLogger("arm7aix_grasp.config")

DEFAULT_ARM_PROVIDER = "arm7aix_controller"
DEFAULT_CAMERA_PROVIDER = "orbbec_camera"
DEFAULT_Z_M = 0.1024
DEFAULT_TOPIC_NAMES = {
    "end_pose": "/arm7aix/end_pose",
    "pos_command": "/arm7aix/pos_command",
    "joint_states": "/arm7aix/joint_states",
    "joint_command": "/arm7aix/joint_command",
}
DEFAULT_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
_WORKSPACE_KEYS = frozenset(
    {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
)
_TOPIC_KEYS = frozenset(DEFAULT_TOPIC_NAMES)
_CAMERA_INTRINSIC_KEYS = frozenset(
    {"width", "height", "fx", "fy", "cx", "cy", "coeffs"}
)


class ConfigError(ValueError):
    """manifest config 里的值非法（工具调用时转成 ok=false 返回）。"""


# ── 数值/向量解析 ────────────────────────────────────────────────────

def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} 必须是有限数值，不是布尔值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是数值，收到 {value!r}") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{name} 必须是有限数值，收到 {value!r}")
    return result


def parse_metres(text: Any, name: str = "z") -> Optional[float]:
    """解析人工指定的公制长度："" / None -> None；"0.1024"、"102.4mm"、"0.1m"。"""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return finite(text, name)
    raw = str(text).strip().lower().replace("，", "")
    if not raw:
        return None
    scale = 1.0
    if raw.endswith("mm"):
        raw, scale = raw[:-2].strip(), 1e-3
    elif raw.endswith("m"):
        raw = raw[:-1].strip()
    try:
        value = float(raw) * scale
    except ValueError as exc:
        raise ConfigError(
            f"{name} 无法解析：{text!r}（可写 0.1024 或 102.4mm）"
        ) from exc
    if not math.isfinite(value):
        raise ConfigError(f"{name} 必须是有限数值，收到 {text!r}")
    return value


def normalize_quaternion(value: Any, name: str = "quaternion") -> tuple[float, ...]:
    """校验并归一化 xyzw 四元数（允许非单位长度、允许整体取反）。"""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} 必须是 [qx,qy,qz,qw]，收到 {value!r}")
    if len(value) != 4:
        raise ConfigError(f"{name} 必须有 4 个分量，收到 {len(value)} 个")
    q = tuple(finite(v, f"{name}[{i}]") for i, v in enumerate(value))
    norm = math.sqrt(sum(v * v for v in q))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ConfigError(f"{name} 不能是零四元数")
    return tuple(v / norm for v in q)


def parse_quaternion(text: Any, name: str = "quaternion") -> Optional[tuple[float, ...]]:
    """解析 "qx,qy,qz,qw"（xyzw）；空字符串/None -> None（用配置默认值）。"""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return normalize_quaternion(text, name)
    raw = str(text).strip().replace("，", ",").replace(";", ",")
    if not raw:
        return None
    parts = [p for p in raw.replace(" ", ",").split(",") if p != ""]
    if len(parts) != 4:
        raise ConfigError(
            f"{name} 必须是 4 个用逗号分隔的数值（qx,qy,qz,qw），收到 {text!r}"
        )
    try:
        return normalize_quaternion([float(p) for p in parts], name)
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{name} 无法解析：{text!r}") from exc


def _matrix9(value: Any) -> Optional[tuple[float, ...]]:
    """接受长度 9 的行优先列表或 3x3 嵌套列表。"""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        value = [p for p in raw.replace("，", ",").replace(";", ",").replace(" ", ",").split(",") if p]
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(row, (list, tuple)) and len(row) == 3 for row in value
    ):
        value = [v for row in value for v in row]
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise ConfigError("homography 必须是 9 个数（行优先）或 3x3 嵌套列表")
    return tuple(finite(v, f"homography[{i}]") for i, v in enumerate(value))


def _camera_intrinsics(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError("camera_intrinsics 必须是对象")
    unknown = set(value) - _CAMERA_INTRINSIC_KEYS
    if unknown:
        raise ConfigError(f"camera_intrinsics 有未知字段：{sorted(unknown)}")
    width = int(finite(value.get("width", 0), "camera_intrinsics.width") or 0)
    height = int(finite(value.get("height", 0), "camera_intrinsics.height") or 0)
    fx = finite(value.get("fx"), "camera_intrinsics.fx")
    fy = finite(value.get("fy"), "camera_intrinsics.fy")
    cx = finite(value.get("cx"), "camera_intrinsics.cx")
    cy = finite(value.get("cy"), "camera_intrinsics.cy")
    coeffs_raw = value.get("coeffs", [])
    if coeffs_raw is None:
        coeffs_raw = []
    if not isinstance(coeffs_raw, (list, tuple)):
        raise ConfigError("camera_intrinsics.coeffs 必须是数值列表")
    coeffs = [finite(c, f"camera_intrinsics.coeffs[{i}]") for i, c in enumerate(coeffs_raw)]
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "coeffs": coeffs,
    }


# ── 路径解析 ─────────────────────────────────────────────────────────

def _package_root() -> Path:
    # <deploy>/skills/arm7aix_grasp/arm7aix_grasp_skill/config.py -> <deploy>/skills/arm7aix_grasp
    return Path(__file__).resolve().parents[1]


def _deploy_root() -> Path:
    # <deploy>/skills/arm7aix_grasp -> <deploy>
    return _package_root().parent.parent


def _path_roots() -> list[Path]:
    """相对路径的搜索根：部署根优先（manifest 里写 config/handeye/... 即可）。"""
    roots: list[Path] = []
    for variable in ("ROBONIX_DEPLOY_DIR", "RBNX_INVOCATION_CWD"):
        value = os.environ.get(variable, "").strip()
        if value:
            path = Path(value).expanduser()
            if path.is_dir() and path not in roots:
                roots.append(path)
    for candidate in (_deploy_root(), _package_root()):
        if candidate not in roots:
            roots.append(candidate)
    return roots


def resolve_path(value: Any) -> Optional[Path]:
    """把配置里的路径解析成绝对路径：绝对路径原样；相对路径按部署根/包根尝试。"""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    roots = _path_roots()
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    # 文件还不存在时，把部署根作为默认落点（报错信息里能看到期望路径）。
    return (roots[0] / path).resolve()


def check_workspace(workspace: Mapping[str, float], x: float, y: float, z: float) -> None:
    """按 workspace 边界检查目标（严格大于下限、小于上限）；未配置的键不限制。"""
    axes = (
        ("x", x, "x_min", "x_max"),
        ("y", y, "y_min", "y_max"),
        ("z", z, "z_min", "z_max"),
    )
    for axis, value, low_key, high_key in axes:
        low = workspace.get(low_key)
        high = workspace.get(high_key)
        if low is not None and value <= low:
            raise ConfigError(
                f"目标 {axis}={value:.4f} 不在 workspace 范围内"
                f"（{low_key}={low:.4f}），拒绝运动"
            )
        if high is not None and value >= high:
            raise ConfigError(
                f"目标 {axis}={value:.4f} 不在 workspace 范围内"
                f"（{high_key}={high:.4f}），拒绝运动"
            )


def finite_list(value: Any, name: str, expected_len: int) -> Optional[tuple[float, ...]]:
    """校验定长数值列表：空/None -> None；否则必须是 expected_len 个有限数。"""
    if value is None or value == "" or value == []:
        return None
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} 必须是 {expected_len} 个数值的列表")
    if len(value) != expected_len:
        raise ConfigError(f"{name} 必须有 {expected_len} 个数值，收到 {len(value)} 个")
    return tuple(finite(v, f"{name}[{i}]") for i, v in enumerate(value))


def normalized_list(value: Any, name: str, expected_len: int = 6) -> Optional[tuple[float, ...]]:
    """归一化 [0,1] 定长列表（手部 preset 覆盖用）。"""
    values = finite_list(value, name, expected_len)
    if values is None:
        return None
    for index, item in enumerate(values):
        if not 0.0 <= item <= 1.0:
            raise ConfigError(f"{name}[{index}] 必须在 [0,1]，收到 {item}")
    return values


def joint_names_list(value: Any) -> tuple[str, ...]:
    if value is None or value == "" or value == []:
        return DEFAULT_JOINT_NAMES
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ConfigError("joint_names 必须是 7 个关节名")
    names = tuple(str(item).strip() for item in value)
    if any(not item for item in names) or len(set(names)) != len(names):
        raise ConfigError("joint_names 不能为空或重复")
    return names


# ── 配置对象 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillConfig:
    arm_provider_id: str = DEFAULT_ARM_PROVIDER
    camera_provider_id: str = DEFAULT_CAMERA_PROVIDER

    calibration_path: Optional[Path] = None
    homography: Optional[tuple[float, ...]] = None
    camera_intrinsics: Optional[dict] = None
    orientation_path: Optional[Path] = None
    default_quaternion_xyzw: Optional[tuple[float, ...]] = None
    default_z_m: Optional[float] = DEFAULT_Z_M

    dry_run: bool = False
    feedback_timeout_s: float = 2.0
    motion_timeout_s: float = 30.0
    position_tolerance_m: float = 0.005
    angle_tolerance_rad: float = 0.03
    max_travel_m: float = 0.0
    # 两级接近：先移动到目标正上方 approach_height_m（米），再下降到目标；
    # 0 = 关闭（恢复单次移动）。
    approach_height_m: float = 0.20
    # 是否允许工具调用参数 z/quaternion 覆盖配置目标。默认 false：目标 Z/四元数
    # 以 orientation_path / manifest 默认值为准，避免上层模型自行猜测姿态。
    allow_tool_overrides: bool = False
    workspace: dict = field(default_factory=dict)
    topics: dict = field(default_factory=dict)

    # ── 抓取流程（grab / place / pick / home）─────────────────────────
    # 灵巧手（O6）provider id 与手部开合等待时间。O6 运动较慢（完全到位约
    # 10~15 s），这里的 settle 是"命令下发后等待手指动作"的时间，按现场调。
    hand_provider_id: str = "o6_hand"
    hand_open_settle_s: float = 1.0
    hand_close_settle_s: float = 3.0
    # 可选：覆盖官方 open/fist 预设（归一化 [0,1]，1=闭合），长度 6。
    hand_open_values: Optional[tuple[float, ...]] = None
    hand_close_values: Optional[tuple[float, ...]] = None
    # 七轴名称（joint_command 用）与 home 关节目标（rad）。
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES
    joint_tolerance_rad: float = 0.02
    home_joints: Optional[tuple[float, ...]] = None
    # 固定抓取/放置位姿 [x, y, z]（米，base_link）；为空时 grab 报错、
    # place 使用"原地放下"（当前 XY + 配置的目标 Z）。
    grasp_pose: Optional[tuple[float, ...]] = None
    place_pose: Optional[tuple[float, ...]] = None

    camera_topic: str = ""
    snapshot_dir: Path = field(default_factory=lambda: Path("rbnx-build/data/snapshots"))
    snapshot_timeout_s: float = 5.0

    # VLM 开放词汇检测（detect_object / move_to_object）
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""
    vlm_timeout_s: float = 60.0
    vlm_contact_ratio: float = 0.5
    vlm_save_annotated: bool = True
    vlm_prompt_extra: str = ""

    @classmethod
    def from_cfg(cls, cfg: Any) -> "SkillConfig":
        if cfg is None:
            cfg = {}
        if isinstance(cfg, str):
            import json as _json
            try:
                cfg = _json.loads(cfg) if cfg.strip() else {}
            except _json.JSONDecodeError as exc:
                raise ConfigError(f"config 不是合法 JSON：{exc}") from exc
        if not isinstance(cfg, Mapping):
            raise ConfigError(f"config 必须是对象，收到 {type(cfg).__name__}")

        default_z_raw = cfg.get("default_z_m", DEFAULT_Z_M)
        if default_z_raw is None or default_z_raw == "":
            default_z = None
        else:
            default_z = finite(default_z_raw, "default_z_m")

        quat_raw = cfg.get("default_quaternion_xyzw", [])
        default_quat = None
        if quat_raw is not None and quat_raw != "" and quat_raw != []:
            default_quat = normalize_quaternion(quat_raw, "default_quaternion_xyzw")

        workspace_raw = cfg.get("workspace") or {}
        if not isinstance(workspace_raw, Mapping):
            raise ConfigError("workspace 必须是对象")
        unknown = set(workspace_raw) - _WORKSPACE_KEYS
        if unknown:
            raise ConfigError(f"workspace 有未知字段：{sorted(unknown)}")
        workspace = {
            key: finite(value, f"workspace.{key}") for key, value in workspace_raw.items()
        }
        for low, high in (("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")):
            if low in workspace and high in workspace and workspace[low] >= workspace[high]:
                raise ConfigError(f"workspace.{low} 必须小于 workspace.{high}")

        topics_raw = cfg.get("topics") or {}
        if not isinstance(topics_raw, Mapping):
            raise ConfigError("topics 必须是对象")
        unknown = set(topics_raw) - _TOPIC_KEYS
        if unknown:
            raise ConfigError(f"topics 有未知字段：{sorted(unknown)}")
        topics = {str(key): str(value) for key, value in topics_raw.items() if str(value).strip()}
        for key, topic in topics.items():
            if not topic.startswith("/"):
                raise ConfigError(f"topics.{key} 必须是以 / 开头的绝对话题名")

        def _positive(name: str, default: float) -> float:
            value = finite(cfg.get(name, default), name)
            if value < 0:
                raise ConfigError(f"{name} 不能为负")
            return value

        position_tolerance = finite(
            cfg.get("position_tolerance_m", 0.005), "position_tolerance_m"
        )
        angle_tolerance = finite(
            cfg.get("angle_tolerance_rad", 0.03), "angle_tolerance_rad"
        )
        if position_tolerance <= 0 or angle_tolerance <= 0:
            raise ConfigError("position_tolerance_m / angle_tolerance_rad 必须大于 0")

        contact_ratio = finite(cfg.get("vlm_contact_ratio", 0.5), "vlm_contact_ratio")
        if not 0.0 <= contact_ratio <= 1.0:
            raise ConfigError("vlm_contact_ratio 必须在 [0, 1]（0=框中心，1=框底部中点）")

        hand_open_values = normalized_list(cfg.get("hand_open_values"), "hand_open_values")
        hand_close_values = normalized_list(cfg.get("hand_close_values"), "hand_close_values")
        joint_names = joint_names_list(cfg.get("joint_names"))
        home_joints = finite_list(cfg.get("home_joints"), "home_joints", 7)
        grasp_pose = finite_list(cfg.get("grasp_pose"), "grasp_pose", 3)
        place_pose = finite_list(cfg.get("place_pose"), "place_pose", 3)

        def _env_or_cfg(key: str, env_name: str) -> str:
            value = str(cfg.get(key) or "").strip()
            if value:
                return value
            return os.environ.get(env_name, "").strip()

        snapshot_dir_raw = str(cfg.get("snapshot_dir", "")).strip()
        snapshot_dir = (
            resolve_path(snapshot_dir_raw)
            if snapshot_dir_raw
            else _package_root() / "rbnx-build/data/snapshots"
        )

        return cls(
            arm_provider_id=str(cfg.get("arm_provider_id", DEFAULT_ARM_PROVIDER)).strip()
            or DEFAULT_ARM_PROVIDER,
            camera_provider_id=str(
                cfg.get("camera_provider_id", DEFAULT_CAMERA_PROVIDER)
            ).strip()
            or DEFAULT_CAMERA_PROVIDER,
            calibration_path=resolve_path(cfg.get("calibration_path")),
            homography=_matrix9(cfg.get("homography")),
            camera_intrinsics=_camera_intrinsics(cfg.get("camera_intrinsics")),
            orientation_path=resolve_path(cfg.get("orientation_path")),
            default_quaternion_xyzw=default_quat,
            default_z_m=default_z,
            dry_run=bool(cfg.get("dry_run", False)),
            feedback_timeout_s=_positive("feedback_timeout_s", 2.0),
            motion_timeout_s=_positive("motion_timeout_s", 30.0),
            position_tolerance_m=position_tolerance,
            angle_tolerance_rad=angle_tolerance,
            max_travel_m=_positive("max_travel_m", 0.0),
            approach_height_m=_positive("approach_height_m", 0.20),
            allow_tool_overrides=bool(cfg.get("allow_tool_overrides", False)),
            workspace=workspace,
            hand_provider_id=str(cfg.get("hand_provider_id", "o6_hand")).strip() or "o6_hand",
            hand_open_settle_s=_positive("hand_open_settle_s", 1.0),
            hand_close_settle_s=_positive("hand_close_settle_s", 3.0),
            hand_open_values=hand_open_values,
            hand_close_values=hand_close_values,
            joint_names=joint_names,
            joint_tolerance_rad=_positive("joint_tolerance_rad", 0.02),
            home_joints=home_joints,
            grasp_pose=grasp_pose,
            place_pose=place_pose,
            topics=topics,
            camera_topic=str(cfg.get("camera_topic", "")).strip(),
            snapshot_dir=snapshot_dir,
            snapshot_timeout_s=_positive("snapshot_timeout_s", 5.0),
            vlm_base_url=_env_or_cfg("vlm_base_url", "VLM_BASE_URL"),
            vlm_api_key=_env_or_cfg("vlm_api_key", "VLM_API_KEY"),
            vlm_model=_env_or_cfg("vlm_model", "VLM_MODEL"),
            vlm_timeout_s=_positive("vlm_timeout_s", 60.0),
            vlm_contact_ratio=contact_ratio,
            vlm_save_annotated=bool(cfg.get("vlm_save_annotated", True)),
            vlm_prompt_extra=str(cfg.get("vlm_prompt_extra", "")).strip(),
        )
