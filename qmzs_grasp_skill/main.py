# SPDX-License-Identifier: MulanPSL-2.0
"""qmzs_grasp — ARM7AIX 七轴 + LinkerHand O6 抓取技能。

高层抓取流程（参考 robonix.skill.jaka_grasp）：

  * grab()                   固定位姿抓取：张手 → hover → 下降 → 闭合 → 抬起
  * pick(object[, pixel])    感知抓取：VLM/像素 → 位姿 → 同上
  * release()                原地张开手（零机械臂运动）
  * place()                  放下：hover → 下降 → 张手 → 抬起
  * home()                   关节空间复位到 home_joints（手不动）

底层视觉/运动辅助：

  * pixel_to_pose(u, v[, z, quaternion])   只算位姿，不下发运动
  * move_to_pixel(u, v[, z, quaternion])   算完两级接近到目标（不控制手）
  * detect_object(object[, z, quaternion]) 相机取帧 + VLM 检测 → 位姿（不运动）
  * move_to_object(object[, z, quaternion]) 检测 → 位姿 → 两级接近（不控制手）
  * calibration_status()                   查看标定/目标/手部/话题状态
  * snapshot([label])                      抓一帧相机画面存 JPEG

几何：固定相机 + 桌面标定面的平面单应（handeye-calibration 产物）
把像素映射为 base_link 的 XY；Z 与四元数由人工配置（或工具参数给出）。
运动只走 arm7aix_controller 原语的 ROS 2 契约 pos_command / end_pose，
关节限位、七轴 IK、轨迹规划、故障保护都在原语里，本技能不碰串口。

Lifecycle（skill）：rbnx boot 停在 INACTIVE；executor 在第一次 MCP 调用时
发 CMD_ACTIVATE。on_activate 解析 arm（必要）与相机（可选）话题并启动 ROS 桥；
on_deactivate 停桥。配置/标定问题不会让技能进入 ERROR，而是在每次工具调用
返回可读的错误（改完 manifest 重启即可）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from robonix_api import ATLAS, Deferred, Err, Ok, Skill

from qmzs_grasp_mcp import (  # noqa: E402  (codegen 产物)
    CalibrationStatus_Request,
    CalibrationStatus_Response,
    DetectObject_Request,
    DetectObject_Response,
    Grab_Request,
    Grab_Response,
    Home_Request,
    Home_Response,
    MoveToObject_Request,
    MoveToObject_Response,
    MoveToPixel_Request,
    MoveToPixel_Response,
    Pick_Request,
    Pick_Response,
    PixelToPose_Request,
    PixelToPose_Response,
    Place_Request,
    Place_Response,
    Release_Request,
    Release_Response,
    Snapshot_Request,
    Snapshot_Response,
)

from qmzs_grasp_skill.calibration import CalibrationCache, CalibrationError
from qmzs_grasp_skill.config import (
    ConfigError,
    SkillConfig,
    check_workspace,
    finite,
)
from qmzs_grasp_skill.grasp import (
    grasp_target,
    home_arm,
    place_target,
    release_hand,
)
from qmzs_grasp_skill.hand import HandError, connect_hand
from qmzs_grasp_skill.motion import perform_move
from qmzs_grasp_skill.ros_bridge import PoseSample, RosBridge, RosBridgeError
from qmzs_grasp_skill.target import Target, TargetResolver
from qmzs_grasp_skill.vlm import (
    VlmError,
    detect_objects,
    draw_detections,
)

logging.basicConfig(
    level=os.environ.get("QMZS_GRASP_LOG_LEVEL", "INFO"),
    format="[qmzs_grasp] %(levelname)s %(message)s",
)
log = logging.getLogger("qmzs_grasp")

skill = Skill(
    id=os.environ.get("RBNX_INSTANCE_NAME", "qmzs_grasp"),
    namespace="robonix/skill/qmzs_grasp",
)

_ARM_CONTRACTS = {
    "end_pose": "robonix/primitive/arm/end_pose",
    "pos_command": "robonix/primitive/arm/pos_command",
    "joint_states": "robonix/primitive/arm/joint_states",
    "joint_command": "robonix/primitive/arm/joint_command",
}
_DEFAULT_TOPICS = {
    "end_pose": "/arm7aix/end_pose",
    "pos_command": "/arm7aix/pos_command",
    "joint_states": "/arm7aix/joint_states",
    "joint_command": "/arm7aix/joint_command",
}
_CAMERA_RGB_CONTRACT = "robonix/primitive/camera/rgb"

_state: dict = {
    "config": None,          # SkillConfig | None
    "config_error": None,    # str | None
    "calibration": None,     # CalibrationCache | None
    "resolver": None,        # TargetResolver | None
    "bridge": None,          # RosBridge | None
    "bridge_error": None,    # str | None
    "hand": None,            # PrimitiveHand | None
    "hand_error": None,      # str | None
    "topics": {},            # key -> topic
    "camera_topic": "",
    "lock": threading.RLock(),
    "sequence_lock": threading.Lock(),  # 串行化抓取/运动序列
}


# ── 小工具 ──────────────────────────────────────────────────────────

def _require_config() -> SkillConfig:
    if _state["config_error"]:
        raise ConfigError(
            f"qmzs_grasp 配置无效：{_state['config_error']}；"
            "请修改 manifest 里 skill: - name: qmzs_grasp 的 config 后重启"
        )
    config = _state["config"]
    if config is None:
        raise ConfigError("qmzs_grasp 尚未初始化（config 未加载）")
    return config


def _require_resolver() -> TargetResolver:
    resolver = _state["resolver"]
    if resolver is None:
        raise ConfigError("qmzs_grasp 尚未初始化（target resolver 未加载）")
    return resolver


def _ensure_hand():
    """懒连接 o6_hand 的 hand/info + hand/move_joint（失败在工具调用时报错）。"""
    hand = _state["hand"]
    if hand is not None:
        return hand
    with _state["lock"]:
        if _state["hand"] is not None:
            return _state["hand"]
        config = _require_config()
        try:
            hand = connect_hand(
                skill,
                config.hand_provider_id,
                open_values_override=config.hand_open_values,
                close_values_override=config.hand_close_values,
                open_settle_s=config.hand_open_settle_s,
                close_settle_s=config.hand_close_settle_s,
            )
        except HandError as exc:
            _state["hand_error"] = str(exc)
            raise
        _state["hand"] = hand
        _state["hand_error"] = None
        return hand


@contextmanager
def _sequence():
    """同一时刻只允许一个抓取/运动序列；被占用时立即报错而不是排队。"""
    lock = _state["sequence_lock"]
    if not lock.acquire(blocking=False):
        raise ConfigError("另一个抓取/运动流程正在执行，请等它结束后再调用")
    try:
        yield
    finally:
        lock.release()


def _parse_pixel_text(text: str) -> tuple[float, float]:
    pieces = [piece for piece in re.split(r"[，,;；\s]+", str(text or "").strip()) if piece]
    if len(pieces) != 2:
        raise ConfigError(f'pixel 必须是 "u,v"（标定分辨率像素），收到 {text!r}')
    return finite(pieces[0], "pixel u"), finite(pieces[1], "pixel v")


def _require_grasp_pose(config: SkillConfig):
    if not config.grasp_pose:
        raise ConfigError(
            "未配置 grasp_pose（[x, y, z] 米）：grab 需要固定抓取位姿；"
            "感知抓取请用 pick，或在本部署 manifest 的 qmzs_grasp config 里配置"
        )
    return config.grasp_pose


def _place_target(config: SkillConfig, resolver: TargetResolver) -> Target:
    """place 的目标：配置的 place_pose，否则原地放下（当前 XY + 目标 Z）。"""
    if config.place_pose:
        return resolver.configured_pose(
            x=config.place_pose[0],
            y=config.place_pose[1],
            z=config.place_pose[2],
        )
    bridge = _ensure_bridge()
    pose = bridge.latest_pose()
    if pose is None:
        raise RosBridgeError(
            "原地放置需要当前末端位姿，但没有收到 end_pose 反馈；"
            "请确认 arm7aix_controller 已激活，或配置 place_pose"
        )
    z = config.default_z_m if config.default_z_m is not None else pose.z
    target = resolver.configured_pose(x=pose.x, y=pose.y, z=z)
    target.sources["place"] = "原地放下（当前 XY + 配置目标 Z）"
    return target


def _resolve_ros2_topic(contract_id: str, provider_id: str) -> Optional[str]:
    """经 atlas 解析一个 ROS 2 话题名；失败返回 None（调用方回退默认名）。"""
    try:
        capability = ATLAS.find_unique_capability(
            contract_id=contract_id, transport="ros2", provider_id=provider_id
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("atlas 未解析 %s: %s", contract_id, exc)
        return None
    try:
        channel = skill.connect_capability(capability, contract_id, "ros2")
    except Exception as exc:  # noqa: BLE001
        log.warning("连接 %s 失败：%s", contract_id, exc)
        return None
    topic = str(getattr(channel, "endpoint", "") or "")
    return topic or None


def _resolve_topics(config: SkillConfig) -> dict:
    """话题优先级：manifest topics 覆盖 > atlas 解析 > /arm7aix/... 默认名。"""
    resolved: dict = {}
    pending = []
    for key, contract_id in _ARM_CONTRACTS.items():
        override = (config.topics or {}).get(key)
        if override:
            resolved[key] = override
        else:
            pending.append(key)

    deadline = time.monotonic() + 20.0
    while pending and time.monotonic() < deadline:
        for key in list(pending):
            topic = _resolve_ros2_topic(_ARM_CONTRACTS[key], config.arm_provider_id)
            if topic:
                resolved[key] = topic
                pending.remove(key)
        if pending:
            time.sleep(2.0)
    for key in pending:
        log.warning(
            "arm 契约 %s 未经 atlas 解析，回退默认话题 %s",
            _ARM_CONTRACTS[key],
            _DEFAULT_TOPICS[key],
        )
        resolved[key] = _DEFAULT_TOPICS[key]
    return resolved


def _resolve_camera_topic(config: SkillConfig) -> str:
    if config.camera_topic:
        return config.camera_topic
    topic = _resolve_ros2_topic(_CAMERA_RGB_CONTRACT, config.camera_provider_id)
    if topic:
        log.info("camera/rgb resolved: %s", topic)
        return topic
    log.warning(
        "相机契约 %s 未经 atlas 解析（provider=%s）；snapshot/detect 不可用，"
        "其余工具不受影响",
        _CAMERA_RGB_CONTRACT,
        config.camera_provider_id,
    )
    return ""


def _ensure_bridge() -> RosBridge:
    """懒启动 ROS 桥（on_activate 时失败/未完成时再试一次）。"""
    bridge = _state["bridge"]
    if bridge is not None:
        return bridge
    with _state["lock"]:
        bridge = _state["bridge"]
        if bridge is not None:
            return bridge
        config = _require_config()
        topics = dict(_state["topics"]) or _resolve_topics(config)
        bridge = RosBridge(
            pos_command_topic=topics["pos_command"],
            end_pose_topic=topics["end_pose"],
            joint_states_topic=topics.get("joint_states", ""),
            joint_command_topic=topics.get("joint_command", ""),
            joint_names=config.joint_names,
            camera_topic=_state["camera_topic"],
        )
        try:
            bridge.start()
        except RosBridgeError as exc:
            _state["bridge_error"] = str(exc)
            raise
        _state["bridge_error"] = None
        _state["bridge"] = bridge
        return bridge


def _fail(kind: str, exc: Exception) -> str:
    log.warning("%s failed: %s", kind, exc)
    return f"{kind} 失败：{exc}"


def _source_note(target: Target) -> str:
    return target.note()


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _safe_label(text: str, default: str = "frame") -> str:
    label = re.sub(
        r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", str(text or default)
    ).strip("_")[:40]
    return label or default


def _save_jpeg(image_bgr, path) -> str:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise RosBridgeError(f"保存图片需要 opencv：{exc}") from exc
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image_bgr):
        raise RosBridgeError(f"写入图片失败：{target}")
    return str(target)


# ── 位姿组合 / 运动执行 ─────────────────────────────────────────────

def _pixel_to_calibration(
    pixel_u: float, pixel_v: float, frame_width: int, frame_height: int
) -> tuple[float, float]:
    """把相机帧像素换算到标定分辨率（相机分辨率变化时按比例缩放）。"""
    cache = _state["calibration"]
    if cache is None:
        return pixel_u, pixel_v
    try:
        calibration = cache.get()
    except CalibrationError:
        return pixel_u, pixel_v
    size = calibration.pixel_size
    if not size:
        return pixel_u, pixel_v
    calib_w, calib_h = size
    if calib_w == frame_width and calib_h == frame_height:
        return pixel_u, pixel_v
    log.warning(
        "相机帧分辨率 %dx%d 与标定分辨率 %dx%d 不同，按比例缩放像素"
        "（建议把相机原语配成标定分辨率）",
        frame_width,
        frame_height,
        calib_w,
        calib_h,
    )
    return (
        pixel_u * calib_w / float(frame_width),
        pixel_v * calib_h / float(frame_height),
    )


def _final_fields(final: Optional[PoseSample]) -> dict:
    return {
        "final_x": final.x if final else 0.0,
        "final_y": final.y if final else 0.0,
        "final_z": final.z if final else 0.0,
        "final_qx": final.qx if final else 0.0,
        "final_qy": final.qy if final else 0.0,
        "final_qz": final.qz if final else 0.0,
        "final_qw": final.qw if final else 0.0,
    }


def _q_fields(quaternion) -> dict:
    return {
        "qx": quaternion[0],
        "qy": quaternion[1],
        "qz": quaternion[2],
        "qw": quaternion[3],
    }


# ── VLM 检测 ────────────────────────────────────────────────────────

@dataclass
class DetectionOutcome:
    detection: object  # vlm.Detection
    frame: object      # ros_bridge.Frame
    pixel_u: float
    pixel_v: float
    annotated_path: str
    detection_count: int
    class_names: list


def _detect_from_camera(
    config: SkillConfig, instruction: str
) -> Optional[DetectionOutcome]:
    """相机取一帧 → VLM 检测 → 抓取点像素（已换算到标定分辨率）。

    没有检测到目标时返回 None；VLM/相机配置问题抛 VlmError / RosBridgeError。
    """
    bridge = _ensure_bridge()
    if not bridge.camera_topic:
        raise RosBridgeError(
            "相机话题未解析；确认部署里有相机原语（如 orbbec_camera）并已 ACTIVE，"
            "或在 config 里设置 camera_topic"
        )
    if not (config.vlm_base_url and config.vlm_model):
        raise VlmError(
            "VLM 未配置：设置 manifest 的 vlm_base_url/vlm_model（或环境变量 "
            "VLM_BASE_URL/VLM_API_KEY/VLM_MODEL）"
        )
    frame = bridge.wait_frame(config.snapshot_timeout_s)
    if frame is None:
        raise RosBridgeError(
            f"{config.snapshot_timeout_s:.1f}s 内没有收到相机画面"
            f"（{bridge.camera_topic}）"
        )
    detections = detect_objects(
        frame.bgr,
        instruction,
        base_url=config.vlm_base_url,
        api_key=config.vlm_api_key,
        model=config.vlm_model,
        timeout_s=config.vlm_timeout_s,
        prompt_extra=config.vlm_prompt_extra,
    )
    if not detections:
        return None
    contacts = [
        detection.contact_pixel(
            frame.width, frame.height, ratio=config.vlm_contact_ratio
        )
        for detection in detections
    ]
    annotated_path = ""
    if config.vlm_save_annotated:
        try:
            annotated = draw_detections(frame.bgr, detections, contacts)
            annotated_path = _save_jpeg(
                annotated, config.snapshot_dir / f"detect_{_stamp()}.jpg"
            )
        except Exception as exc:  # noqa: BLE001 - 标注失败不影响检测
            log.warning("保存检测标注图失败：%s", exc)
    pixel_u, pixel_v = _pixel_to_calibration(
        contacts[0][0], contacts[0][1], frame.width, frame.height
    )
    return DetectionOutcome(
        detection=detections[0],
        frame=frame,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
        annotated_path=annotated_path,
        detection_count=len(detections),
        class_names=[d.class_name for d in detections],
    )


def _detect_target(
    config: SkillConfig,
    resolver: TargetResolver,
    instruction: str,
    z_text,
    quaternion_text,
    *,
    require_quaternion: bool,
) -> Optional[tuple[DetectionOutcome, Target]]:
    found = _detect_from_camera(config, instruction)
    if found is None:
        return None
    target = resolver.resolve(
        u=found.pixel_u,
        v=found.pixel_v,
        z_text=z_text,
        quaternion_text=quaternion_text,
        require_quaternion=require_quaternion,
    )
    check_workspace(config.workspace, target.x, target.y, target.z)
    return found, target


def _detection_note(found: DetectionOutcome) -> str:
    note = (
        f"VLM 类别={found.detection.class_name}，"
        f"抓取像素=({found.pixel_u:.1f}, {found.pixel_v:.1f})"
    )
    if found.detection_count > 1:
        note += (
            f"，共检测到 {found.detection_count} 个目标"
            f"（取第一个：{', '.join(found.class_names[:4])}）"
        )
    if found.annotated_path:
        note += f"，标注图={found.annotated_path}"
    return note


# ── 生命周期 ────────────────────────────────────────────────────────

@skill.on_init
def init(cfg) -> Ok | Err:
    """REGISTERED → INACTIVE：解析配置、创建标定缓存（不碰 ROS / atlas）。"""
    _state["config_error"] = None
    _state["config"] = None
    _state["calibration"] = None
    _state["resolver"] = None
    try:
        config = SkillConfig.from_cfg(cfg)
    except ConfigError as exc:
        _state["config_error"] = str(exc)
        log.error("invalid qmzs_grasp config: %s", exc)
        return Ok()  # 错误在工具调用时返回，便于改配置后重启
    _state["config"] = config
    cache = CalibrationCache(config)
    _state["calibration"] = cache
    _state["resolver"] = TargetResolver(config, cache)
    try:
        cache.get()
    except CalibrationError as exc:
        log.warning("calibration not ready: %s", exc)
    log.info(
        "CMD_INIT ok (arm_provider=%s camera_provider=%s calibration=%s orientation=%s "
        "vlm=%s)",
        config.arm_provider_id,
        config.camera_provider_id,
        config.calibration_path or "inline",
        config.orientation_path or "(none)",
        config.vlm_model or "(unconfigured)",
    )
    return Ok()


@skill.on_activate
def activate() -> Ok | Deferred:
    """INACTIVE → ACTIVE：解析话题、启动 ROS 桥（失败延后到工具调用时报错）。"""
    try:
        config = _require_config()
    except ConfigError as exc:
        log.error("activate without valid config: %s", exc)
        return Ok()
    if _state["bridge"] is not None:
        return Ok()  # 幂等：重复 activate 不重复建节点
    _state["topics"] = _resolve_topics(config)
    _state["camera_topic"] = _resolve_camera_topic(config)
    try:
        bridge = RosBridge(
            pos_command_topic=_state["topics"]["pos_command"],
            end_pose_topic=_state["topics"]["end_pose"],
            joint_states_topic=_state["topics"].get("joint_states", ""),
            joint_command_topic=_state["topics"].get("joint_command", ""),
            joint_names=config.joint_names,
            camera_topic=_state["camera_topic"],
        )
        bridge.start()
        _state["bridge"] = bridge
        _state["bridge_error"] = None
        log.info(
            "CMD_ACTIVATE ok (pos_command=%s, end_pose=%s, camera=%s, vlm=%s)",
            _state["topics"]["pos_command"],
            _state["topics"]["end_pose"],
            _state["camera_topic"] or "(none)",
            config.vlm_model or "(unconfigured)",
        )
    except RosBridgeError as exc:
        _state["bridge_error"] = str(exc)
        log.warning("ROS bridge not ready (tools will report): %s", exc)
    if _state["bridge"] is not None:
        try:
            _state["hand"] = connect_hand(
                skill,
                config.hand_provider_id,
                open_values_override=config.hand_open_values,
                close_values_override=config.hand_close_values,
                open_settle_s=config.hand_open_settle_s,
                close_settle_s=config.hand_close_settle_s,
            )
            _state["hand_error"] = None
        except HandError as exc:
            _state["hand_error"] = str(exc)
            log.warning("hand primitive not ready (grasp tools will report): %s", exc)
    return Ok()


@skill.on_deactivate
def deactivate() -> Ok:
    bridge = _state["bridge"]
    _state["bridge"] = None
    _state["hand"] = None
    if bridge is not None:
        try:
            bridge.stop()
        except Exception:  # noqa: BLE001
            log.exception("stopping ROS bridge failed")
    log.info("CMD_DEACTIVATE ok")
    return Ok()


@skill.on_shutdown
def shutdown() -> None:
    deactivate()
    log.info("shutdown")


# ── MCP 工具 ────────────────────────────────────────────────────────

@skill.mcp("robonix/skill/qmzs_grasp/pixel_to_pose")
def pixel_to_pose(req: PixelToPose_Request) -> PixelToPose_Response:
    """像素(u,v) → 基座(base_link)位姿，只计算不运动。

    手眼标定把像素换算成 base_link 的 XY（米），再叠加人工指定的法兰 Z 与
    固定四元数（xyzw），组成 [x, y, z, qx, qy, qz, qw]。像素必须使用标定
    分辨率（默认 640×480）。z/quaternion 留空时取标定网页保存的目标设置
    （orientation_path）或 manifest 默认值。当用户说“这个像素对应机械臂哪个
    位置”“把像素换算成坐标”时调用。不会移动机械臂。
    注意：z/quaternion 参数默认被忽略（manifest allow_tool_overrides=false）；调用时请留空，不要自行猜测
    或换算四元数，目标 Z/四元数以部署配置（config/handeye/grasp_orientation.json）为准。
    """
    try:
        config = _require_config()
        resolver = _require_resolver()
        target = resolver.resolve(
            u=req.u,
            v=req.v,
            z_text=req.z,
            quaternion_text=req.quaternion,
            require_quaternion=False,
        )
        check_workspace(config.workspace, target.x, target.y, target.z)
        q = target.quaternion or (0.0, 0.0, 0.0, 0.0)
        detail = _source_note(target)
        if target.quaternion is None:
            detail += "；四元数未配置，仅返回 XY/Z"
        return PixelToPose_Response(
            ok=True,
            message=(
                f"像素 ({req.u:.1f}, {req.v:.1f}) -> base_link XY "
                f"({target.x:.4f}, {target.y:.4f}) m，z={target.z:.4f} m；{detail}"
            ),
            x=target.x,
            y=target.y,
            z=target.z,
            **_q_fields(q),
            undistorted_u=target.projection.undistorted_u,
            undistorted_v=target.projection.undistorted_v,
        )
    except (ConfigError, CalibrationError) as exc:
        return PixelToPose_Response(ok=False, message=_fail("pixel_to_pose", exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("pixel_to_pose crashed")
        return PixelToPose_Response(ok=False, message=_fail("pixel_to_pose", exc))


@skill.mcp("robonix/skill/qmzs_grasp/move_to_pixel")
def move_to_pixel(req: MoveToPixel_Request) -> MoveToPixel_Response:
    """把机械臂移动到像素 (u,v) 对应的位姿：标定 XY + 人工 Z + 固定四元数。

    先用与 pixel_to_pose 相同的几何细节算出目标位姿，再分两段运动：
    先到目标正上方 approach_height_m（默认 0.20 m）的 hover，再从 hover 下到目标；
    hover 未到位就停止，不会继续下降。每段都发 pos_command 并等 end_pose 到位。
    像素必须使用标定分辨率（默认 640×480）；z/quaternion 留空取
    orientation_path 或 manifest 默认值。dry_run=true 时只算不发。
    当用户说“把机械臂移到这个像素/这个点”“移动到像素坐标对应的位置”时调用。
    调用前确认机械臂周围安全、没有遮挡物。
    注意：z/quaternion 参数默认被忽略（manifest allow_tool_overrides=false）；调用时请留空，不要自行猜测
    或换算四元数，目标 Z/四元数以部署配置（config/handeye/grasp_orientation.json）为准。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        resolver = _require_resolver()
        target = resolver.resolve(
            u=req.u,
            v=req.v,
            z_text=req.z,
            quaternion_text=req.quaternion,
            require_quaternion=True,
        )
        with _sequence():
            outcome = perform_move(config, target, _ensure_bridge)
        return MoveToPixel_Response(
            ok=outcome.ok,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=outcome.message,
            x=target.x,
            y=target.y,
            z=target.z,
            **_q_fields(target.quaternion),
            elapsed_sec=outcome.elapsed,
            **_final_fields(outcome.final),
        )
    except (ConfigError, CalibrationError, RosBridgeError) as exc:
        return MoveToPixel_Response(
            ok=False,
            message=_fail("move_to_pixel", exc),
            elapsed_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("move_to_pixel crashed")
        return MoveToPixel_Response(
            ok=False,
            message=_fail("move_to_pixel", exc),
            elapsed_sec=time.monotonic() - started,
        )


@skill.mcp("robonix/skill/qmzs_grasp/detect_object")
def detect_object(req: DetectObject_Request) -> DetectObject_Response:
    """用 VLM 在相机画面里找目标物体，返回它的像素和机械臂目标位姿（不运动）。

    流程：相机原语取一帧 → OpenAI 兼容多模态端点做开放词汇检测（自然语言
    描述，如“红色的方块”“最右边的杯子”）→ 有向框 → 抓取点像素 → 手眼标定
    XY + 人工 Z + 固定四元数。Z/四元数留空取 orientation_path 或 manifest
    默认值。会保存一张标注了框和抓取点的 JPEG（annotated_path），方便核对
    VLM 是否框对。需要相机原语 + VLM 配置（vlm_* 或环境变量 VLM_*）。
    当用户问“红色积木在哪里”“看看桌上有什么”时调用。不会移动机械臂。
    注意：z/quaternion 参数默认被忽略（manifest allow_tool_overrides=false）；调用时请留空，不要自行猜测
    或换算四元数，目标 Z/四元数以部署配置（config/handeye/grasp_orientation.json）为准。
    """
    try:
        config = _require_config()
        resolver = _require_resolver()
        result = _detect_target(
            config,
            resolver,
            req.object,
            req.z,
            req.quaternion,
            require_quaternion=False,
        )
        if result is None:
            return DetectObject_Response(
                ok=True,
                found=False,
                message=(
                    f"VLM 未在画面中找到「{req.object}」"
                    "（检查光照、目标描述或 vlm_prompt_extra）"
                ),
            )
        found, target = result
        q = target.quaternion or (0.0, 0.0, 0.0, 0.0)
        detail = _source_note(target)
        if target.quaternion is None:
            detail += "；四元数未配置，仅返回 XY/Z"
        return DetectObject_Response(
            ok=True,
            found=True,
            message=(
                f"找到「{found.detection.class_name}」：{_detection_note(found)}；"
                f"base_link (x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f})；"
                f"{detail}"
            ),
            class_name=found.detection.class_name,
            u=found.pixel_u,
            v=found.pixel_v,
            x=target.x,
            y=target.y,
            z=target.z,
            **_q_fields(q),
            annotated_path=found.annotated_path,
        )
    except (ConfigError, CalibrationError, RosBridgeError, VlmError) as exc:
        return DetectObject_Response(
            ok=False, found=False, message=_fail("detect_object", exc)
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("detect_object crashed")
        return DetectObject_Response(
            ok=False, found=False, message=_fail("detect_object", exc)
        )


@skill.mcp("robonix/skill/qmzs_grasp/move_to_object")
def move_to_object(req: MoveToObject_Request) -> MoveToObject_Response:
    """用 VLM 找到目标物体，然后把机械臂移动到它的目标位姿。

    与 detect_object 相同的检测/换算链路（相机取帧 → VLM 有向框 → 抓取点 →
    标定 XY + 人工 Z + 固定四元数），随后分两段运动：先到目标正上方
    approach_height_m（默认 0.20 m）的 hover，再从 hover 下到目标；hover 未到位
    就停止，不会继续下降。
    Z/四元数留空取 orientation_path 或 manifest 默认值；dry_run=true 时只算
    不发。当用户说“把机械臂移到红色积木上方”“移动到那个杯子”时调用。
    调用前确认机械臂周围安全、没有遮挡物。
    注意：z/quaternion 参数默认被忽略（manifest allow_tool_overrides=false）；调用时请留空，不要自行猜测
    或换算四元数，目标 Z/四元数以部署配置（config/handeye/grasp_orientation.json）为准。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        resolver = _require_resolver()
        result = _detect_target(
            config,
            resolver,
            req.object,
            req.z,
            req.quaternion,
            require_quaternion=True,
        )
        if result is None:
            return MoveToObject_Response(
                ok=True,
                found=False,
                message=f"VLM 未在画面中找到「{req.object}」，未运动",
                elapsed_sec=time.monotonic() - started,
            )
        found, target = result
        with _sequence():
            outcome = perform_move(config, target, _ensure_bridge)
        message = f"{_detection_note(found)}；{outcome.message}"
        return MoveToObject_Response(
            ok=outcome.ok,
            found=True,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=message,
            class_name=found.detection.class_name,
            u=found.pixel_u,
            v=found.pixel_v,
            x=target.x,
            y=target.y,
            z=target.z,
            **_q_fields(target.quaternion),
            elapsed_sec=outcome.elapsed,
            **_final_fields(outcome.final),
        )
    except (ConfigError, CalibrationError, RosBridgeError, VlmError) as exc:
        return MoveToObject_Response(
            ok=False,
            found=False,
            message=_fail("move_to_object", exc),
            elapsed_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("move_to_object crashed")
        return MoveToObject_Response(
            ok=False,
            found=False,
            message=_fail("move_to_object", exc),
            elapsed_sec=time.monotonic() - started,
        )


@skill.mcp("robonix/skill/qmzs_grasp/grab")
def grab(_req: Grab_Request) -> Grab_Response:
    """固定位姿抓取：在配置的 grasp_pose 处张手 → 两级接近 → 闭合手 → 抬起。

    不需要视觉：grasp_pose（[x, y, z] 米，base_link）和固定四元数来自部署配置。
    hold 时先张手，再移动到目标正上方 approach_height_m，然后下降、闭合手、
    等待 hand_close_settle_s、抬回 hover。接近阶段失败时手保持张开并停止。
    当用户说“抓住 / 抓一下 / 在这里抓”时调用；没有配置 grasp_pose 时会报错，
    此时请用 pick（VLM 或像素定位）。调用前确认手周围与机械臂路径安全。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        resolver = _require_resolver()
        pose = _require_grasp_pose(config)
        target = resolver.configured_pose(x=pose[0], y=pose[1], z=pose[2])
        hand = None if config.dry_run else _ensure_hand()
        with _sequence():
            outcome = grasp_target(config, target, _ensure_bridge, hand)
        return Grab_Response(
            ok=outcome.ok,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=outcome.message,
            elapsed_sec=outcome.elapsed,
        )
    except (ConfigError, CalibrationError, RosBridgeError, HandError) as exc:
        return Grab_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("grab", exc),
            elapsed_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("grab crashed")
        return Grab_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("grab", exc),
            elapsed_sec=time.monotonic() - started,
        )


@skill.mcp("robonix/skill/qmzs_grasp/pick")
def pick(req: Pick_Request) -> Pick_Response:
    """感知抓取：找到目标 → 计算位姿 → 张手 → 两级接近 → 闭合手 → 抬起。

    定位方式二选一：
      - pixel 非空（"u,v"，标定分辨率像素）：直接用该像素走手眼标定；
      - pixel 为空：用 VLM 开放词汇检测 object（如“红色方块”“最右边的杯子”，
        需要相机原语 + vlm_* 配置）。
    目标 Z/四元数使用部署固定配置（allow_tool_overrides=false）。
    当用户说“抓住红色积木 / 拿起来 / pick up X”时调用。调用前确认手和路径安全。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        resolver = _require_resolver()
        hand = None if config.dry_run else _ensure_hand()
        with _sequence():
            pixel_text = str(req.pixel or "").strip()
            class_name = ""
            if pixel_text:
                u, v = _parse_pixel_text(pixel_text)
                note = f"使用像素 ({u:.1f}, {v:.1f})"
            else:
                detection = _detect_from_camera(config, req.object)
                if detection is None:
                    return Pick_Response(
                        ok=True,
                        found=False,
                        reached=False,
                        timed_out=False,
                        message=f"VLM 未在画面中找到「{req.object}」，未运动",
                        elapsed_sec=time.monotonic() - started,
                    )
                u, v = detection.pixel_u, detection.pixel_v
                class_name = detection.detection.class_name
                note = _detection_note(detection)
            target = resolver.resolve(
                u=u, v=v, z_text="", quaternion_text="", require_quaternion=True
            )
            outcome = grasp_target(config, target, _ensure_bridge, hand)
        return Pick_Response(
            ok=outcome.ok,
            found=True,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=f"{note}；{outcome.message}",
            class_name=class_name,
            u=u,
            v=v,
            x=target.x,
            y=target.y,
            z=target.z,
            **_q_fields(target.quaternion),
            elapsed_sec=outcome.elapsed,
        )
    except (ConfigError, CalibrationError, RosBridgeError, HandError, VlmError) as exc:
        return Pick_Response(
            ok=False, found=False, reached=False, timed_out=False,
            message=_fail("pick", exc),
            elapsed_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("pick crashed")
        return Pick_Response(
            ok=False, found=False, reached=False, timed_out=False,
            message=_fail("pick", exc),
            elapsed_sec=time.monotonic() - started,
        )


@skill.mcp("robonix/skill/qmzs_grasp/release")
def release(_req: Release_Request) -> Release_Response:
    """原地张开 O6 手，零机械臂运动。

    当用户说“放开 / 松手 / release”时调用；物体留在当前位置（空中松手用
    release，放下用 place）。会等待 hand_open_settle_s（O6 完全到位约 10~15s）。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        hand = None if config.dry_run else _ensure_hand()
        with _sequence():
            outcome = release_hand(config, hand)
        return Release_Response(
            ok=outcome.ok,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=outcome.message,
            elapsed_sec=outcome.elapsed,
        )
    except (ConfigError, CalibrationError, RosBridgeError, HandError) as exc:
        return Release_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("release", exc),
            elapsed_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("release crashed")
        return Release_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("release", exc),
            elapsed_sec=time.monotonic() - started,
        )


@skill.mcp("robonix/skill/qmzs_grasp/place")
def place(_req: Place_Request) -> Place_Response:
    """把手中物体放下：hover → 下降到放置点 → 张开手 → 抬回 hover。

    place_pose（[x, y, z] 米）来自部署配置；未配置时原地放下（当前末端 XY +
    配置的目标 Z）。与 release 的区别是手在下降到放置点之后才张开。
    当用户说“放下 / 放到这里 / place”时调用。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        resolver = _require_resolver()
        hand = None if config.dry_run else _ensure_hand()
        with _sequence():
            target = _place_target(config, resolver)
            outcome = place_target(config, target, _ensure_bridge, hand)
        return Place_Response(
            ok=outcome.ok,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=outcome.message,
            elapsed_sec=outcome.elapsed,
        )
    except (ConfigError, CalibrationError, RosBridgeError, HandError) as exc:
        return Place_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("place", exc),
            elapsed_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("place crashed")
        return Place_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("place", exc),
            elapsed_sec=time.monotonic() - started,
        )


@skill.mcp("robonix/skill/qmzs_grasp/home")
def home(_req: Home_Request) -> Home_Response:
    """关节空间复位到配置的 home_joints（rad）；手保持当前开合状态。

    经 arm7aix_controller 的 joint_command 下发七轴目标并等 joint_states 到位。
    手里有物体时请先 release/place；归位路径可能扫过工作空间，调用前清空路径。
    当用户说“复位 / 回原点 / home”时调用。
    """
    started = time.monotonic()
    try:
        config = _require_config()
        with _sequence():
            outcome = home_arm(config, _ensure_bridge)
        bridge = _state["bridge"]
        positions = bridge.latest_joint_positions() if bridge else None
        final = list(positions) if positions is not None else [0.0] * 7
        return Home_Response(
            ok=outcome.ok,
            reached=outcome.reached,
            timed_out=outcome.timed_out,
            message=outcome.message,
            elapsed_sec=outcome.elapsed,
            final_positions=final,
        )
    except (ConfigError, CalibrationError, RosBridgeError, HandError) as exc:
        return Home_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("home", exc),
            elapsed_sec=time.monotonic() - started,
            final_positions=[0.0] * 7,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("home crashed")
        return Home_Response(
            ok=False, reached=False, timed_out=False,
            message=_fail("home", exc),
            elapsed_sec=time.monotonic() - started,
            final_positions=[0.0] * 7,
        )


@skill.mcp("robonix/skill/qmzs_grasp/calibration_status")
def calibration_status(_req: CalibrationStatus_Request) -> CalibrationStatus_Response:
    """查看当前标定与默认目标（只读）：单应矩阵来源、相机内参、覆盖区域、
    默认 Z/四元数及其来源、运动话题、VLM 配置。现场排障时先调用它确认配置。
    """
    try:
        config = _require_config()
        resolver = _require_resolver()
        details: dict = {
            "arm_provider_id": config.arm_provider_id,
            "camera_provider_id": config.camera_provider_id,
            "topics": dict(_state["topics"]) or dict(config.topics),
            "camera_topic": _state["camera_topic"] or config.camera_topic,
            "dry_run": config.dry_run,
            "motion_timeout_s": config.motion_timeout_s,
            "position_tolerance_m": config.position_tolerance_m,
            "angle_tolerance_rad": config.angle_tolerance_rad,
            "max_travel_m": config.max_travel_m,
            "approach_height_m": config.approach_height_m,
            "workspace": config.workspace,
            "allow_tool_overrides": config.allow_tool_overrides,
            "vlm": {
                "configured": bool(config.vlm_base_url and config.vlm_model),
                "base_url": config.vlm_base_url or None,
                "model": config.vlm_model or None,
                "contact_ratio": config.vlm_contact_ratio,
                "save_annotated": config.vlm_save_annotated,
                "timeout_s": config.vlm_timeout_s,
            },
            "hand": {
                "provider_id": config.hand_provider_id,
                "connected": _state["hand"] is not None,
                "error": _state["hand_error"],
                "open_settle_s": config.hand_open_settle_s,
                "close_settle_s": config.hand_close_settle_s,
            },
            "grasp": {
                "grasp_pose": list(config.grasp_pose) if config.grasp_pose else None,
                "place_pose": list(config.place_pose) if config.place_pose else None,
                "home_joints": list(config.home_joints) if config.home_joints else None,
                "joint_tolerance_rad": config.joint_tolerance_rad,
            },
        }
        calibrated = True
        calibration_details = None
        cache = _state["calibration"]
        try:
            if cache is None:
                raise CalibrationError("标定缓存未初始化")
            calibration_details = cache.get().info()
        except CalibrationError as exc:
            calibrated = False
            calibration_details = {"error": str(exc)}
        details["calibration"] = calibration_details
        details["targets"] = resolver.status()

        bridge = _state["bridge"]
        details["ros_bridge"] = {
            "started": bridge is not None,
            "error": _state["bridge_error"],
            "feedback_age_sec": bridge.feedback_age() if bridge else None,
        }

        if calibrated:
            message = "标定已加载"
        else:
            message = f"标定不可用：{calibration_details.get('error')}"
        if not (config.vlm_base_url and config.vlm_model):
            message += "；VLM 未配置（detect_object / move_to_object 不可用）"
        return CalibrationStatus_Response(
            calibrated=calibrated,
            message=message,
            details_json=json.dumps(
                details, ensure_ascii=False, indent=2, default=str
            ),
        )
    except (ConfigError, CalibrationError) as exc:
        return CalibrationStatus_Response(
            calibrated=False,
            message=_fail("calibration_status", exc),
            details_json=json.dumps({"error": str(exc)}, ensure_ascii=False),
        )


@skill.mcp("robonix/skill/qmzs_grasp/snapshot")
def snapshot(req: Snapshot_Request) -> Snapshot_Response:
    """抓取相机原语当前 RGB 画面并保存为 JPEG，返回文件路径和分辨率。

    像素→位姿工具需要调用方给像素；本工具用于没有现成视觉程序时先取一张
    画面（在电脑上打开量取像素坐标），然后调用 pixel_to_pose/move_to_pixel。
    需要相机原语（robonix/primitive/camera/rgb）在线；不回相机时 ok=false。
    """
    try:
        config = _require_config()
        bridge = _ensure_bridge()
        if not bridge.camera_topic:
            raise RosBridgeError(
                "相机话题未解析；确认部署里有相机原语并已 ACTIVE，"
                "或在 config 里设置 camera_topic"
            )
        frame = bridge.wait_frame(config.snapshot_timeout_s)
        if frame is None:
            raise RosBridgeError(
                f"{config.snapshot_timeout_s:.1f}s 内没有收到相机画面"
                f"（{bridge.camera_topic}）"
            )
        target = config.snapshot_dir / f"{_safe_label(req.label)}_{_stamp()}.jpg"
        saved = _save_jpeg(frame.bgr, target)
        return Snapshot_Response(
            ok=True,
            message=f"已保存相机画面：{saved}",
            path=saved,
            width=int(frame.width),
            height=int(frame.height),
        )
    except (ConfigError, CalibrationError, RosBridgeError) as exc:
        return Snapshot_Response(ok=False, message=_fail("snapshot", exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("snapshot crashed")
        return Snapshot_Response(ok=False, message=_fail("snapshot", exc))


def main() -> int:
    skill.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
