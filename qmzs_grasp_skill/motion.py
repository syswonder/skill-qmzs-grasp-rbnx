# SPDX-License-Identifier: MulanPSL-2.0
"""运动编排：两级接近（先到物体正上方 hover，再竖直下到目标）。

现场问题：直接发一个末端目标，原语按关节空间插值到终点，TCP 可能从侧面扫过
物体、把它碰走。这里把一次调用拆成两个端点：

    1) hover：  (x, y, z + approach_height_m)，固定四元数不变
    2) target： (x, y, z)，固定四元数不变

先到 hover 并等 `end_pose` 到位，再从 hover 下到目标；hover 超时/失败就停止，
不会继续下降。`approach_height_m = 0` 时退化为原来的单次移动。

注意：`pos_command` 是“单个 TCP 终点 + 关节空间轨迹”，原语不保证中间 TCP 路径
是严格的笛卡尔直线；但 hover → target 两个端点只差 Z（同一 XY、同一姿态），
实际下降基本是竖直的，足以避免从侧面撞到物体。若需要严格 MoveL，需要原语
额外暴露线性插值（当前 Robonix arm 契约没有）。
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .config import ConfigError, SkillConfig, check_workspace
from .ros_bridge import PoseSample, RosBridge
from .target import Target

log = logging.getLogger("qmzs_grasp.motion")


@dataclass
class MoveOutcome:
    ok: bool
    reached: bool
    timed_out: bool
    message: str
    elapsed: float
    final: Optional[PoseSample] = None


def plan_waypoints(target: Target, approach_height_m: float) -> list[Target]:
    """把目标拆成 [hover, target]；高度为 0 时只有 target 一个点。"""
    height = float(approach_height_m)
    if not math.isfinite(height) or height <= 0.0:
        return [target]
    return [target.with_z(target.z + height), target]


def _travel_check(
    config: SkillConfig,
    reference: Optional[PoseSample],
    waypoint: Target,
    stage: str,
) -> None:
    if config.max_travel_m <= 0 or reference is None:
        return
    distance = math.dist(
        (waypoint.x, waypoint.y, waypoint.z), reference.position()
    )
    if distance > config.max_travel_m:
        raise ConfigError(
            f"{stage} 目标离参考点 {distance:.3f} m，超过 max_travel_m="
            f"{config.max_travel_m:.3f} m 的限制，拒绝运动；如确认安全请调大该值"
        )


def _dry_run_message(config: SkillConfig, waypoints: list[Target], target: Target) -> str:
    if len(waypoints) > 1:
        hover = waypoints[0]
        return (
            f"dry_run=true：只计算未下发。两段：先到 hover "
            f"(x={hover.x:.4f}, y={hover.y:.4f}, z={hover.z:.4f})，再下到目标 "
            f"(x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f})；"
            f"q=({target.quaternion[0]:.4f}, {target.quaternion[1]:.4f}, "
            f"{target.quaternion[2]:.4f}, {target.quaternion[3]:.4f})；"
            f"{target.note()}"
        )
    return (
        f"dry_run=true：只计算未下发。目标 base_link "
        f"(x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f})，"
        f"q=({target.quaternion[0]:.4f}, {target.quaternion[1]:.4f}, "
        f"{target.quaternion[2]:.4f}, {target.quaternion[3]:.4f})；"
        f"{target.note()}"
    )


def perform_move(
    config: SkillConfig,
    target: Target,
    get_bridge: Callable[[], RosBridge],
) -> MoveOutcome:
    """按 [hover, target] 顺序运动；hover 未到位则停止，不下到目标。"""
    if target.quaternion is None:
        raise ConfigError("目标四元数未配置（perform_move 需要固定姿态）")
    waypoints = plan_waypoints(target, config.approach_height_m)
    for waypoint in waypoints:
        check_workspace(
            config.workspace, waypoint.x, waypoint.y, waypoint.z
        )
    if config.dry_run:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=False,
            message=_dry_run_message(config, waypoints, target),
            elapsed=0.0,
        )

    bridge = get_bridge()
    multi_stage = len(waypoints) > 1
    started = time.monotonic()
    previous_pose: Optional[PoseSample] = bridge.latest_pose()
    for index, waypoint in enumerate(waypoints):
        if multi_stage:
            stage = "approach(hover)" if index == 0 else "descend"
        else:
            stage = "move"
        _travel_check(config, previous_pose, waypoint, stage)

        result = bridge.move_pose(
            x=waypoint.x,
            y=waypoint.y,
            z=waypoint.z,
            quaternion=waypoint.quaternion,
            timeout_s=config.motion_timeout_s,
            position_tolerance_m=config.position_tolerance_m,
            angle_tolerance_rad=config.angle_tolerance_rad,
            feedback_timeout_s=config.feedback_timeout_s,
        )
        final: Optional[PoseSample] = result.get("final")
        if not result.get("reached"):
            elapsed = time.monotonic() - started
            if multi_stage and index == 0:
                detail = (
                    f"第一段（到目标正上方 {config.approach_height_m:.3f} m 的 hover）"
                    "未到位，已停止，不会继续下降。"
                )
            else:
                detail = "未到位（timed_out）。"
            message = (
                f"{detail}目标 base_link "
                f"(x={waypoint.x:.4f}, y={waypoint.y:.4f}, z={waypoint.z:.4f})，"
                f"阶段={stage}，用时 {elapsed:.2f}s；"
                "检查 arm7aix_controller 日志/限位/可达性，或目标被原语拒绝。"
                f"{target.note()}"
            )
            return MoveOutcome(
                ok=True,
                reached=False,
                timed_out=bool(result.get("timed_out")),
                message=message,
                elapsed=elapsed,
                final=final,
            )
        log.info(
            "%s 到位：base_link (%.4f, %.4f, %.4f)",
            stage,
            waypoint.x,
            waypoint.y,
            waypoint.z,
        )
        previous_pose = final

    elapsed = time.monotonic() - started
    if multi_stage:
        message = (
            f"已到位：先到目标正上方 hover (z="
            f"{waypoints[0].z:.4f} m)，再下到目标 "
            f"(x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f})，"
            f"用时 {elapsed:.2f}s；{target.note()}"
        )
    else:
        message = (
            f"已到位：base_link (x={target.x:.4f}, y={target.y:.4f}, "
            f"z={target.z:.4f})，用时 {elapsed:.2f}s；{target.note()}"
        )
    return MoveOutcome(
        ok=True, reached=True, timed_out=False, message=message,
        elapsed=elapsed, final=previous_pose,
    )


def _dry_run_waypoint_message(waypoints: list[Target], labels: list[str]) -> str:
    plan = " → ".join(
        f"{label} (x={wp.x:.4f}, y={wp.y:.4f}, z={wp.z:.4f})"
        for label, wp in zip(labels, waypoints)
    )
    target = waypoints[-1]
    q = target.quaternion or (0.0, 0.0, 0.0, 0.0)
    return (
        f"dry_run=true：只计算未下发。计划：{plan}；"
        f"q=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})；{target.note()}"
    )


def move_waypoints(
    config: SkillConfig,
    waypoints: list[Target],
    get_bridge: Callable[[], RosBridge],
    stage_labels: Optional[list[str]] = None,
) -> MoveOutcome:
    """按顺序移动到一组 TCP 端点（每段都等 end_pose 到位；任一段失败即停止）。

    grab/place/pick 用它拼抓取/放置序列；move_to_pixel 走 perform_move。
    `stage_labels` 与 waypoints 等长，仅用于日志/消息（默认 waypoint1..N）。
    """
    if not waypoints:
        raise ConfigError("move_waypoints 需要至少一个目标")
    labels = list(stage_labels) if stage_labels is not None else [
        f"waypoint{i + 1}" for i in range(len(waypoints))
    ]
    if len(labels) != len(waypoints):
        raise ConfigError(
            f"stage_labels 数量 {len(labels)} 与 waypoints {len(waypoints)} 不一致"
        )
    for waypoint in waypoints:
        if waypoint.quaternion is None:
            raise ConfigError("目标四元数未配置（move_waypoints 需要固定姿态）")
        check_workspace(config.workspace, waypoint.x, waypoint.y, waypoint.z)
    if config.dry_run:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=False,
            message=_dry_run_waypoint_message(waypoints, labels),
            elapsed=0.0,
        )

    bridge = get_bridge()
    started = time.monotonic()
    previous_pose: Optional[PoseSample] = bridge.latest_pose()
    for index, waypoint in enumerate(waypoints):
        label = labels[index]
        _travel_check(config, previous_pose, waypoint, label)
        result = bridge.move_pose(
            x=waypoint.x,
            y=waypoint.y,
            z=waypoint.z,
            quaternion=waypoint.quaternion,
            timeout_s=config.motion_timeout_s,
            position_tolerance_m=config.position_tolerance_m,
            angle_tolerance_rad=config.angle_tolerance_rad,
            feedback_timeout_s=config.feedback_timeout_s,
        )
        final: Optional[PoseSample] = result.get("final")
        if not result.get("reached"):
            elapsed = time.monotonic() - started
            stopped = ""
            if index == 0 and len(waypoints) > 1:
                stopped = "已停止，不会继续后续段。"
            message = (
                f"{label} 未到位（timed_out）。{stopped}目标 base_link "
                f"(x={waypoint.x:.4f}, y={waypoint.y:.4f}, z={waypoint.z:.4f})，"
                f"用时 {elapsed:.2f}s；检查 arm7aix_controller 日志/限位/可达性，"
                f"或目标被原语拒绝。{waypoint.note()}"
            )
            return MoveOutcome(
                ok=True,
                reached=False,
                timed_out=bool(result.get("timed_out")),
                message=message,
                elapsed=elapsed,
                final=final,
            )
        log.info(
            "%s 到位：base_link (%.4f, %.4f, %.4f)",
            label, waypoint.x, waypoint.y, waypoint.z,
        )
        previous_pose = final

    elapsed = time.monotonic() - started
    plan = " → ".join(labels)
    return MoveOutcome(
        ok=True,
        reached=True,
        timed_out=False,
        message=f"已到位：{plan}，用时 {elapsed:.2f}s；{waypoints[-1].note()}",
        elapsed=elapsed,
        final=previous_pose,
    )
