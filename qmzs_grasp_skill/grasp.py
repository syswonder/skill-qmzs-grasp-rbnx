# SPDX-License-Identifier: MulanPSL-2.0
"""抓取 / 放置 / 复位流程编排（grab、pick、place、release、home）。

流程与 `robonix.skill.jaka_grasp` 对齐：

    grab/pick : 张手 → hover（目标正上方）→ 下降到目标 → 闭合手 → 等待
                → 抬回 hover（失败非致命，物体已在手指间）
    place     : hover → 下降到放置点 → 张手 → 等待 → 抬回 hover
    release   : 原地张手，零机械臂运动
    home      : 关节空间移到 home_joints（joint_command），手不动

所有硬件访问都通过注入的 `bridge`（ROS 2 话题）和 `hand`（hand/move_joint gRPC）
完成；本模块不 import robonix_api，便于离线测试。
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from .config import ConfigError, SkillConfig
from .motion import MoveOutcome, move_waypoints, plan_waypoints
from .ros_bridge import PoseSample, RosBridge
from .target import Target

log = logging.getLogger("qmzs_grasp.grasp")


def _hand_label(hand) -> str:
    return "O6 手"


def _grasp_plan(target: Target, config: SkillConfig) -> str:
    waypoints = plan_waypoints(target, config.approach_height_m)
    labels = (
        ["approach(hover)", "descend"] if len(waypoints) > 1 else ["grasp"]
    )
    plan = " → ".join(
        f"{label} (x={wp.x:.4f}, y={wp.y:.4f}, z={wp.z:.4f})"
        for label, wp in zip(labels, waypoints)
    )
    return f"张手 → {plan} → 闭合手（等待 {config.hand_close_settle_s:g}s）" + (
        f" → 抬回 hover (z={waypoints[0].z:.4f})" if len(waypoints) > 1 else ""
    )


def grasp_target(
    config: SkillConfig,
    target: Target,
    get_bridge: Callable[[], RosBridge],
    hand,
) -> MoveOutcome:
    """张手 → 两级接近 → 闭合手 → 抬起；hover 失败则停止且手保持张开。"""
    if target.quaternion is None:
        raise ConfigError("抓取目标四元数未配置")
    waypoints = plan_waypoints(target, config.approach_height_m)
    labels = (
        ["approach(hover)", "descend"] if len(waypoints) > 1 else ["grasp"]
    )
    lift_waypoint: Optional[Target] = waypoints[0] if len(waypoints) > 1 else None
    if config.dry_run:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=False,
            message=(
                "dry_run=true：只计算未下发。抓取计划："
                f"{_grasp_plan(target, config)}；"
                f"q=({target.quaternion[0]:.4f}, {target.quaternion[1]:.4f}, "
                f"{target.quaternion[2]:.4f}, {target.quaternion[3]:.4f})；"
                f"{target.note()}"
            ),
            elapsed=0.0,
        )

    started = time.monotonic()
    hand.open()
    hand.settle_open()
    log.info("hand opened, approaching target")

    approach = move_waypoints(config, waypoints, get_bridge, stage_labels=labels)
    if not approach.reached:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=approach.timed_out,
            message=(
                "手已张开，接近阶段失败，机械臂保持原位（未闭合手）。"
                f"{approach.message}"
            ),
            elapsed=time.monotonic() - started,
            final=approach.final,
        )

    hand.close()
    hand.settle_close()
    close_note = (
        f"手已闭合（等待 {config.hand_close_settle_s:g}s；O6 完全到位约 10~15s，"
        "请现场观察是否夹住）"
    )
    log.info("hand closed at grasp pose")

    lift: Optional[MoveOutcome] = None
    if lift_waypoint is not None:
        lift = move_waypoints(config, [lift_waypoint], get_bridge, stage_labels=["lift"])
        if not lift.reached:
            return MoveOutcome(
                ok=True,
                reached=False,
                timed_out=lift.timed_out,
                message=(
                    f"已在目标位闭合手（{close_note}），但抬回 hover 失败："
                    f"{lift.message}；物体可能仍在手指间，请人工检查后再操作。"
                ),
                elapsed=time.monotonic() - started,
                final=lift.final or approach.final,
            )

    final: Optional[PoseSample] = (
        lift.final if lift is not None else approach.final
    )
    elapsed = time.monotonic() - started
    return MoveOutcome(
        ok=True,
        reached=True,
        timed_out=False,
        message=(
            f"抓取完成：张手 → hover → 下降到 "
            f"(x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f}) → {close_note}"
            + (
                f" → 抬回 hover (z={waypoints[0].z:.4f})"
                if lift_waypoint is not None
                else "（未抬起：approach_height_m=0）"
            )
            + f"；用时 {elapsed:.2f}s；{target.note()}"
        ),
        elapsed=elapsed,
        final=final,
    )


def place_target(
    config: SkillConfig,
    target: Target,
    get_bridge: Callable[[], RosBridge],
    hand,
) -> MoveOutcome:
    """hover → 下降到放置点 → 张手 → 抬起（place/release 的放置流程）。"""
    if target.quaternion is None:
        raise ConfigError("放置目标四元数未配置")
    waypoints = plan_waypoints(target, config.approach_height_m)
    labels = (
        ["approach(hover)", "descend"] if len(waypoints) > 1 else ["place"]
    )
    lift_waypoint: Optional[Target] = waypoints[0] if len(waypoints) > 1 else None
    if config.dry_run:
        plan = " → ".join(
            f"{label} (x={wp.x:.4f}, y={wp.y:.4f}, z={wp.z:.4f})"
            for label, wp in zip(labels, waypoints)
        )
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=False,
            message=(
                f"dry_run=true：只计算未下发。放置计划：{plan} → 张手"
                f"（等待 {config.hand_open_settle_s:g}s）"
                + (
                    f" → 抬回 hover (z={waypoints[0].z:.4f})"
                    if lift_waypoint is not None
                    else ""
                )
                + f"；{target.note()}"
            ),
            elapsed=0.0,
        )

    started = time.monotonic()
    approach = move_waypoints(config, waypoints, get_bridge, stage_labels=labels)
    if not approach.reached:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=approach.timed_out,
            message=(
                "放置阶段失败，机械臂保持原位；物体仍在手中（手未张开）。"
                f"{approach.message}"
            ),
            elapsed=time.monotonic() - started,
            final=approach.final,
        )

    hand.open()
    hand.settle_open()
    lift: Optional[MoveOutcome] = None
    if lift_waypoint is not None:
        lift = move_waypoints(config, [lift_waypoint], get_bridge, stage_labels=["lift"])
    elapsed = time.monotonic() - started
    if lift is not None and not lift.reached:
        return MoveOutcome(
            ok=True,
            reached=True,
            timed_out=False,
            message=(
                f"已在 (x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f}) "
                f"张开手放置，但抬回 hover 失败：{lift.message}；"
                "物体应已放下，请人工检查。"
            ),
            elapsed=elapsed,
            final=lift.final or approach.final,
        )
    final: Optional[PoseSample] = lift.final if lift is not None else approach.final
    return MoveOutcome(
        ok=True,
        reached=True,
        timed_out=False,
        message=(
            f"放置完成：hover → 下降到 "
            f"(x={target.x:.4f}, y={target.y:.4f}, z={target.z:.4f}) → 张手"
            f"（等待 {config.hand_open_settle_s:g}s）"
            + (
                f" → 抬回 hover (z={waypoints[0].z:.4f})"
                if lift_waypoint is not None
                else "（未抬起：approach_height_m=0）"
            )
            + f"；用时 {elapsed:.2f}s；{target.note()}"
        ),
        elapsed=elapsed,
        final=final,
    )


def release_hand(config: SkillConfig, hand) -> MoveOutcome:
    """原地张手，零机械臂运动。"""
    if config.dry_run:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=False,
            message=(
                f"dry_run=true：只计算未下发。会原地张开 {_hand_label(hand)}"
                f"（等待 {config.hand_open_settle_s:g}s），不移动机械臂。"
            ),
            elapsed=0.0,
        )
    started = time.monotonic()
    hand.open()
    hand.settle_open()
    return MoveOutcome(
        ok=True,
        reached=True,
        timed_out=False,
        message=(
            f"已原地张开 {_hand_label(hand)}"
            f"（等待 {config.hand_open_settle_s:g}s；O6 完全到位约 10~15s，"
            "请现场观察手指是否离开物体）"
        ),
        elapsed=time.monotonic() - started,
    )


def home_arm(
    config: SkillConfig,
    get_bridge: Callable[[], RosBridge],
) -> MoveOutcome:
    """关节空间移动到配置的 home_joints（手不动作）。"""
    if config.home_joints is None:
        raise ConfigError(
            "未配置 home_joints（7 个 rad）；请在 manifest 的 "
            "qmzs_grasp config 里设置后再调用 home"
        )
    if config.dry_run:
        return MoveOutcome(
            ok=True,
            reached=False,
            timed_out=False,
            message=(
                "dry_run=true：只计算未下发。home_joints = "
                f"{[round(float(v), 4) for v in config.home_joints]} rad；手不动作。"
            ),
            elapsed=0.0,
        )
    bridge = get_bridge()
    started = time.monotonic()
    result = bridge.move_joints(
        list(config.home_joints),
        timeout_s=config.motion_timeout_s,
        tolerance_rad=config.joint_tolerance_rad,
        feedback_timeout_s=config.feedback_timeout_s,
    )
    elapsed = time.monotonic() - started
    if result.get("reached"):
        return MoveOutcome(
            ok=True,
            reached=True,
            timed_out=False,
            message=(
                "已复位到 home_joints="
                f"{[round(float(v), 4) for v in config.home_joints]} rad，"
                f"用时 {elapsed:.2f}s；手未动作。"
            ),
            elapsed=elapsed,
            final=bridge.latest_pose(),
        )
    return MoveOutcome(
        ok=True,
        reached=False,
        timed_out=bool(result.get("timed_out")),
        message=(
            "home 未在超时内到位（timed_out）；检查 joint_command 话题、"
            "关节限位和 arm7aix_controller 日志。"
        ),
        elapsed=elapsed,
        final=bridge.latest_pose(),
    )
