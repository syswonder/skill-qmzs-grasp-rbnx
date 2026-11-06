# SPDX-License-Identifier: MulanPSL-2.0
"""LinkerHand O6 手部控制：经 robonix/primitive/hand/* gRPC 契约下发官方预设。

预设来自 linkerhand-python-sdk 的 `LinkerHand/config/O6_positions.yaml`（左手）：
    raw 0..255，255 = 完全张开；契约归一化 [0,1] 且极性相反（1 = 闭合）。
因此 normalized = 1 - raw/255：

    open: raw (255, 179, 255, 255, 255, 255) -> (0, 0.298039, 0, 0, 0, 0)
    fist: raw ( 67, 151,   0,   0,   0,   0) -> (0.737255, 0.407843, 1, 1, 1, 1)

轴顺序与 o6_hand 原语的 hand/info 一致；实际下发按 axis name 匹配，不依赖顺序。
O6 运动较慢（完全到位约 10~15 s）；这里下发后按 settle 时间等待，不阻塞读反馈
（原语的 get_state 是 package-local 扩展，未纳入本技能的消费契约）。
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger("qmzs_grasp.hand")

AXIS_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)

RAW_OPEN = (255, 179, 255, 255, 255, 255)  # 左手 张开（255=张开）
RAW_FIST = (67, 151, 0, 0, 0, 0)           # 左手 握拳（0=闭合）


class HandError(RuntimeError):
    """手部原语不可用、轴不匹配或 move_joint 被拒绝。"""


def normalized_from_raw(raw_values: Sequence[float]) -> tuple[float, ...]:
    """raw 0..255（255=张开）-> 契约归一化 [0,1]（1=闭合）。"""
    return tuple(round(1.0 - float(value) / 255.0, 6) for value in raw_values)


def preset_values(
    raw_values: Sequence[float], axis_names: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    """按 axis name 把官方预设 raw 值映射成 [(name, normalized), ...]。

    `axis_names` 来自 hand/info；必须恰好是 O6 的 6 个已知轴（顺序任意）。
    """
    names = tuple(axis_names)
    if len(names) != len(AXIS_NAMES):
        raise HandError(
            f"手部报告 {len(names)} 个轴，O6 预设需要 {len(AXIS_NAMES)} 个：{names}"
        )
    if set(names) != set(AXIS_NAMES):
        raise HandError(
            "手部轴向与本技能内置的 O6 预设不一致，拒绝下发；"
            f"报告轴={names}，期望={AXIS_NAMES}"
        )
    by_name = {
        axis: round(1.0 - float(raw) / 255.0, 6)
        for axis, raw in zip(AXIS_NAMES, raw_values)
    }
    return tuple((name, by_name[name]) for name in names)


@dataclass
class PrimitiveHand:
    """O6 手部适配器：open() / close() 下发官方预设，settle_* 等待手指动作。"""

    move_stub: object
    pb2: object
    axis_names: tuple[str, ...]
    open_values: tuple[tuple[str, float], ...]
    close_values: tuple[tuple[str, float], ...]
    open_settle_s: float = 1.0
    close_settle_s: float = 3.0

    def _send(self, values: Sequence[tuple[str, float]], name: str) -> None:
        targets = [
            self.pb2.JointValue(name=axis, value=float(value))
            for axis, value in values
        ]
        request = self.pb2.MoveJoint_Request(targets=targets)
        # 与 linker_hand_control 服务一致：重复一帧更稳（O6 接受单帧，但偶发丢帧）。
        for _ in range(2):
            response = self.move_stub.MoveJoint(request)
            if not response.ok:
                raise HandError(f"o6 move_joint 拒绝（{name}）：{response.message}")
            time.sleep(0.05)
        log.info("hand %s sent: %s", name, [(a, round(v, 3)) for a, v in values])

    def open(self) -> None:
        self._send(self.open_values, "open")

    def close(self) -> None:
        self._send(self.close_values, "close")

    def settle_open(self) -> None:
        time.sleep(max(0.0, float(self.open_settle_s)))

    def settle_close(self) -> None:
        time.sleep(max(0.0, float(self.close_settle_s)))


def connect_hand(
    skill,
    provider_id: str,
    *,
    open_values_override: Optional[Sequence[float]] = None,
    close_values_override: Optional[Sequence[float]] = None,
    open_settle_s: float = 1.0,
    close_settle_s: float = 3.0,
) -> PrimitiveHand:
    """经 atlas 连接 o6_hand 的 hand/info + hand/move_joint gRPC 契约。"""
    import grpc
    from robonix_api import ATLAS
    from robonix_api.atlas_types import Transport

    # codegen 由 robonix_api 在 import 时加入 sys.path（见 main.py 的 import 顺序）
    import hand_pb2  # type: ignore
    import robonix_contracts_pb2_grpc as cg  # type: ignore

    def _stub(contract: str, stub_cls):
        capabilities = ATLAS.find_capability(
            contract_id=contract, transport=Transport.GRPC, provider_id=provider_id
        )
        capabilities = [
            cap for cap in capabilities if cap.provider_kind.name == "PRIMITIVE"
        ]
        if not capabilities:
            raise HandError(
                f"没有可用的 {provider_id} 手部原语（{contract} gRPC）"
            )
        channel = skill.connect_capability(
            capabilities[0], contract, Transport.GRPC
        )
        return stub_cls(grpc.insecure_channel(channel.endpoint))

    info_stub = _stub("robonix/primitive/hand/info", cg.RobonixPrimitiveHandInfoStub)
    info = info_stub.GetHandInfo(hand_pb2.GetHandInfo_Request())
    if not info.ok:
        raise HandError(f"hand/info 失败：{info.message}")
    axis_names = tuple(joint.name for joint in info.joints)
    if open_values_override is not None:
        if len(open_values_override) != len(axis_names):
            raise HandError(
                f"hand_open_values 有 {len(open_values_override)} 个值，手部有 "
                f"{len(axis_names)} 个轴"
            )
        open_values = tuple(
            (name, float(value))
            for name, value in zip(axis_names, open_values_override)
        )
    else:
        open_values = preset_values(RAW_OPEN, axis_names)
    if close_values_override is not None:
        if len(close_values_override) != len(axis_names):
            raise HandError(
                f"hand_close_values 有 {len(close_values_override)} 个值，手部有 "
                f"{len(axis_names)} 个轴"
            )
        close_values = tuple(
            (name, float(value))
            for name, value in zip(axis_names, close_values_override)
        )
    else:
        close_values = preset_values(RAW_FIST, axis_names)

    move_stub = _stub(
        "robonix/primitive/hand/move_joint",
        cg.RobonixPrimitiveHandMoveJointStub,
    )
    hand = PrimitiveHand(
        move_stub=move_stub,
        pb2=hand_pb2,
        axis_names=axis_names,
        open_values=open_values,
        close_values=close_values,
        open_settle_s=open_settle_s,
        close_settle_s=close_settle_s,
    )
    log.info(
        "hand connected: provider=%s axes=%s", provider_id, axis_names
    )
    return hand
