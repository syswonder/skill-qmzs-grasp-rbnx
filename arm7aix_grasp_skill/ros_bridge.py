# SPDX-License-Identifier: MulanPSL-2.0
"""ROS 2 桥接：arm pos_command / end_pose 与（可选）相机 rgb。

- 命令侧：向 `geometry_msgs/Pose` 的 pos_command 话题发布目标位姿。原语
  （arm7aix_controller）以 reliable / depth 1 订阅该话题，这里保持一致。
- 反馈侧：订阅 end_pose（原语 best_effort 发布），用实测位姿判断是否到位；
  位姿/四元数都在 base_link 下，位置米、四元数 xyzw。
- 相机：订阅 sensor_msgs/Image（best_effort），只在 snapshot 工具用到。

节点是懒初始化的：pixel_to_pose 这类只算不动的工具不需要 ROS 也能工作。
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger("arm7aix_grasp.ros")


class RosBridgeError(RuntimeError):
    """ROS 环境不可用、没有反馈、话题未解析等（工具返回 ok=false）。"""


@dataclass(frozen=True)
class PoseSample:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    monotonic: float

    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def quaternion(self) -> tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    bgr: object  # numpy.ndarray (H, W, 3) uint8
    monotonic: float


def quaternion_angle(a, b) -> float:
    """两个 xyzw 四元数之间的最小夹角（弧度）。"""
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    dot = min(1.0, max(0.0, dot))
    return 2.0 * math.acos(dot)


class RosBridge:
    """一个 rclpy 节点 + 命令发布器 + 反馈/相机订阅器。"""

    def __init__(
        self,
        *,
        pos_command_topic: str,
        end_pose_topic: str,
        joint_states_topic: str = "",
        joint_command_topic: str = "",
        joint_names: Sequence[str] = (),
        camera_topic: str = "",
    ) -> None:
        self.pos_command_topic = pos_command_topic
        self.end_pose_topic = end_pose_topic
        self.joint_states_topic = joint_states_topic
        self.joint_command_topic = joint_command_topic
        self.joint_names = tuple(str(name) for name in joint_names)
        self.camera_topic = camera_topic

        self._lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._node = None
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._pub = None
        self._joint_pub = None
        self._camera_sub = None
        self._started = False

        self._pose: Optional[PoseSample] = None
        self._pose_count = 0
        self._joint_count = 0
        self._joint_state = None
        self._joint_state_t: Optional[float] = None
        self._frame: Optional[Frame] = None
        self._frame_count = 0
        self._cv_bridge = None

    # ── 生命周期 ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from geometry_msgs.msg import Pose
            from sensor_msgs.msg import Image, JointState
        except Exception as exc:  # noqa: BLE001
            raise RosBridgeError(
                f"ROS 2 Python 依赖不可用（先 source /opt/ros/humble/setup.bash）：{exc}"
            ) from exc

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            node = rclpy.create_node("arm7aix_grasp")
        except Exception as exc:  # noqa: BLE001
            raise RosBridgeError(f"初始化 rclpy 节点失败：{exc}") from exc

        command_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        feedback_qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        def _on_pose(msg) -> None:
            with self._lock:
                self._pose = PoseSample(
                    float(msg.position.x),
                    float(msg.position.y),
                    float(msg.position.z),
                    float(msg.orientation.x),
                    float(msg.orientation.y),
                    float(msg.orientation.z),
                    float(msg.orientation.w),
                    time.monotonic(),
                )
                self._pose_count += 1

        def _on_joint_state(msg) -> None:
            with self._lock:
                self._joint_state = msg
                self._joint_state_t = time.monotonic()
                self._joint_count += 1

        def _on_image(msg) -> None:
            try:
                bgr = self._decode_image(msg)
            except Exception as exc:  # noqa: BLE001
                log.warning("camera frame decode failed: %s", exc)
                return
            if bgr is None:
                return
            with self._lock:
                self._frame = Frame(
                    width=int(msg.width),
                    height=int(msg.height),
                    bgr=bgr,
                    monotonic=time.monotonic(),
                )
                self._frame_count += 1

        self._pub = node.create_publisher(Pose, self.pos_command_topic, command_qos)
        if self.joint_command_topic:
            self._joint_pub = node.create_publisher(
                JointState, self.joint_command_topic, command_qos
            )
        node.create_subscription(Pose, self.end_pose_topic, _on_pose, feedback_qos)
        if self.joint_states_topic:
            node.create_subscription(
                JointState, self.joint_states_topic, _on_joint_state, feedback_qos
            )
        if self.camera_topic:
            self._camera_sub = node.create_subscription(
                Image, self.camera_topic, _on_image, feedback_qos
            )

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(
            target=executor.spin, name="pixel-move-ros", daemon=True
        )
        spin_thread.start()

        self._node = node
        self._executor = executor
        self._spin_thread = spin_thread
        self._started = True
        log.info(
            "ROS 2 ready: pos_command=%s end_pose=%s camera=%s",
            self.pos_command_topic,
            self.end_pose_topic,
            self.camera_topic or "(none)",
        )

    def stop(self) -> None:
        node, executor, spin_thread = self._node, self._executor, self._spin_thread
        self._node = self._executor = self._spin_thread = None
        self._started = False
        self._pub = None
        self._joint_pub = None
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        if spin_thread is not None:
            try:
                spin_thread.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass

    def set_camera_topic(self, topic: str) -> None:
        """on_activate 解析出相机话题后再订阅（需在 start 之前调用）。"""
        self.camera_topic = topic

    # ── 反馈 ────────────────────────────────────────────────────────
    def latest_pose(self) -> Optional[PoseSample]:
        with self._lock:
            return self._pose

    def wait_fresh_feedback(self, timeout_s: float) -> bool:
        """等待 end_pose 至少来一帧（用于下发前的存活检查）。"""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                if self._pose is not None:
                    return True
            time.sleep(0.02)
        with self._lock:
            return self._pose is not None

    def feedback_age(self) -> Optional[float]:
        with self._lock:
            pose = self._pose
        if pose is None:
            return None
        return max(0.0, time.monotonic() - pose.monotonic)

    # ── 运动 ────────────────────────────────────────────────────────
    def move_pose(
        self,
        *,
        x: float,
        y: float,
        z: float,
        quaternion: tuple[float, float, float, float],
        timeout_s: float,
        position_tolerance_m: float,
        angle_tolerance_rad: float,
        feedback_timeout_s: float = 2.0,
    ) -> dict:
        """发布一个 TCP 终点位姿目标，并等待 end_pose 进入容差。

        返回 dict(reached, timed_out, elapsed, final, feedback_before_publish)。
        前置条件不满足（ROS 未初始化 / 没有反馈）时抛 RosBridgeError。
        """
        if not self._started or self._pub is None:
            raise RosBridgeError("ROS 桥未启动")
        if not self.wait_fresh_feedback(feedback_timeout_s):
            raise RosBridgeError(
                f"{feedback_timeout_s:.1f}s 内没有收到 {self.end_pose_topic} 反馈；"
                "确认 arm7aix_controller 已激活、话题名一致、ROS 2 环境已 source"
            )

        try:
            from geometry_msgs.msg import Pose
        except Exception as exc:  # noqa: BLE001
            raise RosBridgeError(f"geometry_msgs 不可用：{exc}") from exc

        target_q = tuple(float(v) for v in quaternion)
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.orientation.x = target_q[0]
        msg.orientation.y = target_q[1]
        msg.orientation.z = target_q[2]
        msg.orientation.w = target_q[3]

        with self._motion_lock:
            with self._lock:
                pose_count_before = self._pose_count
            started = time.monotonic()
            self._pub.publish(msg)
            log.info(
                "pos_command -> (%.4f, %.4f, %.4f) q=(%.4f, %.4f, %.4f, %.4f)",
                x, y, z, *target_q,
            )
            deadline = started + max(0.0, timeout_s)
            last: Optional[PoseSample] = None
            feedback_seen = False
            while True:
                with self._lock:
                    pose = self._pose
                    count = self._pose_count
                if pose is not None and count > pose_count_before:
                    feedback_seen = True
                    last = pose
                    position_error = math.dist(
                        (x, y, z), (pose.x, pose.y, pose.z)
                    )
                    angle_error = quaternion_angle(target_q, pose.quaternion())
                    if (
                        position_error <= position_tolerance_m
                        and angle_error <= angle_tolerance_rad
                    ):
                        elapsed = time.monotonic() - started
                        return {
                            "reached": True,
                            "timed_out": False,
                            "elapsed": elapsed,
                            "final": pose,
                            "feedback_before_publish": True,
                        }
                if time.monotonic() >= deadline:
                    elapsed = time.monotonic() - started
                    if last is None:
                        with self._lock:
                            last = self._pose
                    return {
                        "reached": False,
                        "timed_out": True,
                        "elapsed": elapsed,
                        "final": last,
                        "feedback_before_publish": True,
                        "feedback_seen": feedback_seen,
                    }
                time.sleep(0.02)

    # ── 关节命令（home 等 joint-space 移动）──────────────────────────
    def wait_fresh_joint_feedback(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_state is not None:
                    return True
            time.sleep(0.02)
        with self._lock:
            return self._joint_state is not None

    def latest_joint_positions(self) -> Optional[tuple[float, ...]]:
        with self._lock:
            message = self._joint_state
        if message is None:
            return None
        mapping = (
            dict(zip(message.name, message.position))
            if getattr(message, "name", None)
            else dict(zip(self.joint_names, message.position))
        )
        positions = []
        for name in self.joint_names:
            value = mapping.get(name)
            if value is None:
                return None
            positions.append(float(value))
        return tuple(positions)

    def move_joints(
        self,
        targets: Sequence[float],
        *,
        timeout_s: float,
        tolerance_rad: float,
        feedback_timeout_s: float = 2.0,
    ) -> dict:
        """发布七轴位置目标并等 joint_states 进入容差（home 用）。"""
        if not self._started or self._joint_pub is None:
            raise RosBridgeError("joint_command 未配置或 ROS 桥未启动")
        values = [float(value) for value in targets]
        if len(values) != len(self.joint_names):
            raise RosBridgeError(
                f"关节目标有 {len(values)} 个，期望 {len(self.joint_names)} 个"
            )
        if not self.wait_fresh_joint_feedback(feedback_timeout_s):
            raise RosBridgeError(
                f"{feedback_timeout_s:.1f}s 内没有收到 {self.joint_states_topic} 反馈；"
                "确认 arm7aix_controller 已激活、话题名一致"
            )
        with self._motion_lock:
            with self._lock:
                count_before = self._joint_count
            started = time.monotonic()
            try:
                from sensor_msgs.msg import JointState
            except Exception as exc:  # noqa: BLE001
                raise RosBridgeError(f"sensor_msgs 不可用：{exc}") from exc
            message = JointState()
            message.name = list(self.joint_names)
            message.position = values
            self._joint_pub.publish(message)
            deadline = started + max(0.0, timeout_s)
            last: Optional[tuple[float, ...]] = None
            while True:
                positions = self.latest_joint_positions()
                with self._lock:
                    count = self._joint_count
                if positions is not None and count > count_before:
                    last = positions
                    if max(
                        abs(current - target)
                        for current, target in zip(positions, values)
                    ) <= tolerance_rad:
                        return {
                            "reached": True,
                            "timed_out": False,
                            "elapsed": time.monotonic() - started,
                            "final": positions,
                        }
                if time.monotonic() >= deadline:
                    return {
                        "reached": False,
                        "timed_out": True,
                        "elapsed": time.monotonic() - started,
                        "final": last,
                    }
                time.sleep(0.02)

    # ── 相机 ────────────────────────────────────────────────────────
    def _decode_image(self, msg):
        """sensor_msgs/Image -> BGR ndarray；优先 cv_bridge，缺省时手工解码。"""
        try:
            if self._cv_bridge is None:
                import cv_bridge
                self._cv_bridge = cv_bridge.CvBridge()
            return self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001 - 退回手工解码
            self._cv_bridge = None
        import numpy as np

        encoding = str(getattr(msg, "encoding", "")).lower()
        height, width, step = int(msg.height), int(msg.width), int(msg.step)
        if height <= 0 or width <= 0:
            return None
        buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if encoding in ("rgb8", "bgr8"):
            rows = buffer.reshape(height, step)
            image = rows[:, : width * 3].reshape(height, width, 3)
            if encoding == "rgb8":
                image = image[:, :, ::-1]
            return np.ascontiguousarray(image)
        if encoding in ("mono8", "8uc1"):
            rows = buffer.reshape(height, step)
            gray = rows[:, :width]
            return np.ascontiguousarray(np.dstack([gray, gray, gray]))
        log.debug("unsupported image encoding for manual decode: %s", encoding)
        return None

    def wait_frame(self, timeout_s: float, after_monotonic: float = 0.0) -> Optional[Frame]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                frame = self._frame
            if frame is not None and frame.monotonic >= after_monotonic:
                return frame
            time.sleep(0.02)
        with self._lock:
            frame = self._frame
        if frame is not None and frame.monotonic >= after_monotonic:
            return frame
        return None

    def save_frame(self, path) -> dict:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            raise RosBridgeError(f"保存 JPEG 需要 opencv：{exc}") from exc
        frame = self.wait_frame(0.0)
        if frame is None:
            raise RosBridgeError("没有收到相机画面")
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), frame.bgr):
            raise RosBridgeError(f"写入图片失败：{target}")
        return {
            "path": str(target),
            "width": frame.width,
            "height": frame.height,
            "bgr": frame.bgr,
        }
