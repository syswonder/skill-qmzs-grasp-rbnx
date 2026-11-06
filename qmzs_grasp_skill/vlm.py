# SPDX-License-Identifier: MulanPSL-2.0
"""VLM 开放词汇检测：自然语言目标 → 图像像素（有向框 + 抓取点）。

与 `robonix.skill.vertical_grasp_object` 的 `detect_objects` 同一思路：把相机
一帧 RGB 以 base64 JPEG 发给任意 OpenAI 兼容的多模态端点，要求模型只返回
JSON（有向框：中心/长边/短边/倾角）。技能再用标定单应把抓取点换算成 base_link
位姿。

抓取点：单应只对标定平面（桌面）上的点成立，所以用“有向框底部中点”而不是
框中心 —— 框中心在物体可见投影的中部（高于桌面）。`contact_ratio` 在
框中心（0）和底部中点（1）之间插值；相机有倾角时取 0.5 左右，近正俯视时取 0。

这里没有深度：VLM 只给 2D 框，Z 仍由人工指定（与 qmzs_grasp 的其它工具一致）。
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("qmzs_grasp.vlm")


class VlmError(RuntimeError):
    """VLM 端点不可用、返回无法解析、未配置等（工具返回 ok=false）。"""


# 提示词沿用 vertical_grasp_object/roboarm 的 user_instruction_prompt 结构：
# 归一化坐标、有向框（长边/短边/倾角），只允许输出 JSON。
_DETECT_PROMPT = """你是一个智能机械臂的视觉控制中枢。请根据用户的自然语言指令，分析图片场景，\
找出机械臂需要抓取的目标物体，并用“有向边界框”标出。有向框要贴合物体本身的朝向：\
box_width 是物体较长的一边、box_height 是较短的一边，box_rotation_deg 是物体长边相对图像水平方向的倾角。

坐标系定义：图片左上角为原点 (0, 0)，向右为 x 轴 (0→1)，向下为 y 轴 (0→1)。\
box_center/width/height 都用相对整幅图的归一化比例表示，范围 0~1。机械臂底座通常位于图片下方（y 坐标较大）。

旋转角定义：box_rotation_deg 表示物体长边相对图像“水平向右”方向的夹角，单位度，范围 -90~90。\
0 表示长边水平；正值表示长边从水平位置顺时针（向下）旋转；接近 ±90 表示长边竖直。方形/圆形物体没有明显长边时填 0。

用户指令可能包含颜色/形状特征、相对距离（如“最近的”）或空间位置（如“最右边的”“中间的”）。\
指令来自语音转写，可能有歧义，请结合画面合理推断。若有多个符合条件的物体，任选其一即可。

请只输出一个纯净的 JSON 对象（不要 Markdown 代码块、不要多余文字），字段如下：
- "failed"：布尔值。未找到符合条件的目标时为 true，此时 "objects" 为空数组。
- "objects"：数组，每个元素代表一个目标物体，包含：
  - "class_name"：字符串，物体的具体类别，须包含颜色和形状信息，若有文字也一并写入（如“红色圆柱形杯子”）。
  - "box_center_x"：有向框中心 x，0~1，保留 3 位小数。
  - "box_center_y"：有向框中心 y，0~1，保留 3 位小数。
  - "box_width"：物体较长一边的长度，0~1，保留 3 位小数。
  - "box_height"：物体较短一边的长度，0~1，保留 3 位小数。
  - "box_rotation_deg"：物体长边倾角，-90~90，保留 1 位小数。

示例（用户指令“抓最右边的蓝色积木”，方块无明显长边）：
{"failed": false, "objects": [{"class_name": "蓝色方形积木", "box_center_x": 0.821, "box_center_y": 0.131, "box_width": 0.09, "box_height": 0.088, "box_rotation_deg": 0.0}]}

示例（用户指令“抓那支斜放的笔”，笔身从左下到右上倾斜）：
{"failed": false, "objects": [{"class_name": "黑色长杆钢笔", "box_center_x": 0.42, "box_center_y": 0.55, "box_width": 0.30, "box_height": 0.05, "box_rotation_deg": -35.0}]}

示例（用户指令“抓黄色的球”，画面中没有）：
{"failed": true, "objects": []}

现在，请根据图片执行任务。用户指令："{instruction}"
"""


def _extract_json(text: str) -> str:
    """模型可能把 JSON 包在 Markdown 代码块里，剥掉。"""
    match = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S)
    return match.group(1).strip() if match else text.strip()


def _chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    return url


def _post_vlm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout_s: float,
) -> str:
    """OpenAI 兼容多模态 chat 请求（stdlib urllib，不引入额外依赖）。"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        _chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "OpenAI/Python/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001
            pass
        raise VlmError(f"VLM HTTP {exc.code}：{detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise VlmError(f"VLM 端点不可达：{exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise VlmError(f"VLM 请求失败：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise VlmError(f"VLM 响应不是合法 JSON：{exc}") from exc
    choices = body.get("choices") or []
    if not choices:
        raise VlmError(f"VLM 没有返回 choices：{str(body)[:300]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # 少数端点返回 content 数组
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise VlmError("VLM 返回内容为空")
    return content


@dataclass(frozen=True)
class Detection:
    """归一化有向框（0~1，图像坐标：左上原点、向右 x、向下 y）。"""

    class_name: str
    center_x: float
    center_y: float
    width: float
    height: float
    rotation_deg: float = 0.0

    def corners(
        self, image_width: int, image_height: int
    ) -> list[tuple[float, float]]:
        """有向框四个角点（像素坐标，顺时针：左上/右上/右下/左下按旋转后顺序）。"""
        cx, cy = self.center_x * image_width, self.center_y * image_height
        half_w = self.width * image_width / 2.0
        half_h = self.height * image_height / 2.0
        theta = math.radians(self.rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        points: list[tuple[float, float]] = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            dx, dy = sx * half_w, sy * half_h
            points.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
        return points

    def contact_pixel(
        self,
        image_width: int,
        image_height: int,
        *,
        ratio: float = 0.5,
    ) -> tuple[float, float]:
        """物体与桌面的接触点像素：框中心 → 有向框底部中点之间按 ratio 插值。"""
        ratio = min(1.0, max(0.0, float(ratio)))
        corners = self.corners(image_width, image_height)
        center = (self.center_x * image_width, self.center_y * image_height)
        # 图像 v 越大越靠下：取 v 最大的两个角点，其中点即底部中点。
        bottom = sorted(corners, key=lambda point: point[1])[-2:]
        bottom_center = (
            (bottom[0][0] + bottom[1][0]) / 2.0,
            (bottom[0][1] + bottom[1][1]) / 2.0,
        )
        return (
            center[0] + ratio * (bottom_center[0] - center[0]),
            center[1] + ratio * (bottom_center[1] - center[1]),
        )


def _finite01(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VlmError(f"VLM 框字段 {name} 不是数值：{value!r}") from exc
    if not math.isfinite(result):
        raise VlmError(f"VLM 框字段 {name} 不是有限数值：{value!r}")
    return result


def parse_detections(payload: dict) -> list[Detection]:
    """解析 VLM 的 JSON 回复；`failed=true` / 空数组返回空列表。"""
    if not isinstance(payload, dict):
        raise VlmError(f"VLM 回复不是 JSON 对象：{str(payload)[:200]}")
    if payload.get("failed") is True:
        return []
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise VlmError(f"VLM 回复缺少 objects 数组：{str(payload)[:200]}")
    detections: list[Detection] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        try:
            center_x = _finite01(item.get("box_center_x"), "box_center_x")
            center_y = _finite01(item.get("box_center_y"), "box_center_y")
            width = _finite01(item.get("box_width"), "box_width")
            height = _finite01(item.get("box_height"), "box_height")
            rotation = _finite01(item.get("box_rotation_deg", 0.0), "box_rotation_deg")
        except VlmError as exc:
            # 单个框字段坏掉不影响其它目标；整体结构错误（objects 不是数组）才抛。
            log.warning("忽略 VLM 返回的非法框：%s", exc)
            continue
        if width <= 0 or height <= 0:
            continue
        if not (-0.05 <= center_x <= 1.05 and -0.05 <= center_y <= 1.05):
            continue
        detections.append(
            Detection(
                class_name=str(item.get("class_name") or "object")[:80],
                center_x=min(1.0, max(0.0, center_x)),
                center_y=min(1.0, max(0.0, center_y)),
                width=min(1.0, max(1e-4, width)),
                height=min(1.0, max(1e-4, height)),
                rotation_deg=rotation,
            )
        )
    return detections


def detect_objects(
    image_bgr,
    instruction: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float = 60.0,
    prompt_extra: str = "",
) -> list[Detection]:
    """一帧 BGR 图 + 自然语言指令 → 检测结果（空列表 = 没找到）。"""
    instruction = (instruction or "").strip()
    if not instruction:
        raise VlmError("目标描述为空；请说明要找什么物体")
    if not base_url:
        raise VlmError(
            "VLM 未配置：设置 config vlm_base_url 或环境变量 VLM_BASE_URL"
        )
    if not model:
        raise VlmError("VLM 未配置：设置 config vlm_model 或环境变量 VLM_MODEL")
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise VlmError(f"编码图像需要 opencv：{exc}") from exc
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise VlmError("JPEG 编码失败")
    image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    prompt = _DETECT_PROMPT.replace("{instruction}", instruction)
    if prompt_extra:
        prompt += "\n补充说明：" + str(prompt_extra)
    content = _post_vlm(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_b64=image_b64,
        timeout_s=timeout_s,
    )
    raw = _extract_json(content)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VlmError(f"VLM 回复不是合法 JSON：{raw[:300]}") from exc
    detections = parse_detections(payload)
    log.info(
        "VLM 检测到 %d 个目标（instruction=%r）",
        len(detections),
        instruction,
    )
    return detections


def draw_detections(image_bgr, detections: list[Detection], contacts: list[tuple[float, float]]):
    """在帧上画有向框、类别和抓取点（用于现场核对 VLM 是否框对）。"""
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return image_bgr
    canvas = np.array(image_bgr, copy=True)
    height, width = canvas.shape[:2]
    for detection, contact in zip(detections, contacts):
        corners = np.array(
            detection.corners(width, height), dtype=np.int32
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [corners], True, (0, 255, 0), 2)
        cv2.drawMarker(
            canvas,
            (int(round(contact[0])), int(round(contact[1]))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.putText(
            canvas,
            detection.class_name[:30],
            (int(corners[:, 0, 0].min()), max(12, int(corners[:, 0, 1].min()) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        # 抓取点坐标写右侧，方便对照
        cv2.putText(
            canvas,
            f"({contact[0]:.0f},{contact[1]:.0f})",
            (int(contact[0]) + 8, int(contact[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
