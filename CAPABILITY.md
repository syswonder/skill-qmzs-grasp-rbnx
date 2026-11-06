---
description: ARM7AIX 七轴 + LinkerHand O6 抓取技能（参考 jaka_grasp）。高层：grab 固定位姿抓取（张手→两级接近→闭合→抬起）、pick 感知抓取（VLM 或像素定位）、release 原地张手、place 放下、home 关节复位；底层：pixel_to_pose/move_to_pixel/detect_object/move_to_object/calibration_status/snapshot。手眼标定像素→base_link XY + 人工 Z + 固定四元数；两级接近避免碰倒物体；手部经 o6_hand 原语 hand/move_joint 下发官方 open/fist 预设；VLM 需要 OpenAI 兼容多模态端点；标定文件自动重载。
---

# arm7aix_grasp — 七轴 + O6 抓取技能

Provider id `arm7aix_grasp`，namespace `robonix/skill/arm7aix_grasp`。LLM 工具名：

- 抓取流程：`grab` / `pick` / `release` / `place` / `home`
- 底层辅助：`pixel_to_pose` / `move_to_pixel` / `detect_object` /
  `move_to_object` / `calibration_status` / `snapshot`

技能不碰硬件：像素→XY 用固定相机的平面单应（本地 numpy/opencv 计算）；机械臂
运动只发布到 `arm7aix_controller` 原语的 `pos_command`（Pose）、`joint_command`
（JointState），从 `end_pose` / `joint_states` 读反馈判断到位；手部只经
`o6_hand` 原语的 `hand/move_joint` 下发 open/fist 预设。关节限位、七轴 IK、
轨迹规划、故障保护都在原语里。

## 坐标系与单位

- 位姿一律在 `base_link`，位置米，四元数 xyzw，参考点是法兰中心 `tool0`
  （不是灵巧手指尖）。
- 像素 `u, v` 必须是手眼标定的分辨率（标定模型 `camera.width × camera.height`，
  当前 640×480）；技能先按标定内参去畸变，再乘单应矩阵。
- `z` 是人工指定的法兰高度（米），也接受 `102.4mm` 写法。
- 四元数是人工指定的固定法兰姿态，`"qx,qy,qz,qw"`，非单位长度会被归一化。

## 几何前提

固定相机 + 桌面标定面：单应把“桌面上一点”的像素映射到该点在 base_link 的 XY。
机械臂法兰移动到该 XY 的**正上方**（Z 人工给）时 XY 成立，与法兰高度无关。
如果拿同一像素反推悬空物体表面，会因相机倾角产生视差 —— 那种场景需要深度/完整
外参，不在本技能范围内。

## 抓取流程

| 工具 | 行为 |
|---|---|
| `grab` | 固定位姿抓取：张手 → hover（`grasp_pose.z + approach_height_m`）→ 下降到 `grasp_pose` → 闭合手 → 等待 `hand_close_settle_s` → 抬回 hover。`grasp_pose: [x,y,z]` 来自 manifest；未配置时报错（请用 pick）。接近失败时手保持张开并停止。 |
| `pick` | 感知抓取：`pixel="u,v"`（标定分辨率）或 `object` + VLM 检测 → 位姿（固定 Z/四元数）→ 与 grab 相同的张手/两级接近/闭合/抬起。返回 `found` / `class_name` / 像素 / 位姿。 |
| `release` | 原地张开手（零机械臂运动）；物体留在当前位置。 |
| `place` | 放下：hover → 下降到 `place_pose`（未配置则当前 XY + 目标 Z 原地放下）→ 张手 → 抬回 hover。 |
| `home` | 关节空间移动到 `home_joints`（`joint_command`），等 `joint_states` 进入 `joint_tolerance_rad`；手不动作。未配置 `home_joints` 时报错。 |

## 底层辅助工具

| 工具 | 行为 |
|---|---|
| `pixel_to_pose` | `u, v` [+ `z`] [+ `quaternion`] → 返回 `[x, y, z, qx, qy, qz, qw]` 与去畸变像素。只算不动。四元数未配置时返回 ok=true 但 q 为 0 并提示。 |
| `move_to_pixel` | 同样的几何，然后**两级接近**：先到目标正上方 `approach_height_m` 的 hover，再下到目标；每段都发 `pos_command` 并等 `end_pose` 进入容差。返回 `reached` / `timed_out` / `elapsed_sec` / 最终实测位姿。`dry_run=true` 时只算不发。 |
| `detect_object` | 相机取一帧 → VLM 开放词汇检测自然语言描述的目标 → 有向框 → 抓取点像素 → 返回像素 + `[x, y, z, qx, qy, qz, qw]`。只算不动；保存标注 JPEG（`annotated_path`）。 |
| `move_to_object` | `detect_object` 的检测链路 + `move_to_pixel` 的两级接近运动链路，返回 `found` / `reached` / `timed_out`。 |
| `calibration_status` | 只读：标定是否加载、矩阵来源/路径、相机内参、覆盖区域、默认 Z/四元数来源、话题与反馈新鲜度。 |
| `snapshot` | （可选）从 `robonix/primitive/camera/rgb` 抓一帧存 JPEG，返回绝对路径与分辨率；没有相机原语时 ok=false，不影响其它工具。 |

所有工具同步阻塞；每个运动段在 `motion_timeout_s`（默认 30 s）内等到位。
同一时刻只允许一个抓取/运动序列（被占用时返回“另一个流程正在执行”）。

## 手部（LinkerHand O6）

- 通过 atlas 解析 `o6_hand` 的 `hand/info` + `hand/move_joint`（gRPC）；轴向按
  `hand/info` 的 name 匹配，不按顺序。
- open/fist 预设来自 linkerhand-python-sdk 的 O6 左手 `O6_positions.yaml`
  （raw 255=张开 → 契约归一化 `1-raw/255`，1=闭合）。可用
  `hand_open_values` / `hand_close_values` 覆盖（长度 6，[0,1]）。
- O6 运动较慢（完全到位约 10~15 s）；命令下发后按 `hand_open_settle_s` /
  `hand_close_settle_s` 等待，**没有位置反馈验证**（本技能不消费 get_state）。
  抓取是否成功请现场观察手指/物体；消息里会注明“unverified”式限制说明。
- `release` / `place` 只张开手；`grab` / `pick` 闭合手；`home` 不动手。

## VLM 检测（detect_object / move_to_object）

- 相机：经 atlas 解析 `robonix/primitive/camera/rgb`（或配置 `camera_topic`），
  取一帧 BGR；分辨率应等于标定分辨率（默认 640×480），不同则按比例缩放并告警。
- VLM：任意 OpenAI 兼容多模态端点（`vlm_base_url` / `vlm_api_key` / `vlm_model`，
  默认回退环境变量 `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL`）。请求把帧以
  base64 JPEG 发出，要求模型只返回 JSON：有向框中心/长边/短边/倾角（归一化）。
- 抓取点：单应只对标定平面（桌面）成立，框中心通常高于桌面，所以默认取
  “框中心 → 有向框底部中点”的插值（`vlm_contact_ratio`，默认 0.5）。相机近
  正俯视可用 0；有倾角时用 0.3~0.7，现场看 `annotated_path` 标注图配合
  `pixel_to_pose`/卷尺核对后再调。
- 标注图：默认保存到 `snapshot_dir`，返回 `annotated_path`（绿框 + 红叉抓取点），
  便于核对 VLM 是否框对、像素是否落在标定覆盖区域内。
- 局限：VLM 只给 2D 框，没有深度/尺度，Z 仍由人工指定；同一画面多个目标时
  取第一个（消息里给出类别列表）。不要把它当成精度保证，首次现场务必看标注图。

## 运动方式：两级接近（避免碰倒物体）

`move_to_pixel` / `move_to_object` 默认先移动到目标**正上方** `approach_height_m`
（默认 0.20 m）的 hover，再从 hover 下降到目标：

1. `hover = (x, y, z + approach_height_m)`，XY 与四元数不变；
2. hover 到位后才发第二段 `(x, y, z)`；hover 超时/失败立即停止，不继续下降。

这样 TCP 不会从侧面扫过物体把它碰走；`approach_height_m: 0` 退化为原来的单次
移动。两段都会检查 `workspace` 边界；hover 超出 `z_max` 会直接报错（调小高度或
调 `workspace`）。注意 `pos_command` 是单个 TCP 终点 + 关节空间轨迹，原语不保证
中间 TCP 路径是严格笛卡尔直线；但 hover → target 只差 Z（同 XY、同姿态），实际
下降基本竖直。需要严格 MoveL 需要原语额外暴露线性插值（当前没有）。

## 目标值优先级

| 值 | 优先级（高 → 低） |
|---|---|
| Z | 工具参数 `z`（仅 `allow_tool_overrides=true`）→ `orientation_path`（标定网页保存的 `target_z_mm`）→ `default_z_m` |
| 四元数 | 工具参数 `quaternion`（仅 `allow_tool_overrides=true`）→ `orientation_path` → `default_quaternion_xyzw` |

**默认 `allow_tool_overrides=false`**：工具调用里的 `z` / `quaternion` 会被忽略，
目标只认部署配置。这样即使上层模型（Pilot）自行猜测“垂直向下”的四元数或 Z，
也不会覆盖现场标定的固定姿态；需要按调用参数临时改目标时把该配置改成 `true`。

`orientation_path` 指向 handeye-calibration 的 `config/grasp_orientation.json`；
文件存在但内容非法时报错，不静默换姿态。标定模型与目标文件都按 mtime 自动重载。

## 安全与限制

- 不做碰撞检查、不约束中间路径；运动前确认机械臂周围与桌面安全。这不是硬件急停。
- 像素必须落在标定覆盖区域（`workspace_hull`）内，否则报错；覆盖不够回标定网页补点。
- 可选 `workspace`（base_link 边界）与 `max_travel_m`（单次最大位移）做粗保护，
  默认不限制；建议现场确认后打开。
- 目标不可达/被限位拒绝时 `pos_command` 不会到位，`move_to_pixel` 返回
  `timed_out`，需查 `rbnx-boot/logs/arm7aix_controller.log`。

## 依赖与配置

经 atlas 解析（可在 manifest `topics` 显式覆盖）：

| 契约 | 用途 | transport |
|---|---|---|
| `robonix/primitive/arm/pos_command` | 发目标位姿 | ROS 2 |
| `robonix/primitive/arm/end_pose` | 到位判定反馈 | ROS 2 |
| `robonix/primitive/arm/joint_states` | 存活检查 | ROS 2 |
| `robonix/primitive/camera/rgb` | snapshot（可选） | ROS 2 |

配置字段见 `config.spec`；部署 manifest 里已配置标定/目标路径与默认 Z/四元数。

## Lifecycle

skill：`rbnx boot` 停在 INACTIVE，executor 在第一次 MCP 调用时发 CMD_ACTIVATE。
`on_init` 解析配置与标定（不碰 ROS/atlas）；`on_activate` 解析话题并启动 rclpy 桥
（失败延后到工具调用时报错）；`on_deactivate` 停桥。
