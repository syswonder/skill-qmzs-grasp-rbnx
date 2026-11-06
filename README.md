# qmzs_grasp — ARM7AIX 七轴 + LinkerHand O6 抓取技能

Robonix 技能（skill）包，参考 [`robonix.skill.jaka_grasp`](https://packages.robonix.ai/packages/robonix.skill.jaka_grasp/)
组织。两部分：

- **高层抓取流程**：`grab` / `pick` / `release` / `place` / `home`
- **底层视觉/运动辅助**：`pixel_to_pose` / `move_to_pixel` / `detect_object` /
  `move_to_object` / `calibration_status` / `snapshot`

```
像素 (u,v) --手眼标定单应--> base_link XY（米）
  + 人工固定 Z / 四元数
  → 两级接近（hover → 下降）
  → O6 手 open/fist 预设
  → pos_command / end_pose 等到位
```

## 高层工具

| MCP 工具 | 说明 |
|---|---|
| `grab` | 固定位姿抓取：张手 → hover（`grasp_pose.z + approach_height_m`）→ 下降到 `grasp_pose` → 闭合手 → 等待 → 抬回 hover |
| `pick` | 感知抓取：`pixel="u,v"`（标定分辨率）或 VLM 检测 `object` → 抓取流程，返回 `found`/像素/位姿 |
| `release` | 原地张开 O6 手，零机械臂运动 |
| `place` | 放下：hover → 下降到 `place_pose`（未配置则原地放下）→ 张手 → 抬回 hover |
| `home` | 关节空间移动到 `home_joints`（`joint_command`），手不动作 |

## 底层辅助

| MCP 工具 | 说明 |
|---|---|
| `pixel_to_pose` | 像素 → 位姿，只算不运动（现场核对用） |
| `move_to_pixel` | 像素 → 两级接近运动（不控制手） |
| `detect_object` | VLM 找自然语言描述的目标 → 像素 + 位姿（不运动） |
| `move_to_object` | VLM 检测 + 两级接近运动（不控制手） |
| `calibration_status` | 标定/目标/手部/话题/反馈状态 |
| `snapshot` | 抓相机原语一帧存 JPEG |

## 依赖

- 标定模型：部署内 `config/handeye/calibration.json`（固定相机桌面平面单应 +
  相机内参 + 覆盖区域；由 handeye-calibration 导出）
- 目标设置：`config/handeye/grasp_orientation.json`（固定四元数 + 法兰 Z）
- arm 原语：`pos_command` / `end_pose` / `joint_states` / `joint_command`
- 手部原语：`o6_hand` 的 `hand/info` + `hand/move_joint`（官方 open/fist 预设）
- 相机原语（可选，snapshot/detect/pick 用）：`camera/rgb`
- VLM（可选）：OpenAI 兼容多模态端点，`vlm_*` 配置或环境变量 `VLM_*`

## 构建与启动

```bash
rbnx build -p skills/qmzs_grasp
# 或随部署一起：./build.sh
```

部署 manifest 的 `skill:` 段声明本包；boot 后停在 INACTIVE，第一次 MCP 调用时
由 executor 激活。配置字段见 [config.spec](config.spec)，接口语义见
[CAPABILITY.md](CAPABILITY.md)，完整现场说明见
[docs/qmzs_grasp.md](../../docs/qmzs_grasp.md)。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试全部离线（假 bridge / 假手 / 假 OpenAI 端点），不需要 ROS、相机或机械臂。

## License

MulanPSL-2.0；见 [LICENSE](LICENSE)。源文件（Python/Shell）带
`SPDX-License-Identifier: MulanPSL-2.0` 头。
