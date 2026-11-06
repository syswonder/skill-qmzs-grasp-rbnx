# arm7aix_grasp 技能运行配置（robonix_manifest.yaml 里 skill: - name: arm7aix_grasp
# 的 config 块，整段以 JSON 传给 on_init）。
#
# 本文件只说明字段与默认值；现场请按机械臂/标定结果覆盖。
config:
  # string, default "arm7aix_controller".
  # 通过 atlas 解析 arm 契约时钉住的 provider id（manifest 里 primitive 的 name）。
  arm_provider_id: arm7aix_controller

  # string, default "orbbec_camera".
  # snapshot 工具解析 robonix/primitive/camera/rgb 时钉住的相机 provider id。
  camera_provider_id: orbbec_camera

  # string, default "".
  # handeye-calibration 的标定模型 JSON（planar/calibration.json）路径：
  #   绝对路径，或相对部署根目录的路径。
  # 该文件包含 homography（去畸变像素 -> base_link XY，米）、camera（内参/分辨率）、
  # workspace_hull（标定覆盖区域）与 table_z_mm。为空时改用下面的 homography。
  # 文件存在时按 mtime 自动重载，标定网页重新求解后无需重启技能。
  calibration_path: ""

  # list[float], 长度 9（行优先）或 3x3 嵌套，default 空。
  # 直接配置平面单应矩阵：输入为“去畸变后的像素”，输出 base_link XY（米）。
  # 仅在 calibration_path 为空或文件不存在时使用；用于没有标定文件时手工填。
  homography: []

  # map, default 空。
  # 与 inline homography 配套的相机内参（去畸变用）：
  #   {width: 640, height: 480, fx: ..., fy: ..., cx: ..., cy: ..., coeffs: [...]}
  # 标定文件的 camera 字段会覆盖它。都没有时像素按“已去畸变”处理。
  camera_intrinsics: {}

  # list[float] (4), default 空。
  # 人工指定的固定法兰四元数 xyzw。为空且 orientation_path 也没有有效值时，
  # pixel_to_pose 仍会返回 XY（四元数填 0 并提示），但 move_to_pixel 会报错。
  default_quaternion_xyzw: []

  # float, default 0.1024.
  # 人工指定的目标法兰 Z（米，base_link）。为空/未配置时 move_to_pixel 必须在
  # 调用参数里给 z。可在调用参数里用 "102.4mm" 覆盖。
  default_z_m: 0.1024

  # string, default "".
  # 可选：标定网页保存的目标设置文件（config/grasp_orientation.json），内容：
  #   {"target_quaternion_xyzw": [qx,qy,qz,qw], "target_z_mm": 102.4}
  # 优先级：工具参数 > 该文件 > default_quaternion_xyzw / default_z_m。
  orientation_path: ""

  # ── 运动 ────────────────────────────────────────────────────────────
  # bool, default false. true = 只计算不发布（工具仍返回组合好的位姿）。
  dry_run: false

  # float, seconds, default 2.0. 下发前等待 end_pose 新鲜反馈的上限。
  feedback_timeout_s: 2.0

  # float, seconds, default 30.0. 发布后等待到位的上限。
  motion_timeout_s: 30.0

  # float, metres, default 0.005. 到位位置容差（与 end_pose 比较）。
  position_tolerance_m: 0.005

  # float, radians, default 0.03. 到位姿态容差（四元数夹角）。
  angle_tolerance_rad: 0.03

  # float, metres, default 0.0. 单次最大位移；0 = 不检查。
  # 用于防止像素/标定出错时的一次性大跨度运动，建议现场设为 0.4~0.6。
  max_travel_m: 0.0

  # float, metres, default 0.20. 两级接近高度：
  # 先移动到 (x, y, z + approach_height_m) 的正上方 hover，再从 hover 竖直
  # 下降到目标；hover 未到位则停止、不继续下降。0 = 关闭两级接近。
  # 机械臂/工作空间限制导致 hover 超界时会直接报错（调小该值或调 workspace）。
  approach_height_m: 0.20

  # bool, default false.
  # 是否允许工具调用参数 z/quaternion 覆盖配置目标。false 时（推荐）忽略这两个
  # 参数，目标只认 orientation_path / default_z_m / default_quaternion_xyzw，
  # 避免上层模型（Pilot）自行猜测姿态覆盖固定配置；true 时才以调用参数优先。
  allow_tool_overrides: false

  # map, default 空。base_link 工作空间边界，缺省键不限制：
  #   {x_min: -0.6, x_max: 0.6, y_min: -0.6, y_max: 0.6, z_min: 0.02, z_max: 0.8}
  workspace: {}

  # map, default 空。话题覆盖（一般不需要；正常经 atlas 解析）：
  #   {end_pose: /arm7aix/end_pose, pos_command: /arm7aix/pos_command,
  #    joint_states: /arm7aix/joint_states}
  topics: {}

  # string, default "". snapshot 用的话题覆盖；为空时经 atlas 解析 camera/rgb。
  camera_topic: ""

  # string, default "rbnx-build/data/snapshots"（相对包目录/部署根解析）。
  # snapshot 保存 JPEG 的目录。
  snapshot_dir: "rbnx-build/data/snapshots"

  # float, seconds, default 5.0. snapshot 等待一帧 RGB 的上限。
  snapshot_timeout_s: 5.0

  # ── VLM 开放词汇检测（detect_object / move_to_object）────────────────
  # string, default ""（回退环境变量 VLM_BASE_URL）。
  # OpenAI 兼容的多模态聊天端点，/chat/completions 会自动补全。
  vlm_base_url: ""

  # string, default ""（回退环境变量 VLM_API_KEY）。
  vlm_api_key: ""

  # string, default ""（回退环境变量 VLM_MODEL）。
  # 必须支持图片输入（多模态）。
  vlm_model: ""

  # float, seconds, default 60.0。单次 VLM 请求超时。
  vlm_timeout_s: 60.0

  # float, default 0.5，范围 [0,1]。
  # 抓取点在有向框“中心 → 底部中点”之间的插值：
  #   0 = 框中心（近正俯视相机；物体在画面上居中）；
  #   1 = 框底部中点（相机有倾角时更靠近桌面接触点）；
  #   0.5 = 折中（默认）。单应只对桌面平面成立，框中心通常高于桌面，
  #   所以有倾角时不要用 0；现场用 detect_object 的标注图核对后再调。
  vlm_contact_ratio: 0.5

  # bool, default true。检测时把有向框 + 抓取点画到帧上，保存 JPEG 到
  # snapshot_dir 并在 detect_object 返回 annotated_path，便于核对 VLM 是否框对。
  vlm_save_annotated: true

  # string, default ""。追加到检测提示词末尾的补充说明（例如“只找桌上的物体”）。
  vlm_prompt_extra: ""

  # ── 抓取流程（grab / pick / place / release / home）────────────────
  # string, default "o6_hand"。手部原语 provider id；技能经 atlas 消费
  # hand/info + hand/move_joint（gRPC），不直接开 CAN。
  hand_provider_id: o6_hand

  # float, seconds, default 1.0 / 3.0。手部 open/close 命令下发后等待手指动作
  # 的时间（O6 完全到位约 10~15 s，现场按实际响应调；没有位置反馈）。
  hand_open_settle_s: 1.0
  hand_close_settle_s: 3.0

  # list[float], 长度 6，default 空（用官方预设）。
  # 覆盖手部 open/close 的归一化目标 [0,1]（1=闭合）；一般不需要。
  hand_open_values: []
  hand_close_values: []

  # list[string], 长度 7，default joint1..joint7。
  # joint_command 的关节名顺序（与 arm7aix_controller 一致）。
  joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]

  # float, rad, default 0.02。home 的到位容差。
  joint_tolerance_rad: 0.02

  # list[float], 长度 7，rad，default 空。home 工具的关节目标；为空时 home 报错。
  home_joints: []

  # list[float], 长度 3，metres，default 空。grab 的固定抓取位姿 [x, y, z]
  # （base_link；姿态用固定四元数）。为空时 grab 报错，请用 pick 定位。
  grasp_pose: []

  # list[float], 长度 3，metres，default 空。place 的放置位姿 [x, y, z]。
  # 为空时 place 原地放下（当前末端 XY + 目标 Z）。
  place_pose: []
