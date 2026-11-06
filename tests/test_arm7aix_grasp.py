# SPDX-License-Identifier: MulanPSL-2.0
"""arm7aix_grasp 离线回归测试：标定解析、去畸变/单应投影、配置解析、缓存重载。

不依赖 ROS、相机或机械臂。运行：
    .venv/bin/python -m unittest tests.test_arm7aix_grasp -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # package/repo root
sys.path.insert(0, str(ROOT))

from arm7aix_grasp_skill.calibration import (  # noqa: E402
    Calibration,
    CalibrationCache,
    CalibrationError,
    Projection,
    read_orientation_file,
)
from arm7aix_grasp_skill.config import (  # noqa: E402
    ConfigError,
    SkillConfig,
    check_workspace,
    parse_metres,
    parse_quaternion,
)
from arm7aix_grasp_skill.grasp import (  # noqa: E402
    grasp_target,
    home_arm,
    place_target,
    release_hand,
)
from arm7aix_grasp_skill.hand import (  # noqa: E402
    AXIS_NAMES,
    HandError,
    normalized_from_raw,
    preset_values,
)
from arm7aix_grasp_skill.motion import perform_move, plan_waypoints  # noqa: E402
from arm7aix_grasp_skill.ros_bridge import PoseSample, quaternion_angle  # noqa: E402
from arm7aix_grasp_skill.target import Target, TargetResolver  # noqa: E402
from arm7aix_grasp_skill.vlm import (  # noqa: E402
    Detection,
    VlmError,
    detect_objects,
    parse_detections,
)


CAMERA = {
    "width": 640,
    "height": 480,
    "fx": 600.0,
    "fy": 600.0,
    "cx": 320.0,
    "cy": 240.0,
    "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
}

# pixel (320,240) -> (0,0) m; 1 px = 1 mm, v 轴方向取反（仅测试用的人为映射）。
HOMOGGRAPHY = [
    [0.001, 0.0, -0.32],
    [0.0, -0.001, 0.24],
    [0.0, 0.0, 1.0],
]


def _model(homography=None, camera=None, hull=None):
    return {
        "type": "fixed_camera_tabletop_homography",
        "homography": homography if homography is not None else HOMOGGRAPHY,
        "camera": camera if camera is not None else dict(CAMERA),
        "workspace_hull": hull,
        "table_z_mm": 123.4,
    }


class CalibrationProjectionTests(unittest.TestCase):
    def test_pixel_without_distortion_maps_through_homography(self):
        calibration = Calibration.from_model(_model(), source="test")
        projection = calibration.project(320, 240)
        self.assertAlmostEqual(projection.x, 0.0, places=9)
        self.assertAlmostEqual(projection.y, 0.0, places=9)
        self.assertEqual((projection.undistorted_u, projection.undistorted_v), (320.0, 240.0))

        projection = calibration.project(420, 140)
        self.assertAlmostEqual(projection.x, 0.1, places=9)
        self.assertAlmostEqual(projection.y, 0.1, places=9)

    def test_nonzero_distortion_changes_projection(self):
        raw = Calibration.from_model(_model(), source="raw")
        distorted_model = _model(camera={**CAMERA, "coeffs": [0.2, -0.1, 0.0, 0.0, 0.05]})
        distorted = Calibration.from_model(distorted_model, source="distorted")
        a = raw.project(100, 80)
        b = distorted.project(100, 80)
        self.assertGreater(abs(a.x - b.x) + abs(a.y - b.y), 1e-6)
        self.assertNotEqual((a.undistorted_u, a.undistorted_v), (b.undistorted_u, b.undistorted_v))

    def test_workspace_hull_rejects_pixels_outside_calibrated_area(self):
        hull = [[300, 200], [340, 200], [340, 280], [300, 280]]
        calibration = Calibration.from_model(_model(hull=hull), source="test")
        inside = calibration.project(320, 240)
        self.assertTrue(math.isfinite(inside.x))
        with self.assertRaises(CalibrationError):
            calibration.project(10, 10)

    def test_inline_homography_without_camera_treats_pixels_as_undistorted(self):
        calibration = Calibration.from_inline([*HOMOGGRAPHY[0], *HOMOGGRAPHY[1], *HOMOGGRAPHY[2]])
        projection = calibration.project(320, 240)
        self.assertEqual(projection.undistorted_u, 320.0)
        self.assertAlmostEqual(projection.x, 0.0, places=9)

    def test_degenerate_homography_rejected(self):
        with self.assertRaises(CalibrationError):
            Calibration.from_inline([1, 0, 0, 0])
        with self.assertRaises(CalibrationError):
            Calibration.from_inline([[1, 0, 0], [0, math.nan, 0], [0, 0, 1]])

    def test_projection_far_outside_robot_range_rejected(self):
        calibration = Calibration.from_inline([1e3, 0, 0, 0, 1e3, 0, 0, 0, 1])
        with self.assertRaises(CalibrationError):
            calibration.project(100, 100)

    def test_missing_homography_in_model_rejected(self):
        with self.assertRaises(CalibrationError):
            Calibration.from_model({"camera": CAMERA}, source="test")


class ConfigParsingTests(unittest.TestCase):
    def test_metres_parsing(self):
        self.assertAlmostEqual(parse_metres("0.1024"), 0.1024)
        self.assertAlmostEqual(parse_metres("102.4mm"), 0.1024)
        self.assertAlmostEqual(parse_metres("0.1 m"), 0.1)
        self.assertIsNone(parse_metres(""))
        self.assertIsNone(parse_metres(None))
        with self.assertRaises(ConfigError):
            parse_metres("abc")
        with self.assertRaises(ConfigError):
            parse_metres("nan")

    def test_quaternion_parsing_and_normalisation(self):
        self.assertEqual(parse_quaternion(""), None)
        self.assertEqual(parse_quaternion(None), None)
        self.assertEqual(parse_quaternion("0,0,0,2"), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(parse_quaternion([1, 0, 0, 0]), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ConfigError):
            parse_quaternion("0,0,0")
        with self.assertRaises(ConfigError):
            parse_quaternion("0,0,0,0")
        with self.assertRaises(ConfigError):
            parse_quaternion("1,2,x,4")

    def test_config_rejects_unknown_workspace_and_bad_topics(self):
        with self.assertRaises(ConfigError):
            SkillConfig.from_cfg({"workspace": {"xmain": 1}})
        with self.assertRaises(ConfigError):
            SkillConfig.from_cfg({"topics": {"end_pose": "arm7aix/end_pose"}})
        with self.assertRaises(ConfigError):
            SkillConfig.from_cfg({"workspace": {"x_min": 1.0, "x_max": 0.0}})

    def test_config_defaults(self):
        config = SkillConfig.from_cfg({})
        self.assertAlmostEqual(config.default_z_m, 0.1024)
        self.assertIsNone(config.default_quaternion_xyzw)
        self.assertFalse(config.dry_run)
        self.assertEqual(config.arm_provider_id, "arm7aix_controller")
        self.assertAlmostEqual(config.position_tolerance_m, 0.005)

    def test_grasp_related_config_defaults_and_validation(self):
        config = SkillConfig.from_cfg({})
        self.assertEqual(config.hand_provider_id, "o6_hand")
        self.assertEqual(len(config.joint_names), 7)
        self.assertIsNone(config.home_joints)
        self.assertIsNone(config.grasp_pose)
        self.assertIsNone(config.place_pose)
        explicit = SkillConfig.from_cfg(
            {
                "home_joints": [0.1] * 7,
                "grasp_pose": [0.1, 0.2, 0.11],
                "place_pose": [0.1, 0.0, 0.11],
                "hand_open_values": [0.0] * 6,
                "hand_close_values": [1.0] * 6,
                "joint_names": ["a", "b", "c", "d", "e", "f", "g"],
            }
        )
        self.assertEqual(explicit.home_joints, tuple([0.1] * 7))
        self.assertEqual(explicit.grasp_pose, (0.1, 0.2, 0.11))
        self.assertEqual(explicit.joint_names[0], "a")
        for bad in (
            {"home_joints": [0.0] * 6},
            {"grasp_pose": [0.1, 0.2]},
            {"hand_open_values": [2.0] * 6},
            {"joint_names": ["a"]},
        ):
            with self.assertRaises(ConfigError):
                SkillConfig.from_cfg(bad)

    def test_workspace_check(self):
        workspace = {"x_min": -0.5, "x_max": 0.5, "z_min": 0.02, "z_max": 0.8}
        check_workspace(workspace, 0.1, 0.2, 0.1)  # 不抛
        with self.assertRaises(ConfigError):
            check_workspace(workspace, 0.9, 0.2, 0.1)
        with self.assertRaises(ConfigError):
            check_workspace(workspace, 0.1, 0.2, 0.01)


class CalibrationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, path: Path, homography) -> None:
        path.write_text(json.dumps(_model(homography=list(homography))), encoding="utf-8")

    def test_inline_homography_is_used_when_file_missing(self):
        config = SkillConfig.from_cfg(
            {"calibration_path": str(self.root / "missing.json"), "homography": HOMOGGRAPHY}
        )
        calibration = CalibrationCache(config).get()
        self.assertAlmostEqual(calibration.project(420, 140).x, 0.1, places=9)

    def test_missing_calibration_reports_clear_error(self):
        config = SkillConfig.from_cfg({"calibration_path": str(self.root / "missing.json")})
        cache = CalibrationCache(config)
        with self.assertRaises(CalibrationError) as ctx:
            cache.get()
        self.assertIn("标定", str(ctx.exception))

    def test_file_is_reloaded_when_mtime_changes(self):
        path = self.root / "calibration.json"
        self._write(path, HOMOGGRAPHY)
        config = SkillConfig.from_cfg({"calibration_path": str(path)})
        cache = CalibrationCache(config)
        self.assertAlmostEqual(cache.get().project(420, 140).x, 0.1, places=9)

        shifted = [list(HOMOGGRAPHY[0]), list(HOMOGGRAPHY[1]), list(HOMOGGRAPHY[2])]
        shifted[0][2] = -0.32 + 0.05
        self._write(path, shifted)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertAlmostEqual(cache.get().project(420, 140).x, 0.15, places=9)

    def test_broken_calibration_file_reports_error_not_crash(self):
        path = self.root / "calibration.json"
        path.write_text("{not json", encoding="utf-8")
        config = SkillConfig.from_cfg({"calibration_path": str(path)})
        cache = CalibrationCache(config)
        with self.assertRaises(CalibrationError):
            cache.get()
        self.assertIn("标定文件", cache.error or "")


class OrientationFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "grasp_orientation.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_quaternion_and_z(self):
        self.path.write_text(
            json.dumps({"target_quaternion_xyzw": [0, 0, 0, 2], "target_z_mm": 130}),
            encoding="utf-8",
        )
        values = read_orientation_file(self.path)
        self.assertEqual(values["target_quaternion_xyzw"], (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(values["target_z_m"], 0.13)

    def test_invalid_orientation_is_an_error(self):
        self.path.write_text(
            json.dumps({"target_quaternion_xyzw": [0, 0, 0, 0]}), encoding="utf-8"
        )
        with self.assertRaises(CalibrationError):
            read_orientation_file(self.path)


class QuaternionGeometryTests(unittest.TestCase):
    def test_angle(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        flipped = (0.0, 0.0, 0.0, -1.0)
        half_turn = (1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(quaternion_angle(identity, flipped), 0.0, places=9)
        self.assertAlmostEqual(quaternion_angle(identity, half_turn), math.pi, places=9)


class TargetResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calib_path = self.root / "calibration.json"
        self.calib_path.write_text(json.dumps(_model()), encoding="utf-8")
        self.orientation_path = self.root / "grasp_orientation.json"

    def tearDown(self):
        self.temp.cleanup()

    def _config(self, **overrides):
        cfg = {
            "calibration_path": str(self.calib_path),
            "orientation_path": str(self.orientation_path),
            "default_z_m": 0.1024,
            "default_quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
        }
        cfg.update(overrides)
        return SkillConfig.from_cfg(cfg)

    def _resolver(self, config):
        return TargetResolver(config, CalibrationCache(config))

    def test_manifest_defaults_when_orientation_file_missing(self):
        target = self._resolver(self._config()).resolve(
            u=420, v=140, z_text="", quaternion_text="", require_quaternion=True
        )
        self.assertAlmostEqual(target.x, 0.1, places=9)
        self.assertAlmostEqual(target.y, 0.1, places=9)
        self.assertAlmostEqual(target.z, 0.1024)
        self.assertEqual(target.quaternion, (1.0, 0.0, 0.0, 0.0))
        self.assertIn("manifest", target.sources["z"])

    def test_orientation_file_overrides_manifest_defaults(self):
        self.orientation_path.write_text(
            json.dumps({"target_quaternion_xyzw": [0, 0, 0, 2], "target_z_mm": 88.0}),
            encoding="utf-8",
        )
        target = self._resolver(self._config()).resolve(
            u=320, v=240, z_text="", quaternion_text="", require_quaternion=True
        )
        self.assertAlmostEqual(target.z, 0.088)
        self.assertEqual(target.quaternion, (0.0, 0.0, 0.0, 1.0))
        self.assertIn("orientation_path", target.sources["z"])
        self.assertIn("orientation_path", target.sources["quaternion"])

    def test_tool_arguments_win_when_overrides_enabled(self):
        self.orientation_path.write_text(
            json.dumps({"target_quaternion_xyzw": [0, 0, 0, 1], "target_z_mm": 88.0}),
            encoding="utf-8",
        )
        target = self._resolver(self._config(allow_tool_overrides=True)).resolve(
            u=320,
            v=240,
            z_text="120mm",
            quaternion_text="1,0,0,0",
            require_quaternion=True,
        )
        self.assertAlmostEqual(target.z, 0.12)
        self.assertEqual(target.quaternion, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(target.sources["z"], "工具参数")

    def test_missing_quaternion_rejected_for_motion(self):
        config = self._config(default_quaternion_xyzw=[])
        with self.assertRaises(ConfigError):
            self._resolver(config).resolve(
                u=320, v=240, z_text="", quaternion_text="", require_quaternion=True
            )
        # 只算像素→位姿时允许没有四元数
        target = self._resolver(config).resolve(
            u=320, v=240, z_text="", quaternion_text="", require_quaternion=False
        )
        self.assertIsNone(target.quaternion)

    def test_tool_overrides_ignored_by_default(self):
        self.orientation_path.write_text(
            json.dumps({"target_quaternion_xyzw": [0, 0, 0, 1], "target_z_mm": 110}),
            encoding="utf-8",
        )
        target = self._resolver(self._config()).resolve(
            u=320,
            v=240,
            z_text="50mm",
            quaternion_text="0,1,0,0",
            require_quaternion=True,
        )
        self.assertAlmostEqual(target.z, 0.110)
        self.assertEqual(target.quaternion, (0.0, 0.0, 0.0, 1.0))
        self.assertIn("ignored_tool_overrides", target.sources)
        self.assertIn("已忽略工具参数", target.note())

    def test_tool_overrides_honoured_when_enabled(self):
        target = self._resolver(self._config(allow_tool_overrides=True)).resolve(
            u=320,
            v=240,
            z_text="50mm",
            quaternion_text="0,1,0,0",
            require_quaternion=True,
        )
        self.assertAlmostEqual(target.z, 0.050)
        self.assertEqual(target.quaternion, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(target.sources["z"], "工具参数")

    def test_missing_z_rejected(self):
        config = self._config(default_z_m=None, default_quaternion_xyzw=[1, 0, 0, 0])
        with self.assertRaises(ConfigError):
            self._resolver(config).resolve(
                u=320, v=240, z_text="", quaternion_text="", require_quaternion=False
            )


class VlmParsingTests(unittest.TestCase):
    def test_failed_reply_is_empty(self):
        self.assertEqual(parse_detections({"failed": True, "objects": []}), [])

    def test_parses_valid_object(self):
        detections = parse_detections(
            {
                "failed": False,
                "objects": [
                    {
                        "class_name": "红色方块",
                        "box_center_x": 0.5,
                        "box_center_y": 0.4,
                        "box_width": 0.2,
                        "box_height": 0.1,
                        "box_rotation_deg": 30.0,
                    }
                ],
            }
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.class_name, "红色方块")
        self.assertAlmostEqual(detection.center_x, 0.5)
        self.assertAlmostEqual(detection.rotation_deg, 30.0)

    def test_invalid_boxes_are_skipped(self):
        detections = parse_detections(
            {
                "objects": [
                    {"box_center_x": 0.5, "box_center_y": 0.5, "box_width": 0, "box_height": 0.1},
                    {"box_center_x": "x", "box_center_y": 0.5, "box_width": 0.1, "box_height": 0.1},
                    {"box_center_x": 0.5, "box_center_y": 0.5, "box_width": 0.1, "box_height": 0.1},
                ]
            }
        )
        self.assertEqual(len(detections), 1)

        with self.assertRaises(VlmError):
            parse_detections({"objects": "nope"})
        # 单个框字段非法会被跳过（只有整体结构错误才抛）
        self.assertEqual(parse_detections({"objects": [{"box_center_x": None}]}), [])

    def test_contact_pixel_interpolates_between_center_and_bottom(self):
        detection = Detection(
            class_name="box",
            center_x=0.5,
            center_y=0.5,
            width=0.2,
            height=0.1,
            rotation_deg=0.0,
        )
        # 100x100 图：中心像素 (50,50)，底部中点在 y=55。
        u0, v0 = detection.contact_pixel(100, 100, ratio=0.0)
        self.assertAlmostEqual(u0, 50.0)
        self.assertAlmostEqual(v0, 50.0)
        u1, v1 = detection.contact_pixel(100, 100, ratio=1.0)
        self.assertAlmostEqual(u1, 50.0)
        self.assertAlmostEqual(v1, 55.0)
        uh, vh = detection.contact_pixel(100, 100, ratio=0.5)
        self.assertAlmostEqual(vh, 52.5)

    def test_rotated_box_corners(self):
        detection = Detection(
            class_name="box",
            center_x=0.5,
            center_y=0.5,
            width=0.2,
            height=0.1,
            rotation_deg=90.0,
        )
        corners = detection.corners(100, 100)
        self.assertEqual(len(corners), 4)
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        # 旋转 90° 后长边竖直：x 跨度 10px，y 跨度 20px
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=6)
        self.assertAlmostEqual(max(ys) - min(ys), 20.0, places=6)


class VlmHttpTests(unittest.TestCase):
    """用一个本地假 OpenAI 兼容端点验证请求/解析链路（不访问外网）。"""

    @classmethod
    def setUpClass(cls):
        import http.server
        import threading

        payload = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"failed": false, "objects": ['
                        '{"class_name": "蓝色方块", "box_center_x": 0.25, '
                        '"box_center_y": 0.4, "box_width": 0.1, '
                        '"box_height": 0.08, "box_rotation_deg": 0.0}]}\n```'
                    }
                }
            ]
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # noqa: N802 - 静音
                pass

        cls.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_detect_objects_roundtrip(self):
        import numpy as np

        image = np.zeros((48, 64, 3), dtype=np.uint8)
        detections = detect_objects(
            image,
            "蓝色方块",
            base_url=self.base_url,
            api_key="test",
            model="fake-vlm",
            timeout_s=5.0,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "蓝色方块")
        self.assertAlmostEqual(detections[0].center_x, 0.25)

    def test_missing_instruction_or_endpoint_rejected(self):
        import numpy as np

        image = np.zeros((48, 64, 3), dtype=np.uint8)
        with self.assertRaises(VlmError):
            detect_objects(image, "", base_url=self.base_url, api_key="t", model="m")
        with self.assertRaises(VlmError):
            detect_objects(image, "x", base_url="", api_key="t", model="m")
        with self.assertRaises(VlmError):
            detect_objects(image, "x", base_url=self.base_url, api_key="t", model="")


class VlmConfigTests(unittest.TestCase):
    def test_env_fallback_and_contact_ratio(self):
        import os

        old = {key: os.environ.get(key) for key in ("VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL")}
        try:
            os.environ["VLM_BASE_URL"] = "http://vlm.example/v1"
            os.environ["VLM_API_KEY"] = "k"
            os.environ["VLM_MODEL"] = "m"
            config = SkillConfig.from_cfg({})
            self.assertEqual(config.vlm_base_url, "http://vlm.example/v1")
            self.assertEqual(config.vlm_api_key, "k")
            self.assertEqual(config.vlm_model, "m")
            self.assertAlmostEqual(config.vlm_contact_ratio, 0.5)
            explicit = SkillConfig.from_cfg({"vlm_contact_ratio": 0.0, "vlm_base_url": "http://x"})
            self.assertAlmostEqual(explicit.vlm_contact_ratio, 0.0)
            self.assertEqual(explicit.vlm_base_url, "http://x")
            with self.assertRaises(ConfigError):
                SkillConfig.from_cfg({"vlm_contact_ratio": 1.5})
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class _FakeBridge:
    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])
        self.pose = PoseSample(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, time.monotonic())

    def latest_pose(self):
        return self.pose

    def move_pose(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0) if self.results else "reached"
        if result == "timeout":
            return {"reached": False, "timed_out": True, "elapsed": 0.01, "final": None}
        final = PoseSample(
            kwargs["x"], kwargs["y"], kwargs["z"], *kwargs["quaternion"],
            time.monotonic(),
        )
        self.pose = final
        return {"reached": True, "timed_out": False, "elapsed": 0.01, "final": final}


_PROJECTION = Projection(u=320.0, v=240.0, undistorted_u=320.0,
                         undistorted_v=240.0, x=0.1, y=0.05)


def _target(z=0.11, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return Target(projection=_PROJECTION, z=z, quaternion=quaternion,
                  sources={"z": "test", "quaternion": "test"})


class ApproachMotionTests(unittest.TestCase):
    def test_plan_waypoints_hover_then_target(self):
        target = _target(0.11)
        waypoints = plan_waypoints(target, 0.20)
        self.assertEqual(len(waypoints), 2)
        self.assertAlmostEqual(waypoints[0].z, 0.31)
        self.assertAlmostEqual(waypoints[1].z, 0.11)
        self.assertAlmostEqual(waypoints[0].x, target.x)
        self.assertEqual(waypoints[0].quaternion, target.quaternion)
        self.assertEqual(plan_waypoints(target, 0.0), [target])

    def test_two_stage_move_order(self):
        config = SkillConfig.from_cfg({"approach_height_m": 0.20, "dry_run": False})
        bridge = _FakeBridge()
        outcome = perform_move(config, _target(0.11), lambda: bridge)
        self.assertTrue(outcome.reached)
        self.assertEqual(len(bridge.calls), 2)
        self.assertAlmostEqual(bridge.calls[0]["z"], 0.31)
        self.assertAlmostEqual(bridge.calls[1]["z"], 0.11)
        self.assertAlmostEqual(outcome.final.z, 0.11)
        self.assertIn("先到目标正上方", outcome.message)

    def test_hover_timeout_stops_before_descend(self):
        config = SkillConfig.from_cfg({"approach_height_m": 0.20, "dry_run": False})
        bridge = _FakeBridge(results=["timeout"])
        outcome = perform_move(config, _target(0.11), lambda: bridge)
        self.assertFalse(outcome.reached)
        self.assertTrue(outcome.timed_out)
        self.assertEqual(len(bridge.calls), 1)
        self.assertIn("不会继续下降", outcome.message)

    def test_hover_outside_workspace_rejected(self):
        config = SkillConfig.from_cfg(
            {"approach_height_m": 0.20, "workspace": {"z_max": 0.25}}
        )

        def _no_bridge():
            raise AssertionError("workspace 检查应在建桥/发布前完成")

        with self.assertRaises(ConfigError):
            perform_move(config, _target(0.11), _no_bridge)

    def test_dry_run_lists_both_waypoints_without_bridge(self):
        config = SkillConfig.from_cfg({"approach_height_m": 0.20, "dry_run": True})

        def _no_bridge():
            raise AssertionError("dry_run 不应访问 ROS 桥")

        outcome = perform_move(config, _target(0.11), _no_bridge)
        self.assertTrue(outcome.ok)
        self.assertFalse(outcome.reached)
        self.assertIn("先到 hover", outcome.message)


class FakeHand:
    def __init__(self):
        self.events = []

    def open(self):
        self.events.append("open")

    def close(self):
        self.events.append("close")

    def settle_open(self):
        self.events.append("settle_open")

    def settle_close(self):
        self.events.append("settle_close")


class SequenceBridge(_FakeBridge):
    def __init__(self, results=None):
        super().__init__(results)
        self.joint_calls = []

    def move_joints(self, targets, *, timeout_s, tolerance_rad, feedback_timeout_s=2.0):
        self.joint_calls.append(list(targets))
        return {
            "reached": True,
            "timed_out": False,
            "elapsed": 0.01,
            "final": tuple(float(v) for v in targets),
        }


class HandPresetTests(unittest.TestCase):
    def test_raw_normalization(self):
        self.assertEqual(normalized_from_raw([255, 0]), (0.0, 1.0))
        values = normalized_from_raw([255, 179, 255, 255, 255, 255])
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 1.0 - 179 / 255, places=6)

    def test_preset_mapping_is_by_name_not_order(self):
        reversed_names = list(reversed(AXIS_NAMES))
        values = preset_values([67, 151, 0, 0, 0, 0], reversed_names)
        mapping = dict(values)
        self.assertAlmostEqual(mapping["thumb_cmc_pitch"], 1.0 - 67 / 255, places=6)
        self.assertAlmostEqual(mapping["thumb_cmc_yaw"], 1.0 - 151 / 255, places=6)
        self.assertEqual(mapping["index_mcp_pitch"], 1.0)

    def test_unknown_axis_layout_rejected(self):
        with self.assertRaises(HandError):
            preset_values([0, 0, 0, 0, 0, 0], ["a", "b", "c", "d", "e", "f"])
        with self.assertRaises(HandError):
            preset_values([0, 0, 0, 0, 0, 0], AXIS_NAMES[:5])


class GraspSequenceTests(unittest.TestCase):
    def _config(self, **overrides):
        cfg = {
            "approach_height_m": 0.20,
            "hand_close_settle_s": 0.0,
            "hand_open_settle_s": 0.0,
            "dry_run": False,
        }
        cfg.update(overrides)
        return SkillConfig.from_cfg(cfg)

    def test_grasp_open_hover_descend_close_lift(self):
        bridge = SequenceBridge()
        hand = FakeHand()
        outcome = grasp_target(
            self._config(), _target(0.11), lambda: bridge, hand
        )
        self.assertTrue(outcome.reached)
        self.assertEqual(hand.events, ["open", "settle_open", "close", "settle_close"])
        self.assertEqual(len(bridge.calls), 3)
        self.assertAlmostEqual(bridge.calls[0]["z"], 0.31)
        self.assertAlmostEqual(bridge.calls[1]["z"], 0.11)
        self.assertAlmostEqual(bridge.calls[2]["z"], 0.31)
        self.assertIn("抓取完成", outcome.message)

    def test_hover_failure_keeps_hand_open(self):
        bridge = SequenceBridge(results=["timeout"])
        hand = FakeHand()
        outcome = grasp_target(
            self._config(), _target(0.11), lambda: bridge, hand
        )
        self.assertFalse(outcome.reached)
        self.assertEqual(hand.events, ["open", "settle_open"])
        self.assertEqual(len(bridge.calls), 1)
        self.assertIn("未闭合手", outcome.message)

    def test_grasp_dry_run_touches_nothing(self):
        hand = FakeHand()

        def _no_bridge():
            raise AssertionError("dry_run 不应访问 ROS 桥")

        outcome = grasp_target(
            self._config(dry_run=True), _target(0.11), _no_bridge, hand
        )
        self.assertTrue(outcome.ok)
        self.assertFalse(outcome.reached)
        self.assertEqual(hand.events, [])
        self.assertIn("张手", outcome.message)

    def test_place_sequence_descends_then_opens_then_lifts(self):
        bridge = SequenceBridge()
        hand = FakeHand()
        outcome = place_target(
            self._config(), _target(0.11), lambda: bridge, hand
        )
        self.assertTrue(outcome.reached)
        self.assertEqual(hand.events, ["open", "settle_open"])
        self.assertEqual(len(bridge.calls), 3)
        self.assertAlmostEqual(bridge.calls[1]["z"], 0.11)
        self.assertAlmostEqual(bridge.calls[2]["z"], 0.31)

    def test_release_only_moves_hand(self):
        bridge = SequenceBridge()
        hand = FakeHand()
        outcome = release_hand(self._config(), hand)
        self.assertTrue(outcome.reached)
        self.assertEqual(hand.events, ["open", "settle_open"])
        self.assertEqual(bridge.calls, [])


class HomeSequenceTests(unittest.TestCase):
    def test_home_moves_joints_and_does_not_touch_hand(self):
        bridge = SequenceBridge()
        config = SkillConfig.from_cfg({"home_joints": [0.0] * 7})
        outcome = home_arm(config, lambda: bridge)
        self.assertTrue(outcome.reached)
        self.assertEqual(bridge.joint_calls, [[0.0] * 7])
        self.assertEqual(bridge.calls, [])

    def test_home_without_config_rejected(self):
        with self.assertRaises(ConfigError):
            home_arm(SkillConfig.from_cfg({}), lambda: SequenceBridge())

    def test_home_dry_run_touches_nothing(self):
        config = SkillConfig.from_cfg({"home_joints": [0.1] * 7, "dry_run": True})

        def _no_bridge():
            raise AssertionError("dry_run 不应访问 ROS 桥")

        outcome = home_arm(config, _no_bridge)
        self.assertTrue(outcome.ok)
        self.assertIn("home_joints", outcome.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
