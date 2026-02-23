"""语义融合模块测试。"""

from pathlib import Path

import numpy as np

from aylm.tools.semantic_fusion import (
    CameraIntrinsics,
    FusionConfig,
    SemanticFusion,
)
from aylm.tools.semantic_types import (
    SEMANTIC_COLORS,
    Detection2D,
    SemanticLabel,
    SemanticPointCloud,
)


class TestSemanticFusion:
    """SemanticFusion 类测试。"""

    def test_save_semantic_ply_with_colors(self, tmp_path: Path):
        """测试保存带语义颜色的 PLY 文件。"""
        # 创建测试点云
        n_points = 100
        points = np.random.randn(n_points, 3).astype(np.float64)
        colors = np.random.rand(n_points, 3).astype(np.float64)

        # 创建标签：前50个是 PERSON，后50个是 VEHICLE
        labels = np.zeros(n_points, dtype=np.uint8)
        labels[:50] = SemanticLabel.PERSON.value
        labels[50:] = SemanticLabel.VEHICLE.value

        confidences = np.random.rand(n_points).astype(np.float32) * 0.5 + 0.5

        semantic_pc = SemanticPointCloud(
            points=points,
            colors=colors,
            labels=labels,
            confidences=confidences,
        )

        # 保存 PLY
        output_path = tmp_path / "test_semantic.ply"
        fusion = SemanticFusion()
        fusion.save_semantic_ply(semantic_pc, output_path, include_semantic_colors=True)

        # 验证文件存在
        assert output_path.exists()

        # 读取并验证
        loaded_pc = SemanticFusion.load_semantic_ply(output_path)

        # 验证点数
        assert len(loaded_pc.points) == n_points

        # 验证标签
        assert np.array_equal(loaded_pc.labels, labels)

        # 验证置信度
        np.testing.assert_array_almost_equal(
            loaded_pc.confidences, confidences, decimal=5
        )

        # 验证颜色是语义颜色
        person_color = np.array(SEMANTIC_COLORS[SemanticLabel.PERSON])
        vehicle_color = np.array(SEMANTIC_COLORS[SemanticLabel.VEHICLE])

        # 前50个点应该是 PERSON 颜色
        np.testing.assert_array_almost_equal(
            loaded_pc.colors[:50], np.tile(person_color, (50, 1)), decimal=2
        )

        # 后50个点应该是 VEHICLE 颜色
        np.testing.assert_array_almost_equal(
            loaded_pc.colors[50:], np.tile(vehicle_color, (50, 1)), decimal=2
        )

    def test_save_semantic_ply_without_colors(self, tmp_path: Path):
        """测试保存不带语义颜色的 PLY 文件（保留原始颜色）。"""
        n_points = 50
        points = np.random.randn(n_points, 3).astype(np.float64)
        original_colors = np.random.rand(n_points, 3).astype(np.float64)
        labels = np.full(n_points, SemanticLabel.OBSTACLE.value, dtype=np.uint8)
        confidences = np.ones(n_points, dtype=np.float32) * 0.8

        semantic_pc = SemanticPointCloud(
            points=points,
            colors=original_colors,
            labels=labels,
            confidences=confidences,
        )

        output_path = tmp_path / "test_no_semantic_color.ply"
        fusion = SemanticFusion()
        fusion.save_semantic_ply(
            semantic_pc, output_path, include_semantic_colors=False
        )

        loaded_pc = SemanticFusion.load_semantic_ply(output_path)

        # 颜色应该是原始颜色（有一定精度损失因为转换为 uint8）
        np.testing.assert_array_almost_equal(
            loaded_pc.colors, original_colors, decimal=2
        )

        # 标签应该正确
        assert np.array_equal(loaded_pc.labels, labels)

    def test_ply_contains_all_fields(self, tmp_path: Path):
        """测试 PLY 文件包含所有必需字段。"""
        from plyfile import PlyData

        n_points = 10
        semantic_pc = SemanticPointCloud(
            points=np.random.randn(n_points, 3).astype(np.float64),
            colors=np.random.rand(n_points, 3).astype(np.float64),
            labels=np.array([1, 2, 3, 4, 5, 0, 1, 2, 3, 4], dtype=np.uint8),
            confidences=np.random.rand(n_points).astype(np.float32),
        )

        output_path = tmp_path / "test_fields.ply"
        fusion = SemanticFusion()
        fusion.save_semantic_ply(semantic_pc, output_path, include_semantic_colors=True)

        # 直接读取 PLY 验证字段
        ply_data = PlyData.read(str(output_path))
        vertex = ply_data["vertex"]
        field_names = vertex.data.dtype.names

        # 验证所有必需字段存在
        required_fields = [
            "x",
            "y",
            "z",
            "red",
            "green",
            "blue",
            "semantic_label",
            "semantic_confidence",
        ]
        for field in required_fields:
            assert field in field_names, f"缺少字段: {field}"

    def test_fuse_with_detections(self):
        """测试语义融合功能。"""
        # 创建简单的点云（在相机前方）
        n_points = 100
        points = np.zeros((n_points, 3), dtype=np.float64)
        points[:, 0] = np.random.uniform(-1, 1, n_points)  # X
        points[:, 1] = np.random.uniform(-1, 1, n_points)  # Y
        points[:, 2] = np.random.uniform(1, 5, n_points)  # Z (在相机前方)

        colors = np.ones((n_points, 3), dtype=np.float64) * 0.5

        # 创建检测结果（覆盖整个图像）
        mask = np.ones((480, 640), dtype=bool)
        detection = Detection2D(
            bbox=np.array([0, 0, 640, 480], dtype=np.float32),
            mask=mask,
            class_id=0,  # person
            confidence=0.9,
            semantic_label=SemanticLabel.PERSON,
        )

        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)

        config = FusionConfig(min_confidence=0.5, colorize_semantic=True)
        fusion = SemanticFusion(config)

        result = fusion.fuse(
            points=points,
            colors=colors,
            detections=[detection],
            image_shape=(480, 640),
            intrinsics=intrinsics,
        )

        # 验证结果
        assert len(result.points) == n_points
        assert result.labels is not None
        assert result.confidences is not None

        # 应该有一些点被标记为 PERSON
        person_count = (result.labels == SemanticLabel.PERSON.value).sum()
        assert person_count > 0, "应该有点被标记为 PERSON"
