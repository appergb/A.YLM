"""Tests for pointcloud voxelizer module."""

import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal

from aylm.tools.pointcloud_voxelizer import (
    HAS_OPEN3D,
    HAS_TORCH,
    PointCloud,
    PointCloudVoxelizer,
    VoxelizerConfig,
)


class TestVoxelizerConfig:
    """Test VoxelizerConfig dataclass."""

    def test_default_values(self) -> None:
        config = VoxelizerConfig()
        assert config.voxel_size == 0.05
        assert config.statistical_nb_neighbors == 20
        assert config.statistical_std_ratio == 2.0
        assert config.ransac_distance_threshold == 0.02
        assert config.ransac_n_points == 3
        assert config.ransac_num_iterations == 1000

    def test_custom_values(self) -> None:
        config = VoxelizerConfig(
            voxel_size=0.01,
            statistical_nb_neighbors=30,
            ransac_num_iterations=500,
        )
        assert config.voxel_size == 0.01
        assert config.statistical_nb_neighbors == 30
        assert config.ransac_num_iterations == 500


class TestPointCloud:
    """Test PointCloud dataclass."""

    def test_points_only(self) -> None:
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        pc = PointCloud(points=points)
        assert pc.points.shape == (3, 3)
        assert pc.colors is None
        assert pc.normals is None

    def test_with_colors(self) -> None:
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        pc = PointCloud(points=points, colors=colors)
        assert pc.colors.shape == (2, 3)

    def test_with_normals(self) -> None:
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        normals = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float64)
        pc = PointCloud(points=points, normals=normals)
        assert pc.normals.shape == (2, 3)


class TestPointCloudVoxelizer:
    """Test PointCloudVoxelizer class."""

    def test_init_default_config(self) -> None:
        voxelizer = PointCloudVoxelizer()
        assert voxelizer.config.voxel_size == 0.05

    def test_init_custom_config(self) -> None:
        config = VoxelizerConfig(voxel_size=0.1)
        voxelizer = PointCloudVoxelizer(config=config)
        assert voxelizer.config.voxel_size == 0.1


class TestVoxelDownsampleNumpy:
    """Test voxel downsampling with numpy implementation."""

    def test_downsample_reduces_points(self) -> None:
        np.random.seed(42)
        points = np.random.rand(1000, 3).astype(np.float64)
        pc = PointCloud(points=points)

        config = VoxelizerConfig(voxel_size=0.1)
        voxelizer = PointCloudVoxelizer(config=config)

        result = voxelizer._voxel_downsample_numpy(pc)
        assert len(result.points) < len(pc.points)

    def test_downsample_preserves_colors(self) -> None:
        np.random.seed(42)
        points = np.random.rand(100, 3).astype(np.float64)
        colors = np.random.rand(100, 3).astype(np.float64)
        pc = PointCloud(points=points, colors=colors)

        config = VoxelizerConfig(voxel_size=0.2)
        voxelizer = PointCloudVoxelizer(config=config)

        result = voxelizer._voxel_downsample_numpy(pc)
        assert result.colors is not None
        assert result.colors.shape[1] == 3

    def test_downsample_single_voxel(self) -> None:
        points = np.array(
            [[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03]],
            dtype=np.float64,
        )
        pc = PointCloud(points=points)

        config = VoxelizerConfig(voxel_size=1.0)
        voxelizer = PointCloudVoxelizer(config=config)

        result = voxelizer._voxel_downsample_numpy(pc)
        assert len(result.points) == 1
        expected_mean = points.mean(axis=0)
        assert_array_almost_equal(result.points[0], expected_mean)


class TestRemoveOutliersNumpy:
    """Test statistical outlier removal with numpy implementation."""

    def test_removes_outliers(self) -> None:
        np.random.seed(42)
        cluster = np.random.randn(100, 3) * 0.1
        outlier = np.array([[10.0, 10.0, 10.0]])
        points = np.vstack([cluster, outlier]).astype(np.float64)
        pc = PointCloud(points=points)

        config = VoxelizerConfig(statistical_nb_neighbors=10, statistical_std_ratio=1.0)
        voxelizer = PointCloudVoxelizer(config=config)

        result = voxelizer._remove_outliers_numpy(pc, config)
        assert len(result.points) < len(pc.points)

    def test_preserves_colors_on_removal(self) -> None:
        np.random.seed(42)
        points = np.random.randn(50, 3).astype(np.float64) * 0.1
        colors = np.random.rand(50, 3).astype(np.float64)
        pc = PointCloud(points=points, colors=colors)

        config = VoxelizerConfig(statistical_nb_neighbors=5)
        voxelizer = PointCloudVoxelizer(config=config)

        result = voxelizer._remove_outliers_numpy(pc, config)
        if result.colors is not None:
            assert result.colors.shape[1] == 3


class TestDetectGroundNumpy:
    """Test RANSAC ground detection with numpy implementation."""

    def test_detects_ground_plane(self) -> None:
        np.random.seed(42)
        ground_x = np.random.rand(100) * 10
        ground_y = np.random.rand(100) * 10
        ground_z = np.zeros(100) + np.random.randn(100) * 0.01
        ground = np.column_stack([ground_x, ground_y, ground_z])

        above = np.random.rand(50, 3) * 5
        above[:, 2] += 1.0

        points = np.vstack([ground, above]).astype(np.float64)
        pc = PointCloud(points=points)

        config = VoxelizerConfig(
            ransac_distance_threshold=0.05, ransac_num_iterations=100
        )
        voxelizer = PointCloudVoxelizer(config=config)

        result, plane = voxelizer._detect_ground_numpy(pc, config)
        assert len(result.points) < len(pc.points)
        assert len(plane) == 4


class TestCoordinateTransform:
    """Test coordinate system transformation."""

    def test_opencv_to_robot_transform(self) -> None:
        points = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        pc = PointCloud(points=points)

        result = PointCloudVoxelizer.transform_opencv_to_robot(pc)

        assert_array_almost_equal(result.points[0], [0, -1, 0])
        assert_array_almost_equal(result.points[1], [0, 0, -1])
        assert_array_almost_equal(result.points[2], [1, 0, 0])

    def test_transform_preserves_colors(self) -> None:
        points = np.array([[1, 2, 3]], dtype=np.float64)
        colors = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        pc = PointCloud(points=points, colors=colors)

        result = PointCloudVoxelizer.transform_opencv_to_robot(pc)
        assert result.colors is not None
        assert_array_almost_equal(result.colors, colors)


class TestPlyIO:
    """Test PLY file I/O operations."""

    def test_save_and_load_roundtrip(self) -> None:
        np.random.seed(42)
        points = np.random.rand(100, 3).astype(np.float64)
        colors = np.random.rand(100, 3).astype(np.float64)
        pc = PointCloud(points=points, colors=colors)

        voxelizer = PointCloudVoxelizer()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = Path(f.name)

        try:
            voxelizer.save_ply(pc, filepath)
            assert filepath.exists()

            loaded = voxelizer.load_ply(filepath)
            assert len(loaded.points) == len(pc.points)
            assert loaded.colors is not None
        finally:
            filepath.unlink()

    def test_save_without_colors(self) -> None:
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        pc = PointCloud(points=points)

        voxelizer = PointCloudVoxelizer()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = Path(f.name)

        try:
            voxelizer.save_ply(pc, filepath)
            assert filepath.exists()

            loaded = voxelizer.load_ply(filepath)
            assert len(loaded.points) == 2
            assert loaded.colors is None
        finally:
            filepath.unlink()


class TestGPUAvailability:
    """Test GPU availability detection."""

    def test_has_torch_flag(self) -> None:
        try:
            import torch  # noqa: F401

            assert HAS_TORCH is True
        except ImportError:
            assert HAS_TORCH is False

    def test_has_open3d_flag(self) -> None:
        try:
            import open3d  # noqa: F401

            assert HAS_OPEN3D is True
        except ImportError:
            assert HAS_OPEN3D is False


class TestProcessPipeline:
    """Test the complete processing pipeline."""

    def test_process_creates_output(self) -> None:
        np.random.seed(42)
        points = np.random.rand(500, 3).astype(np.float64)
        colors = np.random.rand(500, 3).astype(np.float64)
        pc = PointCloud(points=points, colors=colors)

        config = VoxelizerConfig(voxel_size=0.1)
        voxelizer = PointCloudVoxelizer(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.ply"
            output_path = Path(tmpdir) / "output.ply"

            voxelizer.save_ply(pc, input_path)

            result = voxelizer.process(
                input_path,
                output_path,
                remove_ground=False,
                transform_coords=False,
            )

            assert output_path.exists()
            assert len(result.points) > 0
            assert len(result.points) < len(pc.points)

    def test_process_with_ground_removal(self) -> None:
        np.random.seed(42)
        ground = np.column_stack(
            [
                np.random.rand(200) * 10,
                np.random.rand(200) * 10,
                np.zeros(200) + np.random.randn(200) * 0.01,
            ]
        )
        objects = np.random.rand(100, 3) * 5
        objects[:, 2] += 1.0

        points = np.vstack([ground, objects]).astype(np.float64)
        pc = PointCloud(points=points)

        config = VoxelizerConfig(voxel_size=0.1, ransac_distance_threshold=0.05)
        voxelizer = PointCloudVoxelizer(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.ply"
            output_path = Path(tmpdir) / "output.ply"

            voxelizer.save_ply(pc, input_path)

            result = voxelizer.process(
                input_path,
                output_path,
                remove_ground=True,
                transform_coords=False,
            )

            assert len(result.points) < len(pc.points)

    def test_process_with_coordinate_transform(self) -> None:
        np.random.seed(42)
        points = np.random.rand(100, 3).astype(np.float64)
        pc = PointCloud(points=points)

        config = VoxelizerConfig(voxel_size=0.5, statistical_nb_neighbors=5)
        voxelizer = PointCloudVoxelizer(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.ply"
            output_path = Path(tmpdir) / "output.ply"

            voxelizer.save_ply(pc, input_path)

            voxelizer.process(
                input_path,
                output_path,
                remove_ground=False,
                transform_coords=True,
            )

            assert output_path.exists()
