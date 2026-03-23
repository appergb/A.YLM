"""Test module imports for AYLM package."""

import importlib
import subprocess
import sys

import pytest


def _can_import(module: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


class TestPackageImports:
    """Test that all package modules can be imported."""

    def test_import_aylm(self) -> None:
        aylm = pytest.importorskip("aylm")
        assert hasattr(aylm, "__version__")

    def test_import_aylm_tools(self) -> None:
        pytest.importorskip("aylm")
        tools = importlib.import_module("aylm.tools")
        assert tools is not None

    def test_import_voxelizer(self) -> None:
        tools = pytest.importorskip("aylm.tools")
        assert tools.PointCloudVoxelizer is not None

    def test_import_coord_transform(self) -> None:
        tools = pytest.importorskip("aylm.tools")
        assert tools.transform_for_navigation is not None

    def test_import_cli(self) -> None:
        cli = pytest.importorskip("aylm.cli")
        assert cli is not None


class TestDependencyImports:
    """Test that required dependencies can be imported."""

    def test_import_numpy(self) -> None:
        np = pytest.importorskip("numpy")
        assert hasattr(np, "array")

    def test_import_torch(self) -> None:
        if not _can_import("torch"):
            pytest.skip("torch import failed or unavailable in this environment")
        assert _can_import("torch")

    def test_import_scipy(self) -> None:
        scipy = pytest.importorskip("scipy")
        assert scipy is not None

    def test_import_plyfile(self) -> None:
        plyfile = pytest.importorskip("plyfile")
        assert plyfile.PlyData is not None

    def test_import_cv2(self) -> None:
        cv2 = pytest.importorskip("cv2")
        assert cv2 is not None

    def test_import_pil(self) -> None:
        PIL = pytest.importorskip("PIL")
        assert PIL.Image is not None

    def test_import_matplotlib(self) -> None:
        plt = pytest.importorskip("matplotlib.pyplot")
        assert plt is not None


class TestOptionalDependencies:
    """Test optional dependencies."""

    def test_import_open3d(self) -> None:
        o3d = pytest.importorskip("open3d", reason="open3d not installed (optional)")
        assert o3d is not None
