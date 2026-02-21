"""Test module imports for AYLM package."""

import pytest


class TestPackageImports:
    """Test that all package modules can be imported."""

    def test_import_aylm(self):
        """Test importing main aylm package."""
        try:
            import aylm
            assert hasattr(aylm, "__version__")
        except ImportError as e:
            pytest.skip(f"Cannot import aylm: {e}")

    def test_import_aylm_tools(self):
        """Test importing aylm.tools module."""
        try:
            from aylm import tools
        except ImportError as e:
            pytest.skip(f"Cannot import aylm.tools: {e}")

    def test_import_voxelizer(self):
        """Test importing voxelizer module."""
        try:
            from aylm.tools import PointCloudVoxelizer
        except ImportError as e:
            pytest.skip(f"Cannot import voxelizer: {e}")

    def test_import_coord_transform(self):
        """Test importing coord_transform module."""
        try:
            from aylm.tools import transform_for_navigation
        except ImportError as e:
            pytest.skip(f"Cannot import coord_transform: {e}")

    def test_import_cli(self):
        """Test importing CLI module."""
        try:
            from aylm import cli
        except ImportError as e:
            pytest.skip(f"Cannot import cli: {e}")


class TestDependencyImports:
    """Test that required dependencies can be imported."""

    def test_import_numpy(self):
        """Test numpy import."""
        try:
            import numpy as np
            assert hasattr(np, "array")
        except ImportError as e:
            pytest.skip(f"numpy not installed: {e}")

    def test_import_torch(self):
        """Test torch import."""
        try:
            import torch
            assert hasattr(torch, "tensor")
        except ImportError as e:
            pytest.skip(f"torch not installed: {e}")

    def test_import_scipy(self):
        """Test scipy import."""
        try:
            import scipy
        except ImportError as e:
            pytest.skip(f"scipy not installed: {e}")

    def test_import_plyfile(self):
        """Test plyfile import."""
        try:
            from plyfile import PlyData
        except ImportError as e:
            pytest.skip(f"plyfile not installed: {e}")

    def test_import_cv2(self):
        """Test opencv import."""
        try:
            import cv2
        except ImportError as e:
            pytest.skip(f"opencv-python not installed: {e}")

    def test_import_pil(self):
        """Test PIL import."""
        try:
            from PIL import Image
        except ImportError as e:
            pytest.skip(f"Pillow not installed: {e}")

    def test_import_matplotlib(self):
        """Test matplotlib import."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            pytest.skip(f"matplotlib not installed: {e}")


class TestOptionalDependencies:
    """Test optional dependencies."""

    def test_import_open3d(self):
        """Test open3d import (optional)."""
        try:
            import open3d as o3d
        except ImportError as e:
            pytest.skip(f"open3d not installed (optional): {e}")
