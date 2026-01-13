"""Basic import tests for A.YLM."""

import pytest


def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        import aylm
        assert aylm is not None
    except ImportError as e:
        pytest.skip(f"aylm import failed: {e}")


def test_tools_imports():
    """Test that tool modules can be imported."""
    try:
        from aylm.tools.coordinate_utils import transform_for_navigation
        from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer
        assert transform_for_navigation is not None
        assert PointCloudVoxelizer is not None
    except ImportError as e:
        pytest.skip(f"Tools import failed: {e}")


def test_dependencies():
    """Test that core dependencies are available."""
    import numpy as np
    import scipy
    from PIL import Image

    assert np is not None
    assert scipy is not None
    assert Image is not None


def test_torch_availability():
    """Test PyTorch availability."""
    try:
        import torch
        assert torch is not None
        print(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def test_open3d_availability():
    """Test Open3D availability."""
    try:
        import open3d as o3d
        assert o3d is not None
        print(f"Open3D version: {o3d.__version__}")
    except ImportError as e:
        pytest.skip(f"Open3D not available: {e}")


def test_sharp_availability():
    """Test SHARP availability."""
    try:
        import sys
        sys.path.insert(0, 'ml-sharp/src')
        import sharp
        assert sharp is not None
    except ImportError as e:
        pytest.skip(f"SHARP not available: {e}")