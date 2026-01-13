#!/usr/bin/env python3
"""Unit tests for coordinate_utils.py"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from coordinate_utils import (
    opencv_to_navigation_coords,
    navigation_to_opencv_coords,
    transform_gaussians,
    load_ply_gaussians,
    save_ply_gaussians
)


class TestCoordinateTransforms:
    """Test coordinate system transformations"""

    def test_opencv_to_navigation_coords(self):
        """Test OpenCV to navigation coordinate transformation"""
        # OpenCV: x=right, y=down, z=forward
        # Navigation: x=forward, y=left, z=up
        opencv_point = np.array([1.0, 2.0, 3.0])  # right, down, forward

        nav_point = opencv_to_navigation_coords(opencv_point)

        # Expected: forward, left, up
        expected = np.array([3.0, -1.0, -2.0])  # forward, left, up
        np.testing.assert_array_almost_equal(nav_point, expected)

    def test_navigation_to_opencv_coords(self):
        """Test navigation to OpenCV coordinate transformation"""
        # Navigation: x=forward, y=left, z=up
        nav_point = np.array([3.0, -1.0, -2.0])  # forward, left, up

        opencv_point = navigation_to_opencv_coords(nav_point)

        # Expected: right, down, forward
        expected = np.array([1.0, 2.0, 3.0])  # right, down, forward
        np.testing.assert_array_almost_equal(opencv_point, expected)

    def test_roundtrip_transform(self):
        """Test that forward and inverse transforms are inverses"""
        original_point = np.array([1.5, -2.3, 4.7])

        # Transform to navigation and back
        nav_point = opencv_to_navigation_coords(original_point)
        back_to_opencv = navigation_to_opencv_coords(nav_point)

        np.testing.assert_array_almost_equal(back_to_opencv, original_point)

    def test_transform_gaussians_positions(self):
        """Test gaussian data transformation"""
        # Create test gaussian data
        test_data = {
            "positions": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "scales": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "rotations": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "colors": np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]),
            "opacities": np.array([0.8, 0.9])
        }

        transformed = transform_gaussians(test_data)

        # Check positions are transformed
        expected_positions = np.array([
            [3.0, -1.0, -2.0],  # First point transformed
            [6.0, -4.0, -5.0]   # Second point transformed
        ])
        np.testing.assert_array_almost_equal(transformed["positions"], expected_positions)

        # Other fields should remain unchanged
        np.testing.assert_array_equal(transformed["scales"], test_data["scales"])
        np.testing.assert_array_equal(transformed["colors"], test_data["colors"])
        np.testing.assert_array_equal(transformed["opacities"], test_data["opacities"])


class TestPLYOperations:
    """Test PLY file operations"""

    def test_ply_roundtrip(self):
        """Test loading and saving PLY files"""
        # Create test gaussian data
        test_data = {
            "positions": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "scales": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "rotations": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "colors": np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]),
            "opacities": np.array([0.8, 0.9])
        }

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save and reload
            save_ply_gaussians(tmp_path, test_data)
            loaded_data = load_ply_gaussians(tmp_path)

            # Check that data is preserved
            np.testing.assert_array_almost_equal(loaded_data["positions"], test_data["positions"])
            np.testing.assert_array_almost_equal(loaded_data["scales"], test_data["scales"])
            np.testing.assert_array_almost_equal(loaded_data["rotations"], test_data["rotations"])
            np.testing.assert_array_almost_equal(loaded_data["colors"], test_data["colors"])
            np.testing.assert_array_almost_equal(loaded_data["opacities"], test_data["opacities"])

        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__])