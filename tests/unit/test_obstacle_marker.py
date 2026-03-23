"""Tests for robust obstacle marker bounding boxes."""

import numpy as np

from aylm.tools.obstacle_marker import ObstacleMarker, ObstacleMarkerConfig


class TestObstacleMarkerBoundingBox:
    """ObstacleMarker bounding box tests."""

    def test_compute_bounding_box_trims_outliers_for_dense_cloud(self) -> None:
        marker = ObstacleMarker(
            ObstacleMarkerConfig(bbox_trim_percentile=0.05, bbox_trim_min_points=20)
        )
        core_points = np.array(
            [[1.0 + i * 0.01, 2.0 + (i % 5) * 0.01, 3.0 + (i % 7) * 0.01] for i in range(60)],
            dtype=np.float64,
        )
        outliers = np.array([[20.0, 2.0, 3.0], [-15.0, 2.0, 3.0]], dtype=np.float64)
        center, dimensions = marker._compute_bounding_box(
            np.vstack([core_points, outliers])
        )

        assert center[0] < 2.0
        assert dimensions[0] < 1.0

    def test_compute_bounding_box_small_cloud_uses_full_extent(self) -> None:
        marker = ObstacleMarker(
            ObstacleMarkerConfig(bbox_trim_percentile=0.1, bbox_trim_min_points=10)
        )
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 4.0, 6.0],
            ],
            dtype=np.float64,
        )
        center, dimensions = marker._compute_bounding_box(points)

        np.testing.assert_allclose(center, [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(dimensions, [2.0, 4.0, 6.0], atol=1e-6)
