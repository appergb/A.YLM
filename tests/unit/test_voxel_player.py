"""Tests for voxel player module."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aylm.tools.voxel_player import (
    HAS_MATPLOTLIB,
    HAS_OPEN3D,
    PlaybackInfo,
    PlaybackState,
    PlayerConfig,
)


class TestPlaybackState:
    """Test PlaybackState enum."""

    def test_all_states_exist(self) -> None:
        assert PlaybackState.STOPPED.value == "stopped"
        assert PlaybackState.PLAYING.value == "playing"
        assert PlaybackState.PAUSED.value == "paused"
        assert len(PlaybackState) == 3


class TestPlayerConfig:
    """Test PlayerConfig dataclass."""

    def test_default_values(self) -> None:
        config = PlayerConfig()
        assert config.fps == 10.0
        assert config.loop is True
        assert config.auto_center is True
        assert config.point_size == 2.0
        assert config.background_color == (0.1, 0.1, 0.1)
        assert config.window_width == 1280
        assert config.window_height == 720
        assert config.window_title == "A.YLM Voxel Player"
        assert config.show_coordinate_frame is True
        assert config.show_info is True

    def test_custom_values(self) -> None:
        config = PlayerConfig(
            fps=30.0,
            loop=False,
            point_size=5.0,
            window_width=1920,
            window_height=1080,
        )
        assert config.fps == 30.0
        assert config.loop is False
        assert config.point_size == 5.0
        assert config.window_width == 1920
        assert config.window_height == 1080


class TestPlaybackInfo:
    """Test PlaybackInfo dataclass."""

    def test_default_values(self) -> None:
        info = PlaybackInfo()
        assert info.current_frame == 0
        assert info.total_frames == 0
        assert info.current_file == ""
        assert info.state == PlaybackState.STOPPED
        assert info.fps == 0.0
        assert info.elapsed_time == 0.0

    def test_progress_property(self) -> None:
        assert PlaybackInfo(total_frames=0).progress == 0.0
        assert PlaybackInfo(current_frame=50, total_frames=100).progress == 0.5
        assert PlaybackInfo(current_frame=100, total_frames=100).progress == 1.0

    def test_custom_values(self) -> None:
        info = PlaybackInfo(
            current_frame=25,
            total_frames=100,
            current_file="frame_025.ply",
            state=PlaybackState.PLAYING,
            fps=15.0,
            elapsed_time=2.5,
        )
        assert info.current_frame == 25
        assert info.total_frames == 100
        assert info.current_file == "frame_025.ply"
        assert info.state == PlaybackState.PLAYING
        assert info.fps == 15.0
        assert info.elapsed_time == 2.5


class TestBackendAvailability:
    """Test backend availability flags."""

    def test_has_open3d_flag(self) -> None:
        try:
            import open3d  # noqa: F401

            assert HAS_OPEN3D is True
        except ImportError:
            assert HAS_OPEN3D is False

    def test_has_matplotlib_flag(self) -> None:
        try:
            import matplotlib  # noqa: F401

            assert HAS_MATPLOTLIB is True
        except ImportError:
            assert HAS_MATPLOTLIB is False


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
class TestVoxelPlayer:
    """Test VoxelPlayer class (requires Open3D)."""

    def test_init_default_config(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        assert player.config.fps == 10.0
        assert player._state == PlaybackState.STOPPED

    def test_init_custom_config(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        config = PlayerConfig(fps=30.0, loop=False)
        player = VoxelPlayer(config=config)
        assert player.config.fps == 30.0
        assert player.config.loop is False

    def test_load_sequence_empty_directory(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        with tempfile.TemporaryDirectory() as tmpdir:
            count = player.load_sequence(Path(tmpdir))
            assert count == 0

    def test_load_sequence_with_files(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for i in range(5):
                (tmppath / f"frame_{i:03d}.ply").touch()

            count = player.load_sequence(tmppath)
            assert count == 5

    def test_load_sequence_single_file(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = Path(f.name)

        try:
            count = player.load_sequence(filepath)
            assert count == 1
        finally:
            filepath.unlink()

    def test_set_fps(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        player.set_fps(30.0)
        assert player._fps == 30.0

    def test_set_fps_clamped(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        player.set_fps(0.1)
        assert player._fps == 1.0

        player.set_fps(100.0)
        assert player._fps == 60.0

    def test_get_info(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        info = player.get_info()
        assert isinstance(info, PlaybackInfo)
        assert info.state == PlaybackState.STOPPED

    def test_stop(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        player._state = PlaybackState.PLAYING
        player.stop()
        assert player._state == PlaybackState.STOPPED

    def test_pause(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        player._state = PlaybackState.PLAYING
        player.pause()
        assert player._state == PlaybackState.PAUSED

    def test_resume(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        player._state = PlaybackState.PAUSED
        player.resume()
        assert player._state == PlaybackState.PLAYING

    def test_seek_valid_index(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for i in range(10):
                (tmppath / f"frame_{i:03d}.ply").touch()
            player.load_sequence(tmppath)
            player.seek(5)

    def test_set_on_frame_change(self) -> None:
        from aylm.tools.voxel_player import VoxelPlayer

        player = VoxelPlayer()
        callback = MagicMock()
        player.set_on_frame_change(callback)
        assert player._on_frame_change == callback


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestMatplotlibVoxelPlayer:
    """Test MatplotlibVoxelPlayer class."""

    def test_init_default_config(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        assert player.config.fps == 10.0
        assert player._state == PlaybackState.STOPPED

    def test_init_custom_config(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        config = PlayerConfig(fps=20.0, loop=False)
        player = MatplotlibVoxelPlayer(config=config)
        assert player.config.fps == 20.0
        assert player.config.loop is False

    def test_load_sequence_empty_directory(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        with tempfile.TemporaryDirectory() as tmpdir:
            count = player.load_sequence(Path(tmpdir))
            assert count == 0

    def test_load_sequence_with_files(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for i in range(5):
                (tmppath / f"frame_{i:03d}.ply").touch()

            count = player.load_sequence(tmppath)
            assert count == 5

    def test_set_fps(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        player.set_fps(25.0)
        assert player._fps == 25.0

    def test_set_fps_clamped(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        player.set_fps(0.5)
        assert player._fps == 1.0

        player.set_fps(100.0)
        assert player._fps == 60.0

    def test_pause(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        player._state = PlaybackState.PLAYING
        player.pause()
        assert player._state == PlaybackState.PAUSED

    def test_stop(self) -> None:
        from aylm.tools.voxel_player import MatplotlibVoxelPlayer

        player = MatplotlibVoxelPlayer()
        player._state = PlaybackState.PLAYING
        player.stop()
        assert player._state == PlaybackState.STOPPED


class TestPlaySequenceFunction:
    """Test the play_sequence convenience function."""

    @pytest.mark.skipif(
        not HAS_OPEN3D and not HAS_MATPLOTLIB,
        reason="No visualization backend available",
    )
    def test_play_sequence_empty_directory(self, capsys: Any) -> None:
        from aylm.tools.voxel_player import play_sequence

        with tempfile.TemporaryDirectory() as tmpdir:
            play_sequence(tmpdir)


class TestVoxelPlayerWithoutOpen3D:
    """Test VoxelPlayer behavior when Open3D is not available."""

    @patch.dict("sys.modules", {"open3d": None})
    def test_import_error_without_open3d(self) -> None:
        pass
