"""体素序列可视化播放器模块。

提供PLY点云序列的实时3D播放功能，支持：
- 顺序播放PLY序列
- 播放控制（播放/暂停/速度调节）
- 键盘快捷键操作
- 可选导出为视频
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    import open3d as o3d

logger = logging.getLogger(__name__)

# 尝试导入Open3D
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None  # type: ignore
    logger.warning("Open3D not available, will use matplotlib fallback")

# 尝试导入matplotlib作为fallback
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available")


class PlaybackState(Enum):
    """播放状态枚举。"""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class PlayerConfig:
    """播放器配置。

    Attributes:
        fps: 播放帧率
        loop: 是否循环播放
        auto_center: 自动居中视图
        point_size: 点大小
        background_color: 背景颜色 (R, G, B)
        window_width: 窗口宽度
        window_height: 窗口高度
        window_title: 窗口标题
        show_coordinate_frame: 显示坐标系
        show_info: 显示播放信息
    """

    fps: float = 10.0
    loop: bool = True
    auto_center: bool = True
    point_size: float = 2.0
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    window_width: int = 1280
    window_height: int = 720
    window_title: str = "A.YLM Voxel Player"
    show_coordinate_frame: bool = True
    show_info: bool = True


@dataclass
class PlaybackInfo:
    """播放信息。

    Attributes:
        current_frame: 当前帧索引
        total_frames: 总帧数
        current_file: 当前文件名
        state: 播放状态
        fps: 当前帧率
        elapsed_time: 已播放时间
    """

    current_frame: int = 0
    total_frames: int = 0
    current_file: str = ""
    state: PlaybackState = PlaybackState.STOPPED
    fps: float = 0.0
    elapsed_time: float = 0.0

    @property
    def progress(self) -> float:
        """获取播放进度 (0.0-1.0)。"""
        if self.total_frames == 0:
            return 0.0
        return self.current_frame / self.total_frames


class VoxelPlayer:
    """体素序列播放器。

    使用Open3D实现PLY点云序列的实时3D可视化播放。

    键盘控制：
        Space: 播放/暂停
        Left/Right: 上一帧/下一帧
        Up/Down: 加速/减速
        R: 重置到开头
        L: 切换循环模式
        C: 重置视角
        Q/Esc: 退出

    Example:
        >>> player = VoxelPlayer(PlayerConfig(fps=15))
        >>> player.load_sequence(Path("output/voxelized"))
        >>> player.play()  # 阻塞直到窗口关闭
    """

    def __init__(self, config: PlayerConfig | None = None):
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for VoxelPlayer")

        self.config = config or PlayerConfig()
        self._ply_files: list[Path] = []
        self._point_clouds: list[Any] = []
        self._preloaded = False

        # 播放状态
        self._state = PlaybackState.STOPPED
        self._current_index = 0
        self._fps = self.config.fps
        self._loop = self.config.loop
        self._start_time: float | None = None

        # 线程控制
        self._play_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # 可视化
        self._vis: o3d.visualization.VisualizerWithKeyCallback | None = None
        self._current_pcd: o3d.geometry.PointCloud | None = None

        # 回调
        self._on_frame_change: Callable[[PlaybackInfo], None] | None = None

    def load_sequence(
        self, input_path: Path, pattern: str = "*.ply", preload: bool = False
    ) -> int:
        """加载PLY序列。

        Args:
            input_path: 输入目录或文件列表
            pattern: 文件匹配模式
            preload: 是否预加载所有点云到内存

        Returns:
            加载的文件数量
        """
        self._ply_files = []
        self._point_clouds = []
        self._preloaded = False

        if input_path.is_file():
            self._ply_files = [input_path]
        else:
            self._ply_files = sorted(input_path.glob(pattern))

        if not self._ply_files:
            logger.warning(f"No PLY files found in {input_path}")
            return 0

        logger.info(f"Found {len(self._ply_files)} PLY files")

        if preload:
            self._preload_all()

        return len(self._ply_files)

    def _preload_all(self):
        """预加载所有点云到内存。"""
        logger.info("Preloading all point clouds...")
        self._point_clouds = []

        for i, path in enumerate(self._ply_files):
            pcd = o3d.io.read_point_cloud(str(path))
            self._point_clouds.append(pcd)
            if (i + 1) % 10 == 0:
                logger.info(f"  Loaded {i + 1}/{len(self._ply_files)}")

        self._preloaded = True
        logger.info("Preloading complete")

    def _load_frame(self, index: int) -> o3d.geometry.PointCloud | None:
        """加载指定帧的点云。"""
        if not 0 <= index < len(self._ply_files):
            return None

        if self._preloaded:
            return self._point_clouds[index]

        return o3d.io.read_point_cloud(str(self._ply_files[index]))

    def _setup_visualizer(self):
        """设置可视化窗口。"""
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._vis.create_window(
            window_name=self.config.window_title,
            width=self.config.window_width,
            height=self.config.window_height,
        )

        # 注册键盘回调
        self._vis.register_key_callback(ord(" "), self._on_space)
        self._vis.register_key_callback(262, self._on_right)  # Right arrow
        self._vis.register_key_callback(263, self._on_left)  # Left arrow
        self._vis.register_key_callback(265, self._on_up)  # Up arrow
        self._vis.register_key_callback(264, self._on_down)  # Down arrow
        self._vis.register_key_callback(ord("R"), self._on_reset)
        self._vis.register_key_callback(ord("L"), self._on_loop_toggle)
        self._vis.register_key_callback(ord("C"), self._on_center)
        self._vis.register_key_callback(ord("Q"), self._on_quit)
        self._vis.register_key_callback(256, self._on_quit)  # Escape

        # 设置渲染选项
        opt = self._vis.get_render_option()
        opt.point_size = self.config.point_size
        opt.background_color = self.config.background_color

        # 添加坐标系
        if self.config.show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            self._vis.add_geometry(coord_frame)

        # 加载第一帧
        self._current_pcd = self._load_frame(0)
        if self._current_pcd is not None:
            self._vis.add_geometry(self._current_pcd)
            if self.config.auto_center:
                self._vis.reset_view_point(True)

    def _update_frame(self, index: int):
        """更新显示的帧。"""
        if self._vis is None:
            return

        new_pcd = self._load_frame(index)
        if new_pcd is None:
            return

        with self._lock:
            if self._current_pcd is not None:
                self._current_pcd.points = new_pcd.points
                if new_pcd.has_colors():
                    self._current_pcd.colors = new_pcd.colors
                self._vis.update_geometry(self._current_pcd)

            self._current_index = index

        # 触发回调
        if self._on_frame_change:
            self._on_frame_change(self.get_info())

    def _playback_loop(self):
        """播放循环线程。"""
        frame_interval = 1.0 / self._fps

        while not self._stop_event.is_set():
            if self._state != PlaybackState.PLAYING:
                time.sleep(0.01)
                continue

            start = time.time()

            # 更新帧
            next_index = self._current_index + 1
            if next_index >= len(self._ply_files):
                if self._loop:
                    next_index = 0
                else:
                    self._state = PlaybackState.STOPPED
                    continue

            self._update_frame(next_index)

            # 控制帧率
            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # 键盘回调
    def _on_space(self, vis):
        """空格键：播放/暂停。"""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED
            logger.info("Paused")
        else:
            self._state = PlaybackState.PLAYING
            if self._start_time is None:
                self._start_time = time.time()
            logger.info("Playing")
        return False

    def _on_right(self, vis):
        """右箭头：下一帧。"""
        if self._current_index < len(self._ply_files) - 1:
            self._update_frame(self._current_index + 1)
        return False

    def _on_left(self, vis):
        """左箭头：上一帧。"""
        if self._current_index > 0:
            self._update_frame(self._current_index - 1)
        return False

    def _on_up(self, vis):
        """上箭头：加速。"""
        self._fps = min(60.0, self._fps * 1.25)
        logger.info(f"FPS: {self._fps:.1f}")
        return False

    def _on_down(self, vis):
        """下箭头：减速。"""
        self._fps = max(1.0, self._fps / 1.25)
        logger.info(f"FPS: {self._fps:.1f}")
        return False

    def _on_reset(self, vis):
        """R键：重置到开头。"""
        self._update_frame(0)
        self._start_time = None
        logger.info("Reset to beginning")
        return False

    def _on_loop_toggle(self, vis):
        """L键：切换循环模式。"""
        self._loop = not self._loop
        logger.info(f"Loop: {'ON' if self._loop else 'OFF'}")
        return False

    def _on_center(self, vis):
        """C键：重置视角。"""
        if self._vis is not None:
            self._vis.reset_view_point(True)
        return False

    def _on_quit(self, vis):
        """Q键/Esc：退出。"""
        self._stop_event.set()
        self._state = PlaybackState.STOPPED
        return True

    def play(self, blocking: bool = True):
        """开始播放。

        Args:
            blocking: 是否阻塞直到窗口关闭
        """
        if not self._ply_files:
            logger.error("No PLY files loaded")
            return

        self._setup_visualizer()
        self._state = PlaybackState.PLAYING
        self._start_time = time.time()
        self._stop_event.clear()

        # 启动播放线程
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

        logger.info("Starting playback...")
        logger.info("  Space: Play/Pause")
        logger.info("  Left/Right: Previous/Next frame")
        logger.info("  Up/Down: Speed up/down")
        logger.info("  R: Reset, L: Toggle loop, C: Center view")
        logger.info("  Q/Esc: Quit")

        if blocking:
            self._run_blocking()

    def _run_blocking(self):
        """阻塞运行可视化循环。"""
        if self._vis is None:
            return

        while not self._stop_event.is_set():
            if not self._vis.poll_events():
                break
            self._vis.update_renderer()

        self._cleanup()

    def _cleanup(self):
        """清理资源。"""
        self._stop_event.set()
        self._state = PlaybackState.STOPPED

        if self._play_thread is not None:
            self._play_thread.join(timeout=1.0)
            self._play_thread = None

        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None

    def stop(self):
        """停止播放。"""
        self._stop_event.set()
        self._state = PlaybackState.STOPPED

    def pause(self):
        """暂停播放。"""
        self._state = PlaybackState.PAUSED

    def resume(self):
        """恢复播放。"""
        self._state = PlaybackState.PLAYING

    def seek(self, frame_index: int):
        """跳转到指定帧。"""
        if 0 <= frame_index < len(self._ply_files):
            self._update_frame(frame_index)

    def set_fps(self, fps: float):
        """设置播放帧率。"""
        self._fps = max(1.0, min(60.0, fps))

    def set_on_frame_change(self, callback: Callable[[PlaybackInfo], None]):
        """设置帧变化回调。"""
        self._on_frame_change = callback

    def get_info(self) -> PlaybackInfo:
        """获取当前播放信息。"""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        current_file = ""
        if 0 <= self._current_index < len(self._ply_files):
            current_file = self._ply_files[self._current_index].name

        return PlaybackInfo(
            current_frame=self._current_index,
            total_frames=len(self._ply_files),
            current_file=current_file,
            state=self._state,
            fps=self._fps,
            elapsed_time=elapsed,
        )

    def export_video(
        self,
        output_path: Path,
        fps: float | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> bool:
        """导出PLY序列为视频文件。

        Args:
            output_path: 输出视频路径
            fps: 视频帧率（默认使用播放器帧率）
            resolution: 视频分辨率 (width, height)

        Returns:
            是否成功导出
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) required for video export")
            return False

        if not self._ply_files:
            logger.error("No PLY files loaded")
            return False

        fps = fps or self._fps
        width = resolution[0] if resolution else self.config.window_width
        height = resolution[1] if resolution else self.config.window_height

        logger.info(f"Exporting video to {output_path}")
        logger.info(f"  Resolution: {width}x{height}, FPS: {fps}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            for i, ply_path in enumerate(self._ply_files):
                # 使用matplotlib渲染帧
                pcd = self._load_frame(i)
                if pcd is None:
                    continue

                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
                ax = fig.add_subplot(111, projection="3d")
                ax.set_facecolor(self.config.background_color)

                if colors is not None:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        c=colors,
                        s=self.config.point_size,
                    )
                else:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        c="white",
                        s=self.config.point_size,
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Frame {i + 1}/{len(self._ply_files)}")

                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                writer.write(img_bgr)
                plt.close(fig)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Exported {i + 1}/{len(self._ply_files)} frames")

            writer.release()
            logger.info(f"Video exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export video: {e}")
            writer.release()
            return False


class MatplotlibVoxelPlayer:
    """基于matplotlib的体素播放器（Open3D不可用时的fallback）。

    功能相对简化，但可在无Open3D环境下运行。
    """

    def __init__(self, config: PlayerConfig | None = None):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for MatplotlibVoxelPlayer")

        self.config = config or PlayerConfig()
        self._ply_files: list[Path] = []
        self._state = PlaybackState.STOPPED
        self._current_index = 0
        self._fps = self.config.fps
        self._loop = self.config.loop

    def load_sequence(self, input_path: Path, pattern: str = "*.ply") -> int:
        """加载PLY序列。"""
        self._ply_files = []

        if input_path.is_file():
            self._ply_files = [input_path]
        else:
            self._ply_files = sorted(input_path.glob(pattern))

        if not self._ply_files:
            logger.warning(f"No PLY files found in {input_path}")
            return 0

        logger.info(f"Found {len(self._ply_files)} PLY files")
        return len(self._ply_files)

    def _load_ply(self, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        """加载PLY文件，返回(points, colors)。"""
        from plyfile import PlyData

        plydata = PlyData.read(str(path))
        vertex = plydata["vertex"]

        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
        colors = None

        if all(p in vertex.data.dtype.names for p in ("red", "green", "blue")):
            colors = (
                np.column_stack(
                    [vertex["red"], vertex["green"], vertex["blue"]]
                ).astype(np.float64)
                / 255.0
            )

        return points, colors

    def play(self) -> None:
        """开始播放（阻塞）。"""
        if not self._ply_files:
            logger.error("No PLY files loaded")
            return

        plt.ion()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        self._state = PlaybackState.PLAYING
        frame_interval = 1.0 / self._fps

        logger.info("Starting matplotlib playback...")
        logger.info("  Close window to stop")

        try:
            while self._state == PlaybackState.PLAYING:
                points, colors = self._load_ply(self._ply_files[self._current_index])

                ax.clear()
                ax.set_facecolor(self.config.background_color)

                if colors is not None:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        c=colors,
                        s=self.config.point_size,
                    )
                else:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        c="white",
                        s=self.config.point_size,
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Frame {self._current_index + 1}/{len(self._ply_files)}")

                plt.draw()
                plt.pause(frame_interval)

                # 检查窗口是否关闭
                if not plt.fignum_exists(fig.number):
                    break

                self._current_index += 1
                if self._current_index >= len(self._ply_files):
                    if self._loop:
                        self._current_index = 0
                    else:
                        break

        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            plt.close(fig)
            self._state = PlaybackState.STOPPED

    def pause(self) -> None:
        """暂停播放。"""
        self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        """停止播放。"""
        self._state = PlaybackState.STOPPED

    def set_fps(self, fps: float) -> None:
        """设置播放帧率。"""
        self._fps = max(1.0, min(60.0, fps))


def play_sequence(
    input_path: str,
    fps: float = 10.0,
    loop: bool = True,
    preload: bool = False,
) -> None:
    """便捷函数：播放PLY序列。

    自动选择可用的播放器后端（Open3D优先，matplotlib作为fallback）。

    Args:
        input_path: PLY文件目录
        fps: 播放帧率
        loop: 是否循环播放
        preload: 是否预加载到内存（仅Open3D后端支持）

    Example:
        >>> from aylm.tools.voxel_player import play_sequence
        >>> play_sequence("output/voxelized", fps=15, loop=True)
    """
    config = PlayerConfig(fps=fps, loop=loop)

    if HAS_OPEN3D:
        player = VoxelPlayer(config)
        count = player.load_sequence(Path(input_path), preload=preload)
        if count == 0:
            logger.error(f"No PLY files found in {input_path}")
            return
        player.play(blocking=True)
    elif HAS_MATPLOTLIB:
        logger.info("Using matplotlib fallback (Open3D not available)")
        player = MatplotlibVoxelPlayer(config)
        count = player.load_sequence(Path(input_path))
        if count == 0:
            logger.error(f"No PLY files found in {input_path}")
            return
        player.play()
    else:
        logger.error("No visualization backend available (need Open3D or matplotlib)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python voxel_player.py <ply_directory> [fps]")
        sys.exit(1)

    input_dir = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    play_sequence(input_dir, fps=fps)
