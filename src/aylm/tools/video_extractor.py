"""异步视频帧提取器。

使用OpenCV读取视频，按配置间隔提取帧，支持图像压缩和异步处理。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import cv2

from .video_types import (
    FrameExtractionMethod,
    FrameExtractionResult,
    FrameInfo,
    VideoConfig,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, float], None]


class VideoExtractor:
    """异步视频帧提取器。"""

    def __init__(
        self,
        config: VideoConfig | None = None,
        input_dir: Path | None = None,
        output_base_dir: Path | None = None,
    ) -> None:
        from .video_config import get_default_config

        self.config = config or get_default_config()
        self.input_dir = input_dir or Path("inputs/videos")
        self.output_base_dir = output_base_dir or Path("inputs/extracted_frames")

    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """获取视频元数据。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return VideoMetadata(
                path=video_path,
                duration=total_frames / fps if fps > 0 else 0.0,
                fps=fps,
                total_frames=total_frames,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        finally:
            cap.release()

    def _calculate_frame_indices(
        self, metadata: VideoMetadata
    ) -> list[tuple[int, float]]:
        """计算要提取的帧索引，返回 (帧索引, 时间戳) 列表。"""
        method = self.config.frame_extraction_method
        fps, total_frames = metadata.fps, metadata.total_frames

        # 计算帧间隔
        interval_float: float
        if method == FrameExtractionMethod.INTERVAL:
            interval_float = float(max(1, int(self.config.frame_interval * fps)))
        elif (
            method == FrameExtractionMethod.UNIFORM
            and self.config.target_fps
            and self.config.target_fps < fps
        ):
            interval_float = fps / self.config.target_fps
        else:
            interval_float = float(
                max(1, int(fps))
            )  # KEYFRAME 和 SCENE_CHANGE 默认每秒一帧

        # 生成帧索引
        if method == FrameExtractionMethod.UNIFORM and (
            not self.config.target_fps or self.config.target_fps >= fps
        ):
            indices = [(i, i / fps) for i in range(total_frames)]
        elif interval_float != int(interval_float):
            # 非整数间隔，使用浮点累加
            indices = []
            i = 0.0
            while int(i) < total_frames:
                frame_idx = int(i)
                indices.append((frame_idx, frame_idx / fps))
                i += interval_float
        else:
            interval_int = int(interval_float)
            indices = [(i, i / fps) for i in range(0, total_frames, interval_int)]

        # 限制最大帧数
        if self.config.max_frames:
            indices = indices[: self.config.max_frames]

        return indices

    def compress_frame(self, frame: Any) -> Any:
        """压缩帧图像（调整大小）。"""
        if self.config.resize_width is None and self.config.resize_height is None:
            return frame

        h, w = frame.shape[:2]
        new_w = self.config.resize_width or w
        new_h = self.config.resize_height or h

        if self.config.keep_aspect_ratio:
            aspect = w / h
            if self.config.resize_width and self.config.resize_height:
                scale = min(self.config.resize_width / w, self.config.resize_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
            elif self.config.resize_width:
                new_h = int(new_w / aspect)
            else:
                new_w = int(new_h * aspect)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _get_image_write_params(self) -> list[int]:
        """获取图像写入参数。"""
        fmt = self.config.output_format.lower()
        quality = self.config.quality
        if fmt in ("jpg", "jpeg"):
            return [cv2.IMWRITE_JPEG_QUALITY, quality]
        if fmt == "webp":
            return [cv2.IMWRITE_WEBP_QUALITY, quality]
        if fmt == "png":
            return [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - quality) // 10))]
        return []

    def extract_frame(
        self,
        video_path: Path,
        frame_index: int,
        output_path: Path,
    ) -> FrameInfo | None:
        """提取单帧。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Cannot read frame {frame_index} from {video_path}")
                return None

            compressed = self.compress_frame(frame)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), compressed, self._get_image_write_params())

            fps = cap.get(cv2.CAP_PROP_FPS)
            return FrameInfo(
                index=frame_index,
                timestamp=frame_index / fps if fps > 0 else 0.0,
                output_path=output_path,
            )
        finally:
            cap.release()

    def _extract_frames_batch(
        self,
        video_path: Path,
        frame_indices: list[tuple[int, float]],
        output_dir: Path,
    ) -> list[FrameInfo]:
        """批量提取帧（单线程）。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        frames: list[FrameInfo] = []
        output_format = self.config.output_format.lower()
        params = self._get_image_write_params()

        try:
            for frame_index, timestamp in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Cannot read frame {frame_index} from {video_path}")
                    continue

                compressed = self.compress_frame(frame)
                output_path = output_dir / f"frame_{frame_index:06d}.{output_format}"
                cv2.imwrite(str(output_path), compressed, params)

                frames.append(
                    FrameInfo(
                        index=frame_index, timestamp=timestamp, output_path=output_path
                    )
                )
        finally:
            cap.release()

        return frames

    def _prepare_extraction(
        self, video_path: Path, output_dir: Path | None
    ) -> tuple[VideoMetadata | None, Path, str | None]:
        """准备帧提取，返回 (元数据, 输出目录, 错误信息)。"""
        try:
            metadata = self.get_video_metadata(video_path)
        except ValueError as e:
            return None, output_dir or Path("."), str(e)

        if output_dir is None:
            output_dir = self.output_base_dir / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        return metadata, output_dir, None

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> FrameExtractionResult:
        """同步提取视频帧。"""
        start_time = time.time()
        metadata, output_dir, error = self._prepare_extraction(video_path, output_dir)

        if error or metadata is None:
            return FrameExtractionResult(
                video_path=video_path, output_dir=output_dir, error_message=error
            )

        frame_indices = self._calculate_frame_indices(metadata)

        if self.config.verbose:
            logger.info(
                f"Extracting {len(frame_indices)} frames from {video_path.name} to {output_dir}"
            )

        frames = self._extract_frames_batch(video_path, frame_indices, output_dir)

        if progress_callback:
            progress_callback(len(frames), len(frame_indices), 100.0)

        return FrameExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            frames=frames,
            total_extracted=len(frames),
            extraction_time=time.time() - start_time,
            metadata=metadata,
        )

    def extract_frames_async(
        self,
        video_path: Path,
        output_dir: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        max_workers: int | None = None,
    ) -> FrameExtractionResult:
        """异步提取视频帧（使用线程池）。"""
        start_time = time.time()
        metadata, output_dir, error = self._prepare_extraction(video_path, output_dir)

        if error or metadata is None:
            return FrameExtractionResult(
                video_path=video_path, output_dir=output_dir, error_message=error
            )

        frame_indices = self._calculate_frame_indices(metadata)
        total_frames = len(frame_indices)

        if self.config.verbose:
            logger.info(
                f"Async extracting {total_frames} frames from {video_path.name} to {output_dir}"
            )

        workers = max_workers or self.config.parallel_extraction
        frames: list[FrameInfo] = []
        output_format = self.config.output_format.lower()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.extract_frame,
                    video_path,
                    frame_index,
                    output_dir / f"frame_{frame_index:06d}.{output_format}",
                ): frame_index
                for frame_index, _ in frame_indices
            }

            for i, future in enumerate(as_completed(futures), 1):
                if frame_info := future.result():
                    frames.append(frame_info)
                if progress_callback:
                    progress_callback(i, total_frames, (i / total_frames) * 100)

        frames.sort(key=lambda f: f.index)

        return FrameExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            frames=frames,
            total_extracted=len(frames),
            extraction_time=time.time() - start_time,
            metadata=metadata,
        )


def extract_video_frames(
    video_path: Path,
    config: VideoConfig | None = None,
    output_dir: Path | None = None,
    async_mode: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> FrameExtractionResult:
    """便捷函数：提取视频帧。"""
    extractor = VideoExtractor(config=config)
    if async_mode:
        return extractor.extract_frames_async(video_path, output_dir, progress_callback)
    return extractor.extract_frames(video_path, output_dir, progress_callback)
