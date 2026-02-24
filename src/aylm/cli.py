#!/usr/bin/env python3
"""AYLM CLI - 命令行接口."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 支持的图像扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def get_version() -> str:
    """获取版本号."""
    try:
        from importlib.metadata import version

        return version("aylm")
    except Exception:
        return "2.0.0"


def get_project_root() -> Path:
    """获取项目根目录."""
    if root := os.environ.get("AYLM_ROOT"):
        return Path(root)
    current = Path(__file__).resolve().parents[3]
    return current if (current / "pyproject.toml").exists() else Path.cwd()


def get_model_path() -> Path:
    """获取模型路径."""
    return get_project_root() / "models" / "sharp_2572gikvuh.pt"


def print_step(counter: list[int], msg: str) -> None:
    """打印步骤信息."""
    counter[0] += 1
    print(f"[{counter[0]}] {msg}")


def check_model_exists() -> bool:
    """检查模型是否存在."""
    return get_model_path().exists()


def require_model() -> Path | None:
    """检查模型，不存在则报错并返回 None."""
    model_path = get_model_path()
    if not model_path.exists():
        logger.error("模型不存在，请先运行: aylm setup --download")
        return None
    return model_path


def get_images(input_dir: Path) -> list[Path]:
    """获取目录中的图像文件."""
    return sorted(f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def cmd_setup(args: argparse.Namespace) -> int:
    """检查环境、安装依赖、下载模型."""
    step = [0]
    print(f"AYLM v{get_version()} - 环境设置\n")

    print_step(step, "检查Python版本...")
    v = sys.version_info
    if v < (3, 9):
        logger.error(f"Python {v.major}.{v.minor} 不兼容，需要 3.9+")
        return 1
    print(f"    Python {v.major}.{v.minor} OK")

    print_step(step, "检查依赖...")
    deps = ["numpy", "torch", "scipy", "plyfile", "PIL", "cv2"]
    missing = [dep for dep in deps if not _check_import(dep)]
    if missing:
        logger.warning("部分依赖未安装，请运行: pip install -e .")

    print_step(step, "检查模型检查点...")
    model_url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    model_path = get_model_path()

    if model_path.exists():
        print(f"    模型已存在: {model_path}")
    elif args.download:
        print(f"    下载模型到: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(model_path), model_url],
                check=True,
                timeout=600,
            )
            print("    下载完成")
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return 1
    else:
        print("    模型不存在，使用 --download 下载")

    print_step(step, "创建目录结构...")
    root = get_project_root()
    for d in ["inputs/input_images", "outputs/output_gaussians", "models"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    print("    目录已创建")

    print("\n设置完成!")
    return 0


def _check_import(module: str) -> bool:
    """检查模块是否可导入."""
    try:
        __import__(module)
        print(f"    + {module}")
        return True
    except ImportError:
        print(f"    - {module} (未安装)")
        return False


def cmd_predict(args: argparse.Namespace) -> int:
    """运行SHARP预测."""
    step = [0]
    print(f"AYLM v{get_version()} - SHARP预测\n")

    root = get_project_root()
    input_dir = Path(args.input) if args.input else root / "inputs/input_images"
    output_dir = Path(args.output) if args.output else root / "outputs/output_gaussians"

    print_step(step, "检查输入...")
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return 1

    images = get_images(input_dir)
    if not images:
        logger.error("未找到图像文件")
        return 1
    print(f"    找到 {len(images)} 张图像")

    print_step(step, "检查模型...")
    if not (model_path := require_model()):
        return 1
    print(f"    模型: {model_path}")

    print_step(step, "运行预测...")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sharp",
        "predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-c",
        str(model_path),
    ]
    result = subprocess.run(cmd, capture_output=not args.verbose)
    if result.returncode != 0:
        err = (
            result.stderr.decode() if result.stderr and not args.verbose else "未知错误"
        )
        logger.error(f"预测失败: {err}" if not args.verbose else "预测失败")
        return 1

    print(f"\n预测完成! 输出: {output_dir}")
    return 0


def cmd_voxelize(args: argparse.Namespace) -> int:
    """运行体素化."""
    step = [0]
    print(f"AYLM v{get_version()} - 体素化\n")

    root = get_project_root()
    input_path = Path(args.input) if args.input else root / "outputs/output_gaussians"
    output_dir = Path(args.output) if args.output else input_path / "voxelized"

    print_step(step, "检查输入...")
    ply_files = (
        [input_path]
        if input_path.is_file()
        else [f for f in input_path.glob("*.ply") if not f.name.startswith("vox")]
    )
    if not ply_files:
        logger.error(f"未找到PLY文件: {input_path}")
        return 1
    print(f"    找到 {len(ply_files)} 个PLY文件")

    print_step(step, "导入体素化模块...")
    try:
        from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        return 1

    print_step(step, "处理文件...")
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = PointCloudVoxelizer(config=VoxelizerConfig(voxel_size=args.voxel_size))
    success = 0

    for ply in ply_files:
        try:
            processor.process(
                ply, output_dir / f"vox_{ply.name}", transform_coords=False
            )
            success += 1
            print(f"    + {ply.name}")
        except Exception as e:
            print(f"    - {ply.name} (失败: {e})")

    print(f"\n体素化完成! {success}/{len(ply_files)} 成功")
    return 0 if success == len(ply_files) else 1


def cmd_process(args: argparse.Namespace) -> int:
    """运行完整流程."""
    print(f"AYLM v{get_version()} - 完整流程\n")

    stages = [
        ("1/3: 环境设置", cmd_setup, argparse.Namespace(download=True)),
        (
            "2/3: SHARP预测",
            cmd_predict,
            argparse.Namespace(
                input=args.input, output=args.output, verbose=args.verbose
            ),
        ),
    ]

    for title, func, stage_args in stages:
        print("=" * 40)
        print(f"阶段 {title}")
        print("=" * 40)
        if func(stage_args) != 0:
            return 1
        print()

    print("=" * 40)
    print("阶段 3/3: 体素化")
    print("=" * 40)
    vox_input = (
        Path(args.output)
        if args.output
        else get_project_root() / "outputs/output_gaussians"
    )
    if (
        cmd_voxelize(
            argparse.Namespace(input=str(vox_input), output=None, voxel_size=0.005)
        )
        != 0
    ):
        return 1

    print("\n完整流程完成!")
    return 0


def cmd_video_process(args: argparse.Namespace) -> int:
    """处理视频文件，提取帧并进行3D重建."""
    step = [0]
    print(f"AYLM v{get_version()} - 视频处理\n")

    video_path = Path(args.input)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return 1

    print_step(step, "检查输入...")
    print(f"    视频: {video_path}")

    print_step(step, "加载配置...")
    try:
        from aylm.tools.video_config import load_or_create_config
        from aylm.tools.video_types import FrameExtractionMethod, GPUAcceleration

        # 确定配置文件路径：优先使用命令行指定，否则查找视频目录下的配置
        config_path = None
        if args.config:
            config_path = Path(args.config)
        else:
            # 自动查找视频所在目录的配置文件
            video_dir_config = video_path.parent / "video_config.yaml"
            if video_dir_config.exists():
                config_path = video_dir_config
                print(f"    发现配置文件: {config_path}")

        config = load_or_create_config(config_path)
        if args.frame_interval:
            config.frame_interval = args.frame_interval
            config.frame_extraction_method = FrameExtractionMethod.INTERVAL
        if args.use_gpu:
            config.gpu_acceleration = GPUAcceleration.AUTO
        print(f"    帧提取方法: {config.frame_extraction_method.value}")
        print(f"    帧间隔: {config.frame_interval}s")
        print(f"    GPU加速: {config.gpu_acceleration.value}")
    except ImportError as e:
        logger.error(f"导入配置模块失败: {e}")
        return 1

    output_dir = (
        Path(args.output)
        if args.output
        else get_project_root() / "outputs/video_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"    输出目录: {output_dir}")

    if not (model_path := require_model()):
        return 1
    print(f"    模型: {model_path.name}")

    print_step(step, "启动视频处理流水线...")
    try:
        from aylm.tools.video_pipeline import (
            VideoPipelineConfig,
            VideoPipelineProcessor,
        )

        # 解析语义检测选项
        enable_semantic = args.semantic and not getattr(args, "no_semantic", False)
        # 解析切片选项
        enable_slice = getattr(args, "slice", True) and not getattr(
            args, "no_slice", False
        )

        pipeline_config = VideoPipelineConfig(
            video_config=config,
            use_gpu=args.use_gpu,
            verbose=args.verbose,
            checkpoint_path=model_path,
            enable_semantic=enable_semantic,
            semantic_model=args.semantic_model,
            semantic_confidence=args.semantic_confidence,
            enable_slice=enable_slice,
            slice_radius=getattr(args, "slice_radius", 10.0),
        )
        stats = VideoPipelineProcessor(pipeline_config).process(video_path, output_dir)

        print("\n视频处理完成!")
        print(f"  提取帧数: {stats.total_frames_extracted}")
        print(f"  处理帧数: {stats.total_frames_processed}")
        print(f"  总耗时: {stats.total_time:.1f}s")
        return 0 if stats.failed_videos == 0 else 1

    except ImportError as e:
        logger.error(f"导入视频处理模块失败: {e}")
        logger.error("请确保已安装所有依赖: pip install -e .")
        return 1
    except Exception as e:
        logger.error(f"视频处理失败: {e}")
        return 1


def cmd_video_extract(args: argparse.Namespace) -> int:
    """从视频中提取帧."""
    step = [0]
    print(f"AYLM v{get_version()} - 视频帧提取\n")

    video_path = Path(args.input)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        return 1

    print_step(step, "检查输入...")
    print(f"    视频: {video_path}")

    output_dir = (
        Path(args.output)
        if args.output
        else get_project_root() / "outputs/extracted_frames"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"    输出目录: {output_dir}")

    print_step(step, "加载配置...")
    try:
        from aylm.tools.video_config import load_or_create_config
        from aylm.tools.video_types import FrameExtractionMethod

        # 确定配置文件路径：优先使用命令行指定，否则查找视频目录下的配置
        config_path = None
        if args.config:
            config_path = Path(args.config)
        else:
            # 自动查找视频所在目录的配置文件
            video_dir_config = video_path.parent / "video_config.yaml"
            if video_dir_config.exists():
                config_path = video_dir_config
                print(f"    发现配置文件: {config_path}")

        config = load_or_create_config(config_path)
        if args.frame_interval:
            config.frame_interval = args.frame_interval
            config.frame_extraction_method = FrameExtractionMethod.INTERVAL
        print(f"    帧提取方法: {config.frame_extraction_method.value}")
        print(f"    帧间隔: {config.frame_interval}s")
    except ImportError as e:
        logger.error(f"导入配置模块失败: {e}")
        return 1

    print_step(step, "提取帧...")
    try:
        from aylm.tools.video_extractor import VideoExtractor

        result = VideoExtractor(config).extract_frames(video_path, output_dir)

        if result.success:
            print("\n帧提取完成!")
            print(f"  提取帧数: {result.total_extracted}")
            print(f"  耗时: {result.extraction_time:.1f}s")
            print(f"  输出目录: {output_dir}")
            return 0
        logger.error(f"帧提取失败: {result.error_message}")
        return 1

    except ImportError as e:
        logger.error(f"导入帧提取模块失败: {e}")
        logger.error("请确保已安装所有依赖: pip install -e .")
        return 1
    except Exception as e:
        logger.error(f"帧提取失败: {e}")
        return 1


def cmd_video_play(args: argparse.Namespace) -> int:
    """播放体素序列."""
    step = [0]
    print(f"AYLM v{get_version()} - 体素序列播放\n")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        return 1

    print_step(step, "检查输入...")
    if not input_path.is_dir():
        logger.error("输入必须是包含PLY文件的目录")
        return 1

    ply_files = sorted(input_path.glob("*.ply"))
    if not ply_files:
        logger.error(f"目录中未找到PLY文件: {input_path}")
        return 1
    print(f"    目录: {input_path}")
    print(f"    PLY文件数: {len(ply_files)}")

    print_step(step, "启动播放器...")
    try:
        from aylm.tools.voxel_player import PlayerConfig, VoxelPlayer

        player = VoxelPlayer(PlayerConfig(fps=args.fps, loop=args.loop))
        player.load_sequence(input_path)
        player.play()
        print("\n播放结束")
        return 0

    except ImportError as e:
        logger.error(f"导入播放器模块失败: {e}")
        logger.error("请确保已安装所有依赖: pip install -e .")
        return 1
    except Exception as e:
        logger.error(f"播放失败: {e}")
        return 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """运行流水线处理（推理与体素化并行）."""
    verbose = not args.quiet if args.quiet else args.verbose
    print(f"AYLM v{get_version()} - 流水线处理\n")

    root = get_project_root()
    input_dir = Path(args.input) if args.input else root / "inputs/input_images"
    output_dir = Path(args.output) if args.output else root / "outputs/output_gaussians"

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return 1

    images = get_images(input_dir)
    if not images:
        logger.error(f"未找到图像文件: {input_dir}")
        return 1

    print(f"[输入] 目录: {input_dir}")
    print(f"[输入] 图像数量: {len(images)}")
    if verbose:
        for img in images[: 10 if len(images) <= 10 else 5]:
            print(f"       - {img.name}")
        if len(images) > 10:
            print(f"       ... 还有 {len(images) - 5} 张")

    if not (model_path := require_model()):
        return 1
    print(f"[模型] {model_path.name}")

    # 解析语义检测和切片选项
    enable_semantic = args.semantic and not getattr(args, "no_semantic", False)
    enable_slice = args.slice and not getattr(args, "no_slice", False)

    print("\n[配置]")
    print(f"  体素尺寸: {args.voxel_size}m")
    print(f"  移除地面: {'否' if args.keep_ground else '是'}")
    print(f"  坐标转换: {'是' if args.transform else '否'}")
    print(f"  语义检测: {'是' if enable_semantic else '否'}")
    if enable_semantic:
        print(f"    模型: {args.semantic_model}")
        print(f"    置信度: {args.semantic_confidence}")
    print(f"  点云切片: {'是' if enable_slice else '否'}")
    if enable_slice:
        print(f"    半径: {args.slice_radius}m")
    print(f"  详细输出: {'是' if verbose else '否'}")

    print("\n[流水线策略]")
    print("  模式: 推理与体素化并行")
    print("  线程: 推理(主线程/GPU) + 体素化(工作线程/CPU)")
    if len(images) >= 2:
        print(f"  预计并行阶段: {len(images) - 1} 次")

    print(f"\n[输出] {output_dir}")
    print("=" * 50)

    try:
        from aylm.tools.pipeline_processor import PipelineConfig, PipelineProcessor

        config = PipelineConfig(
            voxel_size=args.voxel_size,
            remove_ground=not args.keep_ground,
            transform_coords=args.transform,
            checkpoint_path=model_path,
            verbose=verbose,
            enable_semantic=enable_semantic,
            semantic_model=args.semantic_model,
            semantic_confidence=args.semantic_confidence,
            enable_slice=enable_slice,
            slice_radius=args.slice_radius,
        )
        stats = PipelineProcessor(config).process(input_dir, output_dir)

        print("\n" + "=" * 50)
        print("[完成] 流水线处理结束")
        print(f"  成功: {stats.completed_images}/{stats.total_images}")
        if stats.failed_images > 0:
            print(f"  失败: {stats.failed_images}")
        print(f"  总耗时: {stats.total_time:.1f}s")
        if stats.completed_images > 0:
            print(f"  平均: {stats.total_time / stats.completed_images:.1f}s/张")

        return 0 if stats.failed_images == 0 else 1

    except ImportError as e:
        logger.error(f"导入流水线模块失败: {e}")
        logger.error("请确保已安装所有依赖: pip install -e .")
        return 1
    except Exception as e:
        logger.error(f"流水线处理失败: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器."""
    parser = argparse.ArgumentParser(
        prog="aylm", description="AYLM - 3D Gaussian Splatting 工具"
    )
    parser.add_argument("-V", "--version", action="version", version=get_version())
    subs = parser.add_subparsers(dest="command", title="命令")

    # setup
    p = subs.add_parser("setup", help="检查环境、安装依赖、下载模型")
    p.add_argument("--download", action="store_true", help="下载模型检查点")
    p.set_defaults(func=cmd_setup)

    # predict
    p = subs.add_parser("predict", help="运行SHARP预测")
    p.add_argument("-i", "--input", help="输入图像目录")
    p.add_argument("-o", "--output", help="输出目录")
    p.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    p.set_defaults(func=cmd_predict)

    # voxelize
    p = subs.add_parser("voxelize", help="运行体素化")
    p.add_argument("-i", "--input", help="输入PLY文件或目录")
    p.add_argument("-o", "--output", help="输出目录")
    p.add_argument("--voxel-size", type=float, default=0.005, help="体素尺寸(米)")
    p.set_defaults(func=cmd_voxelize)

    # process
    p = subs.add_parser("process", help="运行完整流程")
    p.add_argument("-i", "--input", help="输入图像目录")
    p.add_argument("-o", "--output", help="输出目录")
    p.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    p.set_defaults(func=cmd_process)

    # pipeline
    p = subs.add_parser("pipeline", help="流水线处理（推理与体素化并行）")
    p.add_argument("-i", "--input", help="输入图像目录")
    p.add_argument("-o", "--output", help="输出目录")
    p.add_argument("--voxel-size", type=float, default=0.005, help="体素尺寸(米)")
    p.add_argument("--keep-ground", action="store_true", help="保留地面点")
    p.add_argument("--transform", action="store_true", help="转换到机器人坐标系")
    # 语义检测选项
    p.add_argument(
        "--semantic", action="store_true", default=True, help="启用语义检测（默认开启）"
    )
    p.add_argument("--no-semantic", action="store_true", help="禁用语义检测")
    p.add_argument("--semantic-model", default="yolo11n-seg.pt", help="YOLO模型名称")
    p.add_argument("--semantic-confidence", type=float, default=0.25, help="检测置信度")
    # 切片选项
    p.add_argument("--slice", action="store_true", default=True, help="启用点云切片")
    p.add_argument("--no-slice", action="store_true", help="禁用点云切片")
    p.add_argument("--slice-radius", type=float, default=10.0, help="切片半径(米)")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出（默认开启）",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="安静模式（禁用详细输出）"
    )
    p.set_defaults(func=cmd_pipeline)

    # video 子命令组
    video_parser = subs.add_parser("video", help="视频处理命令")
    video_subs = video_parser.add_subparsers(dest="video_command", title="视频子命令")

    # video process
    p = video_subs.add_parser("process", help="处理视频文件，提取帧并进行3D重建")
    p.add_argument("-i", "--input", required=True, help="输入视频文件")
    p.add_argument("-o", "--output", help="输出目录")
    p.add_argument("-c", "--config", help="配置文件路径(YAML)")
    p.add_argument("--frame-interval", type=float, help="帧间隔(秒)，覆盖配置文件设置")
    p.add_argument("--compression", type=float, default=1.0, help="压缩倍数(默认1.0)")
    p.add_argument("--use-gpu", action="store_true", help="启用GPU加速")
    # 语义检测选项
    p.add_argument("--semantic", action="store_true", default=True, help="启用语义检测")
    p.add_argument("--no-semantic", action="store_true", help="禁用语义检测")
    p.add_argument("--semantic-model", default="yolo11n-seg.pt", help="YOLO模型名称")
    p.add_argument("--semantic-confidence", type=float, default=0.25, help="检测置信度")
    # 点云切片选项
    p.add_argument("--slice", action="store_true", default=True, help="启用点云切片")
    p.add_argument("--no-slice", action="store_true", help="禁用点云切片")
    p.add_argument("--slice-radius", type=float, default=10.0, help="切片半径(米)")
    p.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    p.set_defaults(func=cmd_video_process)

    # video extract
    p = video_subs.add_parser("extract", help="从视频中提取帧")
    p.add_argument("-i", "--input", required=True, help="输入视频文件")
    p.add_argument("-o", "--output", help="输出目录")
    p.add_argument("-c", "--config", help="配置文件路径(YAML)")
    p.add_argument("--frame-interval", type=float, help="帧间隔(秒)，覆盖配置文件设置")
    p.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    p.set_defaults(func=cmd_video_extract)

    # video play
    p = video_subs.add_parser("play", help="播放体素序列")
    p.add_argument("-i", "--input", required=True, help="体素序列目录")
    p.add_argument("--fps", type=float, default=10.0, help="播放帧率(默认10)")
    p.add_argument("--loop", action="store_true", help="循环播放")
    p.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    p.set_defaults(func=cmd_video_play)

    return parser


def main() -> int:
    """CLI入口点."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "video" and (
        not hasattr(args, "video_command") or args.video_command is None
    ):
        print("请指定视频子命令: process, extract, play")
        print("使用 'aylm video --help' 查看详细帮助")
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
