#!/usr/bin/env python3
"""AYLM CLI - 命令行接口.

提供子命令:
- setup: 检查环境、安装依赖、下载模型
- predict: 运行SHARP预测
- voxelize: 运行体素化
- process: 完整流程
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_version() -> str:
    """获取版本号."""
    try:
        from importlib.metadata import version

        return version("aylm")
    except Exception:
        return "2.0.0"


def get_project_root() -> Path:
    """获取项目根目录."""
    if "AYLM_ROOT" in os.environ:
        return Path(os.environ["AYLM_ROOT"])
    current = Path(__file__).resolve().parent.parent.parent.parent
    if (current / "pyproject.toml").exists():
        return current
    return Path.cwd()


def print_step(counter: list, msg: str):
    """打印步骤信息，使用动态计数器."""
    counter[0] += 1
    print(f"[{counter[0]}] {msg}")


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
    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"    + {dep}")
        except ImportError:
            print(f"    - {dep} (未安装)")
            missing.append(dep)

    if missing:
        logger.warning("部分依赖未安装，请运行: pip install -e .")

    print_step(step, "检查模型检查点...")
    model_url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    model_path = get_project_root() / "models" / "sharp_2572gikvuh.pt"

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

    exts = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]
    if not images:
        logger.error("未找到图像文件")
        return 1
    print(f"    找到 {len(images)} 张图像")

    print_step(step, "检查模型...")
    model_path = root / "models" / "sharp_2572gikvuh.pt"
    if not model_path.exists():
        logger.error("模型不存在，请先运行: aylm setup --download")
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
        if not args.verbose:
            logger.error(f"预测失败: {result.stderr.decode() if result.stderr else '未知错误'}")
        else:
            logger.error("预测失败")
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
    if input_path.is_file():
        ply_files = [input_path]
    else:
        ply_files = [
            f for f in input_path.glob("*.ply") if not f.name.startswith("vox")
        ]
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
    config = VoxelizerConfig(voxel_size=args.voxel_size)
    processor = PointCloudVoxelizer(config=config)
    success = 0

    for ply in ply_files:
        out = output_dir / f"vox_{ply.name}"
        try:
            processor.process(ply, out, transform_coords=False)
            success += 1
            print(f"    + {ply.name}")
        except Exception as e:
            print(f"    - {ply.name} (失败: {e})")

    print(f"\n体素化完成! {success}/{len(ply_files)} 成功")
    return 0 if success == len(ply_files) else 1


def cmd_process(args: argparse.Namespace) -> int:
    """运行完整流程."""
    print(f"AYLM v{get_version()} - 完整流程\n")

    print("=" * 40)
    print("阶段 1/3: 环境设置")
    print("=" * 40)
    setup_args = argparse.Namespace(download=True)
    if cmd_setup(setup_args) != 0:
        return 1

    print("\n" + "=" * 40)
    print("阶段 2/3: SHARP预测")
    print("=" * 40)
    predict_args = argparse.Namespace(
        input=args.input, output=args.output, verbose=args.verbose
    )
    if cmd_predict(predict_args) != 0:
        return 1

    print("\n" + "=" * 40)
    print("阶段 3/3: 体素化")
    print("=" * 40)
    root = get_project_root()
    vox_input = Path(args.output) if args.output else root / "outputs/output_gaussians"
    vox_args = argparse.Namespace(
        input=str(vox_input), output=None, voxel_size=0.005
    )
    if cmd_voxelize(vox_args) != 0:
        return 1

    print("\n完整流程完成!")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器."""
    parser = argparse.ArgumentParser(
        prog="aylm",
        description="AYLM - 3D Gaussian Splatting 工具",
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

    return parser


def main():
    """CLI入口点."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
