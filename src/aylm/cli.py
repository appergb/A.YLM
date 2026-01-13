#!/usr/bin/env python3
"""A.YLM CLI 入口点."""

import sys
from pathlib import Path


def show_usage():
    """显示使用说明."""
    print("A.YLM v1.0.0 - Single-Image 3D Reconstruction and Intelligent Navigation System")
    print("=" * 70)
    print()
    print("主要命令:")
    print("  ./run_sharp.sh              # 一键运行完整流程")
    print("  ./run_sharp.sh --help       # 查看详细帮助")
    print("  ./run_sharp.sh --predict    # 仅运行SHARP预测")
    print("  ./run_sharp.sh --voxelize   # 仅运行体素化")
    print()
    print("Python工具:")
    print("  python -m aylm.tools.pointcloud_voxelizer --help")
    print("  python -m aylm.tools.coordinate_utils --help")
    print("  python -m aylm.tools.preload_sharp_model --help")
    print("  python -m aylm.tools.undistort_iphone --help")


def main_cli():
    """主入口点."""
    if len(sys.argv) == 1:
        show_usage()
        return

    command = sys.argv[1]

    if command in ['--help', '-h']:
        show_usage()
    elif command == 'tools':
        # 显示可用工具
        tools_dir = Path(__file__).parent / 'tools'
        if tools_dir.exists():
            print("可用工具:")
            for tool_file in tools_dir.glob('*.py'):
                if tool_file.name != '__init__.py':
                    tool_name = tool_file.stem
                    print(f"  {tool_name}")
    else:
        print(f"未知命令: {command}")
        print("运行 'python -m aylm.cli --help' 查看帮助")


if __name__ == "__main__":
    main()
