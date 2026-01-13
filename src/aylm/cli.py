#!/usr/bin/env python3
"""A.YLM CLI 入口点."""


def main_cli():
    """主CLI入口点."""
    print(
        "A.YLM v1.0.0 - Single-Image 3D Reconstruction and Intelligent Navigation System"
    )
    print("=" * 70)
    print("这个CLI接口正在开发中...")
    print()
    print("目前请使用以下脚本:")
    print("  ./run_sharp.sh              # 一键运行完整流程")
    print("  ./run_sharp.sh --help       # 查看帮助信息")
    print("  ./run_sharp.sh --predict    # 仅运行预测")
    print("  ./run_sharp.sh --voxelize   # 仅运行体素化")
    print()
    print("或者直接使用Python工具模块:")
    print("  python -m aylm.tools.preload_sharp_model --help")
    print("  python -m aylm.tools.pointcloud_voxelizer --help")


if __name__ == "__main__":
    main_cli()
