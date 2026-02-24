#!/bin/bash
# 多帧点云融合脚本
# 用法: ./fuse.sh [输入目录] [输出文件]

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认路径
DEFAULT_INPUT="outputs/video_output/voxelized"
DEFAULT_OUTPUT="outputs/video_output/fused_map.ply"

# 解析参数
INPUT_DIR="${1:-$DEFAULT_INPUT}"
OUTPUT_FILE="${2:-$DEFAULT_OUTPUT}"

# 显示帮助
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "多帧点云融合脚本"
    echo ""
    echo "用法: ./fuse.sh [输入目录] [输出文件] [选项]"
    echo ""
    echo "参数:"
    echo "  输入目录    包含 vox_*.ply 文件的目录 (默认: $DEFAULT_INPUT)"
    echo "  输出文件    融合后的点云文件路径 (默认: $DEFAULT_OUTPUT)"
    echo ""
    echo "选项:"
    echo "  -h, --help  显示帮助"
    echo ""
    echo "示例:"
    echo "  ./fuse.sh                                    # 使用默认路径"
    echo "  ./fuse.sh outputs/my_scan                    # 指定输入目录"
    echo "  ./fuse.sh outputs/my_scan my_map.ply         # 指定输入和输出"
    echo ""
    echo "高级用法 (直接调用 aylm):"
    echo "  aylm fuse -i 输入目录 -o 输出文件 --icp-distance 0.1 --voxel-size 0.05"
    exit 0
fi

# 检查输入目录
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${YELLOW}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 统计点云文件
PLY_COUNT=$(find "$INPUT_DIR" -name "vox_*.ply" 2>/dev/null | wc -l | tr -d ' ')

if [[ "$PLY_COUNT" -eq 0 ]]; then
    echo -e "${YELLOW}错误: 未找到 vox_*.ply 文件${NC}"
    echo "目录: $INPUT_DIR"
    exit 1
fi

# 显示信息
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}多帧点云融合${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "输入目录: ${GREEN}$INPUT_DIR${NC}"
echo -e "点云数量: ${GREEN}$PLY_COUNT${NC} 帧"
echo -e "输出文件: ${GREEN}$OUTPUT_FILE${NC}"
echo ""

# 列出点云文件
echo "点云文件:"
find "$INPUT_DIR" -name "vox_*.ply" | sort | while read f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $(basename "$f") ($SIZE)"
done
echo ""

# 激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -d "$SCRIPT_DIR/aylm_env" ]]; then
    source "$SCRIPT_DIR/aylm_env/bin/activate"
fi

# 执行融合
echo -e "${BLUE}开始融合...${NC}"
echo ""

aylm fuse -i "$INPUT_DIR" -o "$OUTPUT_FILE" -v

# 检查结果
if [[ -f "$OUTPUT_FILE" ]]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}融合完成!${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo -e "输出文件: ${GREEN}$OUTPUT_FILE${NC} ($OUTPUT_SIZE)"

    # 检查位姿文件
    POSES_FILE="${OUTPUT_FILE%.ply}.poses.json"
    if [[ -f "$POSES_FILE" ]]; then
        echo -e "位姿轨迹: ${GREEN}$POSES_FILE${NC}"
    fi

    echo ""
    echo "查看结果:"
    echo "  open $OUTPUT_FILE          # macOS 默认应用打开"
    echo "  meshlab $OUTPUT_FILE       # 使用 MeshLab 查看"
fi
