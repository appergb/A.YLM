#!/bin/bash

# SHARP 一键启动脚本
# 用于部署Apple SHARP模型，生成单张照片的3D高斯模型
# 支持多种智能设备导航应用
#
# 作者: TRIP(appergb)
# 项目参与者: closer, true
# 个人研发项目

set -e  # 遇到错误立即退出
#
# 启动日志与缓存清理策略:
# 1) 启动时删除旧的日志文件（保留本次运行日志）。
# 2) 将当前运行输出写入时间戳日志文件（保存在 $OUTPUT_DIR/logs）。
# 3) 运行完成后清理临时文件与包缓存（但保留本次运行日志以供排查）。

# 激活虚拟环境并设置日志
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "已激活Python虚拟环境 ($(python3 --version))"
else
    echo "警告: 未找到虚拟环境，请确保已正确安装依赖"
fi

# 确保基础路径变量已定义（防止 OUTPUT_DIR 在此之前未设置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/output_gaussians}"
mkdir -p "$OUTPUT_DIR"

# 日志目录与当前运行日志文件
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 启动时清理旧日志（保留当前运行日志目录结构）
if [ -d "$LOG_DIR" ]; then
    rm -f "$LOG_DIR"/*.log || true
fi

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"

# 将后续 stdout/stderr 写入日志文件，同时保留控制台输出
exec > >(tee -a "$LOG_FILE") 2>&1

# 配置参数（支持环境变量覆盖）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARP_DIR="$SCRIPT_DIR/ml-sharp"
VOXELIZER_SCRIPT="$SCRIPT_DIR/scripts/pointcloud_voxelizer.py"
INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/inputs/input_images}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/output_gaussians}"
CHECKPOINT_URL="https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
CHECKPOINT_FILE="${SHARP_MODEL_PATH:-$SCRIPT_DIR/models/sharp_2572gikvuh.pt}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}                A.YLM v1.0.0${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Apple SHARP 3D高斯模型生成工具${NC}"
    echo -e "${BLUE}  Single-image High-Accuracy Real-time Parallax${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_step() {
    echo -e "${GREEN}[步骤 $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

check_dependencies() {
    print_step "1" "检查系统依赖..."

    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未找到，请安装 Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
        echo -e "Python $PYTHON_VERSION 版本兼容"
    else
        print_warning "Python $PYTHON_VERSION 版本较低，建议升级到 3.13"
    fi

    # 检查必要的命令
    if ! command -v curl &> /dev/null; then
        print_error "curl 命令未找到，请安装"
        exit 1
    fi

    echo -e "系统依赖检查完成"
}

setup_directories() {
    print_step "2" "创建工作目录..."

    mkdir -p "$INPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo -e "输入目录: $INPUT_DIR"
    echo -e "输出目录: $OUTPUT_DIR"
}

download_checkpoint() {
    print_step "3" "下载SHARP模型检查点..."

    if [ -f "$CHECKPOINT_FILE" ]; then
        echo -e "检查点文件已存在: $CHECKPOINT_FILE"
        return
    fi

    echo -e "正在下载检查点文件..."
    curl -L -o "$CHECKPOINT_FILE" "$CHECKPOINT_URL"

    if [ ! -f "$CHECKPOINT_FILE" ]; then
        print_error "检查点下载失败"
        exit 1
    fi

    echo -e "检查点下载完成: $CHECKPOINT_FILE"
}

check_model_preloaded() {
    print_step "3.5" "检查模型预加载状态..."

    # 检查环境变量
    if [ "$SHARP_MODEL_PRELOADED" = "1" ]; then
        echo -e "检测到预加载的SHARP模型"
        echo -e "   设备: ${SHARP_MODEL_DEVICE:-未知}"
        echo -e "   跳过模型下载和安装步骤"
        MODEL_PRELOADED=1
    else
        echo -e "未检测到预加载模型，将正常加载"
        MODEL_PRELOADED=0
    fi
}

check_sharp_installation() {
    print_step "4" "检查SHARP安装和格式支持..."

    cd "$SHARP_DIR"
    if ! python3 -c "import sharp" &> /dev/null; then
        print_error "SHARP 未正确安装，请运行安装脚本"
        exit 1
    fi

    echo -e "SHARP 安装正常"

    # 检查基本功能
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import sharp
    print('  • SHARP模块导入成功')
    print('  • 支持标准图像格式: JPG, PNG, TIFF, BMP')
    print('  • 可扩展支持: RAW, HEIC, WebP (需额外安装)')
except ImportError as e:
    print(f'  • 导入失败: {e}')
"
}

check_voxelizer_dependencies() {
    print_step "4.5" "检查体素化工具依赖..."

    # 检查Open3D
    if ! python3 -c "import open3d" &> /dev/null; then
        print_error "Open3D 未安装，请运行: pip install open3d"
        exit 1
    fi

    # 检查体素化脚本
    if [ ! -f "$VOXELIZER_SCRIPT" ]; then
        print_error "体素化脚本不存在: $VOXELIZER_SCRIPT"
        exit 1
    fi

    echo -e "体素化工具依赖正常"
}

process_images() {
    print_step "5" "处理输入图像..."

    # 检查输入目录是否有图像
    # 首先检查是否有任何文件（排除隐藏文件）
    local total_files
    total_files=$(find "$INPUT_DIR" -type f -not -name ".*" | wc -l)

    if [ "$total_files" -eq 0 ]; then
        print_warning "输入目录 $INPUT_DIR 中没有找到图像文件"
        echo -e "请将照片放入 $INPUT_DIR 目录，然后重新运行脚本"
        echo -e "SHARP支持95+种格式，包括:"
        echo -e "  Apple格式: HEIC, HEIF, MOV (Live Photos)"
        echo -e "  RAW格式: CR2, CR3, NEF, ARW, DNG, RAF, RW2, PEF, SRW, ORF 等"
        echo -e "  现代格式: WebP (+ AVIF需要额外安装)"
        echo -e "  传统格式: JPG, JPEG, PNG, BMP, TIFF, GIF 等"
        echo -e ""
        echo -e "直接使用iPhone照片或专业相机RAW文件，无需转换！"
        exit 1
    fi

    # 检查是否有支持的图像格式

    # 构建find命令来查找所有支持的格式
    local find_patterns=""
    # 常见格式优先
    for ext in "jpg" "jpeg" "png" "heic" "heif" "mov" "cr2" "nef" "arw" "dng" "raf" "webp"; do
        find_patterns="$find_patterns -o -iname \"*.$ext\""
    done

    local image_count
    image_count=$(eval "find \"$INPUT_DIR\" -type f \( ${find_patterns# -o} \) 2>/dev/null | wc -l")

    if [ "$image_count" -eq 0 ]; then
        print_warning "找到 $total_files 个文件，但没有支持的图像格式"
        echo -e "SHARP支持的格式包括: JPG, PNG, HEIC, HEIF, MOV, CR2, NEF, ARW, DNG, RAF, WebP 等"
        echo -e "请检查文件格式或转换为支持的格式"
        exit 1
    fi

    echo -e "找到 $image_count 张图像文件"

    # 提示用户关于iPhone照片的去畸变
    echo -e "${YELLOW}提示:${NC} 如果使用iPhone照片，建议先进行去畸变处理"
    echo -e "      超广角镜头可能产生明显的拉伸效果，影响设备导航精度"
}

run_sharp_prediction() {
    print_step "6" "运行SHARP模型预测..."

    cd "$SHARP_DIR"

    echo -e "开始生成3D高斯模型..."
    echo -e "输入目录: $INPUT_DIR"
    echo -e "输出目录: $OUTPUT_DIR"
    echo -e "检查点: $CHECKPOINT_FILE"
    echo -e ""

    # 运行SHARP预测
    sharp predict \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        -c "$CHECKPOINT_FILE"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}3D高斯模型生成完成！${NC}"
    else
        print_error "模型生成失败"
        exit 1
    fi
}

run_voxelization() {
    print_step "7" "运行点云体素化处理..."

    echo -e "开始体素化处理..."
    echo -e "输入目录: $OUTPUT_DIR"
    echo -e "体素化脚本: $VOXELIZER_SCRIPT"
    echo -e ""

    # 直接查找OUTPUT_DIR中的PLY文件（排除子目录中的文件）
    local ply_files=""
    for file in "$OUTPUT_DIR"/*.ply; do
        if [ -f "$file" ]; then
            ply_files="$ply_files$file\n"
        fi
    done

    if [ -z "$ply_files" ]; then
        print_warning "在 $OUTPUT_DIR 中未找到PLY文件，跳过体素化"
        return 0
    fi

    echo -e "发现的PLY文件:"
    echo -e "$ply_files" | while read -r file; do
        if [ -n "$file" ]; then
            echo -e "  • $(basename "$file")"
        fi
    done
    echo -e ""

    # 对每个PLY文件运行体素化
    echo -e "$ply_files" | while read -r ply_file; do
        if [ -n "$ply_file" ] && [ -f "$ply_file" ]; then
            echo -e "正在处理: $(basename "$ply_file")"

            # 使用局部路径规划模式进行体素化
            python3 "$VOXELIZER_SCRIPT" "$ply_file" --local-planning

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}$(basename "$ply_file") 体素化完成${NC}"
            else
                print_error "$(basename "$ply_file") 体素化失败"
                return 1
            fi
        fi
    done

    echo -e "${GREEN}所有文件体素化完成！${NC}"
}

show_results() {
    print_step "8" "显示完整结果..."

    echo -e "${GREEN}完整处理结果:${NC}"
    echo -e "输出目录: $OUTPUT_DIR"
    echo -e ""

    # 显示SHARP输出
    local gaussians_files
    gaussians_files=$(find "$OUTPUT_DIR" -name "*.ply" -not -path "*/cropped_pointclouds/*" -not -path "*/voxelized_outputs/*" 2>/dev/null)
    if [ -n "$gaussians_files" ]; then
        echo -e "${BLUE}SHARP 3D高斯模型:${NC}"
        echo "$gaussians_files" | while read -r file; do
            local size
            size=$(ls -lh "$file" | awk '{print $5}')
            echo -e "  $(basename "$file") (${size})"
        done
        echo -e ""
    fi

    # 显示裁剪后的点云
    local cropped_files
    cropped_files=$(find "$OUTPUT_DIR/cropped_pointclouds" -name "*.ply" 2>/dev/null)
    if [ -n "$cropped_files" ]; then
        echo -e "${BLUE}局部区域裁剪结果:${NC}"
        echo "$cropped_files" | while read -r file; do
            local size
            size=$(ls -lh "$file" | awk '{print $5}')
            echo -e "  $(basename "$file") (${size})"
        done
        echo -e ""
    fi

    # 显示体素化结果
    local voxelized_files
    voxelized_files=$(find "$OUTPUT_DIR/voxelized_outputs" -name "*.ply" 2>/dev/null)
    if [ -n "$voxelized_files" ]; then
        echo -e "${BLUE}体素化结果 (路径规划就绪):${NC}"
        echo "$voxelized_files" | while read -r file; do
            local size
            size=$(ls -lh "$file" | awk '{print $5}')
            echo -e "  $(basename "$file") (${size})"
        done
        echo -e ""
    fi

    echo -e "${BLUE}文件说明:${NC}"
    echo -e "• 原始PLY文件: SHARP生成的3D高斯模型"
    echo -e "• cropped_*.ply: 局部区域裁剪后的点云"
    echo -e "• voxelized_*.ply: 1cm体素网格，适用于路径规划"
    echo -e ""
    echo -e "${BLUE}使用建议:${NC}"
    echo -e "• PLY文件可以使用Polycam、SuperSplat等3DGS渲染器查看"
    echo -e "• 坐标系遵循OpenCV标准 (x右, y下, z前)"
    echo -e "• voxelized文件适用于智能设备路径规划和空间感知"
}

cleanup_temp_files() {
    print_step "9" "清理临时文件..."

    # 可以在这里添加清理逻辑
    # 只删除缓存和临时文件，保留正式结果（*.ply、cropped_pointclouds、voxelized_outputs、logs）
    if [ -d "$OUTPUT_DIR" ]; then
        for entry in "$OUTPUT_DIR"/*; do
            # skip if no matches
            [ -e "$entry" ] || continue

            base="$(basename "$entry")"

            # Preserve logs directory, voxelized outputs, cropped pointclouds, and any .ply files at top level
            if [ "$base" = "logs" ] || [ "$base" = "voxelized_outputs" ] || [ "$base" = "cropped_pointclouds" ]; then
                continue
            fi

            if [ -f "$entry" ]; then
                case "$entry" in
                    *.ply) continue ;;
                esac
            fi

            # Otherwise remove (these are considered cache / intermediates)
            rm -rf "$entry" || true
        done
    fi

    # 清理 pip 缓存（如有权限）
    if command -v pip >/dev/null 2>&1; then
        pip cache purge || true
    fi

    # 如果存在 conda，尝试清理 conda 缓存以释放空间
    if command -v conda >/dev/null 2>&1; then
        conda clean --all -y || true
    fi

    # 清理 /tmp 下本脚本可能产生的临时文件（保守删除）
    if [ -d "/tmp" ]; then
        # 仅删除本脚本可能创建的临时文件（以 run_sharp 或 aylm 前缀为准）
        find /tmp -maxdepth 1 -type f \( -name "run_sharp*" -o -name "aylm*" \) -exec rm -f {} \; || true
    fi

    echo -e "清理完成（保留日志: $LOG_FILE）"
}

main() {
    print_header

    check_dependencies
    setup_directories
    download_checkpoint
    check_model_preloaded
    if [ "$MODEL_PRELOADED" = "0" ]; then
        check_sharp_installation
    fi
    check_voxelizer_dependencies
    process_images
    run_sharp_prediction
    run_voxelization
    show_results
    cleanup_temp_files

    echo -e ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}  SHARP + 体素化完整处理任务完成！${NC}"
    echo -e "${GREEN}================================================${NC}"
}

# 参数处理
case "${1:-}" in
    "--help"|"-h")
        echo "SHARP 一键启动脚本 (支持体素化)"
        echo ""
        echo "用法: $0 [选项]"
        echo ""
        echo "完整流程选项:"
        echo "  (无参数)     运行完整流程：环境检查 + SHARP预测 + 体素化"
        echo ""
        echo "分步骤选项:"
        echo "  --help, -h    显示此帮助信息"
        echo "  --setup       仅执行环境设置，不运行预测"
        echo "  --predict     仅运行SHARP预测（假设环境已设置）"
        echo "  --voxelize    仅运行体素化（假设已有SHARP输出）"
        echo ""
        echo "环境变量:"
        echo "  INPUT_DIR     输入图像目录 (默认: ./inputs/input_images)"
        echo "  OUTPUT_DIR    输出目录 (默认: ./outputs/output_gaussians)"
        echo "  SHARP_MODEL_PATH    SHARP模型文件路径 (默认: ./models/sharp_2572gikvuh.pt)"
        echo "  VOXELIZER_INTERMEDIATE_DIR    体素化中间文件目录 (默认: ./outputs/output_gaussians/cropped_pointclouds)"
        echo "  SHARP_MODEL_PRELOADED    模型预加载标志 (0=未预加载, 1=已预加载)"
        echo ""
        echo "输出文件说明:"
        echo "  *.ply         SHARP生成的3D高斯模型"
        echo "  cropped_*.ply 局部区域裁剪结果"
        echo "  voxelized_*.ply 1cm体素网格（路径规划就绪）"
        echo ""
        exit 0
        ;;
    "--setup")
        print_header
        check_dependencies
        setup_directories
        download_checkpoint
        check_model_preloaded
        if [ "$MODEL_PRELOADED" = "0" ]; then
            check_sharp_installation
        fi
        echo -e "${GREEN}环境设置完成！${NC}"
        exit 0
        ;;
    "--predict")
        print_header
        check_model_preloaded
        if [ "$MODEL_PRELOADED" = "0" ]; then
            check_sharp_installation
        fi
        process_images
        run_sharp_prediction
        show_results
        echo -e "${GREEN}预测完成！${NC}"
        exit 0
        ;;
    "--voxelize")
        print_header
        check_voxelizer_dependencies
        run_voxelization
        show_results
        echo -e "${GREEN}体素化完成！${NC}"
        exit 0
        ;;
    *)
        main
        ;;
esac
