#!/bin/bash
# AYLM v2 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

# 文件扩展名模式
IMAGE_EXTS='-iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.heic" -o -iname "*.webp" -o -iname "*.tiff" -o -iname "*.bmp"'
VIDEO_EXTS='-iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.webm" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.m4v"'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

show_help() {
    cat << EOF
${CYAN}AYLM v2 - 3D Gaussian Splatting 点云处理工具${NC}

用法: ./run.sh [选项] [参数]

${YELLOW}处理选项:${NC}
  -h, --help        显示帮助信息
  --setup           设置环境并下载模型
  --voxelize        体素化处理
  --predict         预测流程
  --full            完整流程（顺序）
  --pipeline        流水线模式（并行）
  --auto            自动选择模式（默认）
  --video           处理视频
  --video-extract   提取视频帧
  --video-play      播放体素序列

${YELLOW}参数:${NC}
  -i, --input       输入路径 (默认: inputs/input_images)
  -o, --output      输出目录
  -c, --config      视频配置文件
  --voxel-size      体素尺寸/米 (默认: 0.005)
  --keep-ground     保留地面点
  --transform       转换坐标系
  --frame-interval  帧间隔/秒 (默认: 1.0)
  --use-gpu         GPU加速
  --fps             播放帧率 (默认: 10)
  --loop            循环播放
  -v, --verbose     详细输出
  -q, --quiet       安静模式
  --check-only      仅检查环境

${YELLOW}语义检测选项:${NC}
  --semantic        启用语义检测
  --no-semantic     禁用语义检测
  --semantic-model  YOLO模型名称 (默认: yolo11n-seg.pt)
  --semantic-confidence  检测置信度 (默认: 0.5)

${YELLOW}点云切片选项:${NC}
  --slice           启用点云切片 (默认)
  --no-slice        禁用点云切片
  --slice-radius    切片半径/米 (默认: 10.0)

${YELLOW}示例:${NC}
  ./run.sh --setup                    # 初始化
  ./run.sh -i ./images                # 处理图像
  ./run.sh --video -i video.mp4       # 处理视频
  ./run.sh --semantic -i ./images     # 启用语义检测
  ./run.sh --slice-radius 5.0         # 设置切片半径
EOF
}

find_venv() {
    for name in aylm_env venv .venv; do
        [[ -d "$SCRIPT_DIR/$name" ]] && echo "$SCRIPT_DIR/$name" && return 0
    done
    return 1
}

activate_venv() {
    local venv_path="$1"
    [[ -f "$venv_path/bin/activate" ]] || { log_error "无法激活虚拟环境: $venv_path"; exit 1; }
    source "$venv_path/bin/activate"
    log_info "已激活虚拟环境: $venv_path"
}

check_and_install_deps() {
    log_info "检查依赖..."
    local missing=()
    for dep in numpy scipy plyfile torch torchvision Pillow opencv-python matplotlib; do
        local import_name="$dep"
        [[ "$dep" == "Pillow" ]] && import_name="PIL"
        [[ "$dep" == "opencv-python" ]] && import_name="cv2"
        python3 -c "import $import_name" 2>/dev/null || missing+=("$dep")
    done

    [[ ${#missing[@]} -gt 0 ]] && { log_info "安装: ${missing[*]}"; pip install --quiet "${missing[@]}"; }
    python3 -c "import aylm" 2>/dev/null || pip install --quiet -e .
    [[ -d "$SCRIPT_DIR/ml-sharp" ]] && python3 -c "import sharp" 2>/dev/null || pip install --quiet -e "$SCRIPT_DIR/ml-sharp/"
}

setup_env() {
    local venv_path
    if venv_path=$(find_venv); then
        log_info "找到虚拟环境: $venv_path"
    else
        venv_path="$SCRIPT_DIR/aylm_env"
        log_info "创建虚拟环境: $venv_path"
        python3 -m venv "$venv_path"
    fi
    activate_venv "$venv_path"
    check_and_install_deps
}

count_files() {
    local path="$1" exts="$2"
    [[ -f "$path" ]] && echo 1 && return
    [[ -d "$path" ]] && eval "find \"$path\" -maxdepth 1 -type f \( $exts \) 2>/dev/null | wc -l | tr -d ' '" || echo 0
}

show_banner() {
    local title="$1"
    echo -e "\n${CYAN}============================================================${NC}"
    echo -e "${CYAN}  A.YLM v2 - $title${NC}"
    echo -e "${CYAN}============================================================${NC}\n"
}

run_voxelize() { log_step "体素化处理..."; python3 -m aylm.cli voxelize "$@"; }
run_predict() { log_step "预测流程..."; python3 -m aylm.cli predict "$@"; }
run_full() { log_step "完整流程（顺序）..."; python3 -m aylm.cli process "$@"; }
run_pipeline() { log_step "流水线处理（并行）..."; python3 -m aylm.cli pipeline "$@"; }
run_video_process() { log_step "视频处理..."; python3 -m aylm.cli video process "$@"; }
run_video_extract() { log_step "提取视频帧..."; python3 -m aylm.cli video extract "$@"; }
run_video_play() { log_step "播放体素序列..."; python3 -m aylm.cli video play "$@"; }

run_auto() {
    local input_dir="$1"; shift
    local image_count=$(count_files "$input_dir" "$IMAGE_EXTS")
    local video_count=$(count_files "inputs/videos" "$VIDEO_EXTS")

    show_banner "智能处理模式"
    echo -e "  图像: ${YELLOW}$image_count${NC} | 视频: ${YELLOW}$video_count${NC}\n"

    if [[ "$image_count" -gt 0 ]]; then
        # 统一使用 pipeline 命令（支持语义检测和切片）
        run_pipeline "$@"
    elif [[ "$video_count" -gt 0 ]]; then
        local first_video=$(eval "find inputs/videos -maxdepth 1 -type f \( $VIDEO_EXTS \) 2>/dev/null | head -1")
        [[ -n "$first_video" ]] && run_video_process -i "$first_video" -o "outputs/video_output" "$@"
    else
        log_error "未找到图像或视频文件"
        echo "  请将文件放入: $input_dir 或 inputs/videos/"
        exit 1
    fi
}

main() {
    local action="auto" input_dir="inputs/input_images" output_dir="" config_file=""
    local extra_args=() check_only=false use_gpu=false frame_interval="" fps="10" loop=false
    # 语义检测和切片参数（默认都启用）
    local semantic=true semantic_model="yolo11n-seg.pt" semantic_confidence="0.5"
    local slice=true slice_radius="10.0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) show_help; exit 0 ;;
            --setup|--voxelize|--predict|--full|--pipeline|--auto|--video|--video-extract|--video-play)
                action="${1#--}"; shift ;;
            -i|--input) input_dir="$2"; extra_args+=("-i" "$2"); shift 2 ;;
            -o|--output) output_dir="$2"; extra_args+=("-o" "$2"); shift 2 ;;
            -c|--config) config_file="$2"; shift 2 ;;
            --voxel-size|--frame-interval) extra_args+=("$1" "$2"); [[ "$1" == "--frame-interval" ]] && frame_interval="$2"; shift 2 ;;
            --keep-ground|--transform|-v|--verbose|-q|--quiet) extra_args+=("$1"); shift ;;
            --use-gpu) use_gpu=true; shift ;;
            --fps) fps="$2"; shift 2 ;;
            --loop) loop=true; shift ;;
            --check-only) check_only=true; shift ;;
            # 语义检测参数
            --semantic) semantic=true; shift ;;
            --no-semantic) semantic=false; shift ;;
            --semantic-model) semantic_model="$2"; shift 2 ;;
            --semantic-confidence) semantic_confidence="$2"; shift 2 ;;
            # 切片参数
            --slice) slice=true; shift ;;
            --no-slice) slice=false; shift ;;
            --slice-radius) slice_radius="$2"; shift 2 ;;
            *) extra_args+=("$1"); shift ;;
        esac
    done

    setup_env

    if [[ "$check_only" == true ]]; then
        show_banner "环境检查"
        echo "  Python: $(python3 --version) | 图像: $(count_files "$input_dir" "$IMAGE_EXTS") | 视频: $(count_files "inputs/videos" "$VIDEO_EXTS")"
        [[ -f "$SCRIPT_DIR/models/sharp_2572gikvuh.pt" ]] && echo -e "  模型: ${GREEN}已下载${NC}" || echo -e "  模型: ${YELLOW}未下载${NC}"
        exit 0
    fi

    # 构建语义检测和切片参数
    local semantic_args=()
    if [[ "$semantic" == true ]]; then
        semantic_args+=("--semantic" "--semantic-model" "$semantic_model" "--semantic-confidence" "$semantic_confidence")
    else
        semantic_args+=("--no-semantic")
    fi

    local slice_args=()
    if [[ "$slice" == true ]]; then
        slice_args+=("--slice" "--slice-radius" "$slice_radius")
    else
        slice_args+=("--no-slice")
    fi

    case "$action" in
        setup) log_step "下载模型..."; python3 -m aylm.cli setup --download ;;
        voxelize) run_voxelize "${extra_args[@]}" ;;
        predict) run_predict "${extra_args[@]}" ;;
        full) run_full "${extra_args[@]}" ;;
        pipeline) run_pipeline "${extra_args[@]}" "${semantic_args[@]}" "${slice_args[@]}" ;;
        video)
            local video_args=("${extra_args[@]}")
            [[ -n "$config_file" ]] && video_args+=("-c" "$config_file")
            [[ "$use_gpu" == true ]] && video_args+=("--use-gpu")
            show_banner "视频处理"
            run_video_process "${video_args[@]}" ;;
        video-extract)
            local extract_args=("${extra_args[@]}")
            [[ -n "$config_file" ]] && extract_args+=("-c" "$config_file")
            [[ ! " ${extra_args[*]} " =~ " -o " ]] && extract_args+=("-o" "outputs/extracted_frames")
            run_video_extract "${extract_args[@]}" ;;
        video-play)
            local play_args=("-i" "$input_dir" "--fps" "$fps")
            [[ "$loop" == true ]] && play_args+=("--loop")
            run_video_play "${play_args[@]}" ;;
        auto) run_auto "$input_dir" "${extra_args[@]}" "${semantic_args[@]}" "${slice_args[@]}" ;;
    esac

    log_info "完成"
}

main "$@"
