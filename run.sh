#!/bin/bash
# AYLM v2 启动脚本
# 自动管理虚拟环境和依赖
# 智能选择单图模式或流水线模式

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

show_help() {
    cat << EOF
${CYAN}AYLM v2 - 3D Gaussian Splatting 点云处理工具${NC}

用法: ./run.sh [选项] [参数]

${YELLOW}选项:${NC}
  --help, -h        显示帮助信息
  --setup           仅设置环境（创建虚拟环境并安装依赖）
  --voxelize        运行体素化处理
  --predict         运行预测流程
  --full            运行完整处理流程（顺序执行）
  --pipeline        强制使用流水线模式（并行执行）
  --auto            自动选择模式（默认，>=2张图使用流水线）

${YELLOW}参数:${NC}
  -i, --input       输入目录或文件 (默认: inputs/input_images)
  -o, --output      输出目录 (默认: outputs/output_gaussians)
  --voxel-size      体素尺寸，单位米 (默认: 0.005)
  --keep-ground     保留地面点
  --transform       转换到机器人坐标系
  -v, --verbose     详细输出
  -q, --quiet       安静模式
  --check-only      仅检查环境，不运行处理

${YELLOW}示例:${NC}
  ./run.sh --setup                          # 初始化环境
  ./run.sh                                  # 自动模式（推荐）
  ./run.sh -i ./my_images                   # 指定输入目录
  ./run.sh --pipeline -i ./my_images        # 强制流水线模式
  ./run.sh --voxelize output.ply            # 体素化单个文件
  ./run.sh --check-only                     # 仅检查环境

${YELLOW}流水线模式说明:${NC}
  当输入目录包含 2 张或更多图像时，自动启用流水线模式：
  - 模型只加载一次到内存
  - 第 N 张图片推理时，第 N-1 张图片同时进行体素化
  - 显著提升多图处理效率

EOF
}

find_venv() {
    for venv_name in "aylm_env" "venv" ".venv"; do
        if [[ -d "$SCRIPT_DIR/$venv_name" ]]; then
            echo "$SCRIPT_DIR/$venv_name"
            return 0
        fi
    done
    return 1
}

create_venv() {
    local venv_path="$SCRIPT_DIR/aylm_env"
    log_info "创建虚拟环境: $venv_path"
    python3 -m venv "$venv_path"
    echo "$venv_path"
}

activate_venv() {
    local venv_path="$1"
    if [[ -f "$venv_path/bin/activate" ]]; then
        source "$venv_path/bin/activate"
        log_info "已激活虚拟环境: $venv_path"
    else
        log_error "无法激活虚拟环境: $venv_path"
        exit 1
    fi
}

check_and_install_deps() {
    log_info "检查依赖..."

    local deps=("numpy" "scipy" "plyfile" "torch" "torchvision" "Pillow" "opencv-python" "matplotlib")
    local missing=()

    for dep in "${deps[@]}"; do
        local import_name="$dep"
        case "$dep" in
            "Pillow") import_name="PIL" ;;
            "opencv-python") import_name="cv2" ;;
        esac

        if ! python3 -c "import $import_name" 2>/dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_info "安装缺失依赖: ${missing[*]}"
        pip install --quiet "${missing[@]}"
    else
        log_info "所有依赖已就绪"
    fi

    # 安装项目本身
    if ! python3 -c "import aylm" 2>/dev/null; then
        log_info "安装 aylm 包..."
        pip install --quiet -e .
    fi

    # 安装 ml-sharp 子模块
    if ! python3 -c "import sharp" 2>/dev/null; then
        if [[ -d "$SCRIPT_DIR/ml-sharp" ]]; then
            log_info "安装 sharp 包..."
            pip install --quiet -e "$SCRIPT_DIR/ml-sharp/"
        fi
    fi
}

setup_env() {
    local venv_path

    if venv_path=$(find_venv); then
        log_info "找到虚拟环境: $venv_path"
    else
        venv_path=$(create_venv)
    fi

    activate_venv "$venv_path"
    check_and_install_deps
}

count_images() {
    local input_dir="$1"
    local count=0

    if [[ -d "$input_dir" ]]; then
        count=$(find "$input_dir" -maxdepth 1 -type f \( \
            -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \
            -o -iname "*.heic" -o -iname "*.webp" -o -iname "*.tiff" \
            -o -iname "*.bmp" \) 2>/dev/null | wc -l | tr -d ' ')
    elif [[ -f "$input_dir" ]]; then
        count=1
    fi

    echo "$count"
}

run_voxelize() {
    log_step "运行体素化处理..."
    python3 -m aylm.cli voxelize "$@"
}

run_predict() {
    log_step "运行预测流程..."
    python3 -m aylm.cli predict "$@"
}

run_full() {
    log_step "运行完整处理流程（顺序模式）..."
    python3 -m aylm.cli process "$@"
}

run_pipeline() {
    log_step "运行流水线处理（并行模式）..."
    python3 -m aylm.cli pipeline "$@"
}

run_auto() {
    local input_dir="$1"
    shift
    local extra_args=("$@")

    # 统计图像数量
    local image_count
    image_count=$(count_images "$input_dir")

    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  A.YLM v2 - 智能处理模式${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
    echo -e "  输入目录:   ${YELLOW}$input_dir${NC}"
    echo -e "  图像数量:   ${YELLOW}$image_count${NC}"

    if [[ "$image_count" -eq 0 ]]; then
        log_error "未找到图像文件"
        echo "  请将图像放入输入目录: $input_dir"
        echo "  支持格式: jpg, jpeg, png, heic, webp, tiff, bmp"
        exit 1
    elif [[ "$image_count" -eq 1 ]]; then
        echo -e "  处理模式:   ${GREEN}单图模式（顺序执行）${NC}"
        echo ""
        echo -e "${CYAN}============================================================${NC}"
        echo ""
        run_full "${extra_args[@]}"
    else
        echo -e "  处理模式:   ${GREEN}流水线模式（并行执行）${NC}"
        echo ""
        echo -e "  ${BLUE}流水线策略:${NC}"
        echo "    - 模型只加载一次到内存"
        echo "    - 第 N 张推理时，第 N-1 张同时体素化"
        echo "    - 预计并行阶段: $((image_count - 1)) 次"
        echo ""
        echo -e "${CYAN}============================================================${NC}"
        echo ""
        run_pipeline "${extra_args[@]}"
    fi
}

main() {
    local action="auto"
    local input_dir="inputs/input_images"
    local output_dir=""
    local extra_args=()
    local check_only=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                show_help
                exit 0
                ;;
            --setup)
                action="setup"
                shift
                ;;
            --voxelize)
                action="voxelize"
                shift
                ;;
            --predict)
                action="predict"
                shift
                ;;
            --full)
                action="full"
                shift
                ;;
            --pipeline)
                action="pipeline"
                shift
                ;;
            --auto)
                action="auto"
                shift
                ;;
            -i|--input)
                input_dir="$2"
                extra_args+=("-i" "$2")
                shift 2
                ;;
            -o|--output)
                output_dir="$2"
                extra_args+=("-o" "$2")
                shift 2
                ;;
            --voxel-size)
                extra_args+=("--voxel-size" "$2")
                shift 2
                ;;
            --keep-ground)
                extra_args+=("--keep-ground")
                shift
                ;;
            --transform)
                extra_args+=("--transform")
                shift
                ;;
            -v|--verbose)
                extra_args+=("-v")
                shift
                ;;
            -q|--quiet)
                extra_args+=("-q")
                shift
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            *)
                extra_args+=("$1")
                shift
                ;;
        esac
    done

    # 设置环境
    setup_env

    # 仅检查模式
    if [[ "$check_only" == true ]]; then
        echo ""
        log_info "环境检查完成"
        echo ""
        echo "  Python:     $(python3 --version)"
        echo "  虚拟环境:   $(which python3)"
        echo "  输入目录:   $input_dir"
        echo "  图像数量:   $(count_images "$input_dir")"
        echo ""

        # 检查模型
        local model_path="$SCRIPT_DIR/models/sharp_2572gikvuh.pt"
        if [[ -f "$model_path" ]]; then
            echo -e "  模型状态:   ${GREEN}已下载${NC}"
        else
            echo -e "  模型状态:   ${YELLOW}未下载${NC} (运行 ./run.sh --setup 下载)"
        fi
        echo ""
        exit 0
    fi

    # 执行操作
    case "$action" in
        setup)
            # 下载模型
            log_step "下载模型..."
            python3 -m aylm.cli setup --download
            log_info "环境设置完成"
            ;;
        voxelize)
            run_voxelize "${extra_args[@]}"
            ;;
        predict)
            run_predict "${extra_args[@]}"
            ;;
        full)
            run_full "${extra_args[@]}"
            ;;
        pipeline)
            run_pipeline "${extra_args[@]}"
            ;;
        auto)
            run_auto "$input_dir" "${extra_args[@]}"
            ;;
    esac

    echo ""
    log_info "完成"
}

main "$@"
