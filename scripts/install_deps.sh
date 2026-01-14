#!/bin/bash
# A.YLM dependency installation script for CI/CD environments
# Handles complex dependencies like Open3D, PyTorch, and CUDA

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[INSTALL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
PYTHON_VERSION=${PYTHON_VERSION:-$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')}

print_step "Detected OS: $OS, Python: $PYTHON_VERSION"

# Install system dependencies
install_system_deps() {
    print_step "Installing system dependencies..."

    if [[ "$OS" == "linux" ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            libgl1 \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libgthread-2.0-0 \
            build-essential \
            cmake \
            ninja-build
    elif [[ "$OS" == "macos" ]]; then
        # macOS system deps are usually available
        print_step "macOS system dependencies should be available"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_step "Installing Python dependencies..."

    python3 -m pip install --upgrade pip

    # Install PyTorch (CPU version for CI)
    print_step "Installing PyTorch (CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps

    # Install Open3D (try multiple methods)
    print_step "Installing Open3D..."
    if ! pip install open3d>=0.18.0; then
        print_warning "pip install failed, trying conda..."
        if command -v conda &> /dev/null; then
            conda install -c conda-forge open3d=0.18.0 -y
        else
            print_error "Open3D installation failed. Please install manually."
            exit 1
        fi
    fi

    # Install project dependencies
    print_step "Installing A.YLM..."
    pip install -e .[dev] --no-deps

    print_step "Installing SHARP..."
    pip install -e ml-sharp/

    # Verify installations
    print_step "Verifying installations..."
    python3 -c "
import torch
import open3d as o3d
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'Open3D: {o3d.__version__}')
print(f'NumPy: {np.__version__}')
print('Core dependencies verified!')
"
}

# Install lightweight dependencies for code quality checks
install_lightweight_deps() {
    print_step "Installing lightweight dependencies for code quality..."

    python3 -m pip install --upgrade pip

    # Install only what's needed for linting and code quality
    pip install -e .[dev] --no-deps
    pip install ruff black isort mypy pre-commit
    pip install numpy scipy pillow plyfile matplotlib click
}

# Main installation
main() {
    case "${1:-full}" in
        "lightweight")
            install_lightweight_deps
            ;;
        "full")
            install_system_deps
            install_python_deps
            ;;
        *)
            echo "Usage: $0 [lightweight|full]"
            exit 1
            ;;
    esac

    print_success "Dependencies installed successfully!"
}

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
