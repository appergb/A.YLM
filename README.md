# A.YLM

Single-image 3D reconstruction and intelligent navigation system based on Apple SHARP model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- Single-image 3D reconstruction using Vision Transformer
- Intelligent voxelization with 1cm precision for navigation
- Support for 95+ image formats including professional RAW
- GPU acceleration with real-time processing (<1s inference)
- Ground detection, coordinate transformation and path planning

## Requirements

- Python 3.9+
- PyTorch 2.8.0+
- Open3D 0.18.0+
- 4GB+ RAM (GPU recommended)

## Installation

```bash
git clone https://github.com/appergb/A.YLM.git
cd A.YLM

pip install -r requirements.txt
pip install -e ml-sharp/
```

## Usage

```bash
# Run complete pipeline
./run_sharp.sh

# Or run individual steps
./run_sharp.sh --setup      # Environment check
./run_sharp.sh --predict    # 3D reconstruction
./run_sharp.sh --voxelize   # Voxelization
```

## Model Preloading (Recommended)

```bash
# Background model preloading
python3 scripts/preload_sharp_model.py --background

# Check preload status
python3 scripts/preload_sharp_model.py --status
```

## Output Files

- `*.ply`: 3D Gaussian splatting model
- `cropped_*.ply`: Local region cropping results
- `voxelized_*.ply`: 1cm voxel grid (navigation ready)

## Configuration

### Environment Variables

```bash
export INPUT_DIR="/path/to/images"
export OUTPUT_DIR="/path/to/output"
export SHARP_MODEL_PATH="/path/to/model.pt"
```

### Custom Parameters

```bash
# Adjust voxel size and range
python3 scripts/pointcloud_voxelizer.py input.ply --voxel-size 0.01 --range 10.0

# Enable visualization
python3 scripts/pointcloud_voxelizer.py input.ply --visualize
```

## Troubleshooting

1. **Python version error**

   ```bash
   python3 --version  # Ensure >= 3.9
   ```

2. **Dependency installation failure**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory insufficient**
   - Close other applications
   - Use `--voxel-size 0.02` for lower precision

4. **GPU unavailable**

   ```bash
   python3 scripts/preload_sharp_model.py --device cpu
   ```

## Project Structure

```text
A.YLM/
├── scripts/                    # Python scripts
│   ├── preload_sharp_model.py  # Model preloading
│   ├── pointcloud_voxelizer.py # Voxelization
│   └── coordinate_utils.py     # Coordinate utilities
├── ml-sharp/                   # SHARP model code
├── src/aylm/                  # Main package
├── inputs/                     # Input images
├── outputs/                    # Output results
├── models/                     # Model weights
└── run_sharp.sh               # Main script
```

## Contributing

Issues and pull requests are welcome.

**Developer**: TRIP (appergb)
**Contributors**: closer, true

## License

This project is licensed under the MIT License.

---

## CI / Automation status — 注意事项

当前仓库的 CI 与自动化配置已做过优化以减少 CI 运行时的磁盘与网络占用（例如：将 Open3D 通过 conda 安装、在 code-quality job 中仅安装轻量级的静态检查工具等）。

重要说明（请仔细阅读）：

- 本版本**不**触发或不保证执行“完全自动化的端到端流程**（full end-to-end automated pipeline）**”——某些运行时/集成测试需要大型二进制依赖（如 Open3D、PyTorch 的 CUDA 变体或其它 GPU 包），这些在 code-quality 检查阶段被刻意排除以避免 CI 运行失败或磁盘耗尽。
- 项目致力于维护稳定的 CI 流程。任何 CI 失败都将被及时修复，以确保代码质量和自动化测试的可靠性。
- 如果你需要强制运行完整集成测试，请在 CI 中使用专门的 integration job、预构建包含所有依赖的 Docker 镜像，或在本地/专用机器上运行完整流程（详见下文的“如何运行完整流程”）。

如何在本地运行完整流程（建议）：

1. 在有足够磁盘与网络权限的机器上创建 conda 环境（推荐）：

```bash
conda create -n aylm python=3.11 -y
conda activate aylm
conda install -c conda-forge -c open3d-admin open3d=0.18.0 numpy scipy plyfile pillow opencv matplotlib -y
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --no-deps
pip install -r requirements.txt
pip install -e ml-sharp/
```

2. 运行完整流程：

```bash
./run_sharp.sh
```

### 若你确实需要 CI 在 GitHub 上运行完整集成（包含全部 heavy deps），建议：

- 使用自托管 runner（有足够磁盘与 GPU 支持），或
- 构建并使用包含所有依赖的预构建 Docker 镜像（推到 ghcr.io 或 Docker Hub），然后在 workflow 中使用 `container:` 指令运行测试。

如果你希望我为你将这些变更推送到远端并触发一次 Actions，我可以立即 push 并监控运行日志，或按你的要求先不推送。请回复“请 push”或“先不 push”。
