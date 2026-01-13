# A.YLM

基于Apple SHARP模型的单图像3D重建和智能导航系统

Single-image 3D reconstruction and intelligent navigation system based on Apple SHARP model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 特性

- 基于Vision Transformer的单图像3D重建
- 智能体素化，1cm精度适用于导航
- 支持95+种图像格式，包括专业RAW格式
- GPU加速，实时处理（推理<1秒）
- 地面检测、坐标转换和路径规划
- 完整的CLI工具和Python API
- 模块化设计，易于扩展

## 系统要求

- Python 3.9+
- PyTorch 2.8.0+
- Open3D 0.18.0+
- CUDA-compatible GPU (推荐)
- 4GB+ RAM

## 安装

```bash
# 克隆仓库
git clone https://github.com/appergb/A.YLM.git
cd A.YLM

# 安装依赖
pip install -r requirements.txt
pip install -e ml-sharp/

# 或者安装开发版本（包含测试依赖）
pip install -e .[dev]
```

## 快速开始

### 使用CLI工具

```bash
# 运行完整流程：环境检查 + 3D重建 + 体素化
aylm process

# 或者分步骤执行
aylm setup      # 环境设置和检查
aylm predict    # 3D重建
aylm voxelize   # 体素化处理

# 查看所有选项
aylm --help
aylm process --help
```

### 使用传统脚本

```bash
# 运行完整流程
./run_sharp.sh

# 分步骤执行
./run_sharp.sh --setup      # 环境检查
./run_sharp.sh --predict    # 3D重建
./run_sharp.sh --voxelize   # 体素化
```

## Python API

```python
from aylm import core, config, utils

# 获取配置
settings = config.get_settings()

# 运行3D重建
# (具体API将在后续版本中完善)

# 工具函数
from aylm.tools.coordinate_utils import transform_for_navigation
from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer
```

## 项目结构

```
A.YLM/
├── src/aylm/                    # 主包
│   ├── cli.py                   # 命令行接口
│   ├── core/                    # 核心功能
│   │   ├── exceptions.py        # 自定义异常
│   │   └── __init__.py
│   ├── config/                  # 配置管理
│   │   ├── settings.py          # 设置类
│   │   └── __init__.py
│   ├── tools/                   # 工具脚本
│   │   ├── coordinate_utils.py  # 坐标转换
│   │   ├── pointcloud_voxelizer.py  # 体素化
│   │   └── undistort_iphone.py  # 图像去畸变
│   ├── utils/                   # 工具函数
│   │   ├── colors.py            # 颜色输出
│   │   ├── file_utils.py        # 文件操作
│   │   └── logging.py           # 日志工具
│   └── __init__.py
├── ml-sharp/                    # SHARP模型 (子模块)
├── models/                      # 模型文件
├── inputs/                      # 输入数据
├── outputs/                     # 输出结果
├── tests/                       # 测试套件
└── scripts/                     # 兼容性脚本
```

## 输出文件

- `*.ply`: SHARP生成的3D高斯模型
- `cropped_*.ply`: 局部区域裁剪结果
- `voxelized_*.ply`: 1cm体素网格（路径规划就绪）

## 配置

### 环境变量

```bash
# 目录配置
export INPUT_DIR="/path/to/input/images"
export OUTPUT_DIR="/path/to/output/directory"
export SHARP_MODEL_PATH="/path/to/sharp/model.pt"

# 模型预加载
export SHARP_MODEL_PRELOADED=1
export SHARP_MODEL_DEVICE="cuda:0"
```
export OUTPUT_DIR="/path/to/output"
export SHARP_MODEL_PATH="/path/to/model.pt"
```

### 自定义参数

```bash
# 运行完整流程（自定义输入输出目录）
aylm process --input-dir /path/to/images --output-dir /path/to/output

# 仅运行体素化（自定义参数）
aylm voxelize --output-dir /path/to/output
python3 -m aylm.tools.pointcloud_voxelizer input.ply --voxel-size 0.01
```

## 开发

### 运行测试

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_config.py -v

# 带覆盖率测试
pytest --cov=aylm --cov-report=html
```

### 代码质量

```bash
# 格式化代码
black src/aylm tests
isort src/aylm tests

# 类型检查
mypy src/aylm

# 代码质量检查
ruff check src/aylm tests
```

### 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 故障排除

### 常见问题

1. **Python版本错误**
   ```bash
   python3 --version  # 确保 >= 3.9
   ```

2. **依赖安装失败**
   ```bash
   # CPU版本PyTorch
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **内存不足**
   - 关闭其他应用程序
   - 使用更大的体素尺寸降低精度

4. **CUDA相关问题**
   ```bash
   # 检查CUDA可用性
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### 获取帮助

- 📖 [文档](https://github.com/appergb/A.YLM/wiki)
- 🐛 [问题跟踪](https://github.com/appergb/A.YLM/issues)
- 💬 [讨论区](https://github.com/appergb/A.YLM/discussions)

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE_MIT](LICENSE_MIT) 文件了解详情。

## 致谢

- **Apple Inc.** - SHARP模型和研究
- **Open3D社区** - 点云处理库
- **PyTorch团队** - 深度学习框架
- **贡献者** - closer, true

---

**作者**: TRIP(appergb)  
**项目状态**: 生产就绪

**Developer**: TRIP (appergb)
**Contributors**: closer, true

## License

This project is licensed under the MIT License.

---

## CI / 持续集成

项目使用 GitHub Actions 进行持续集成，包含以下检查：

### CI Jobs

- **Code Quality**: 代码格式、linting 和类型检查
- **Unit Tests**: 单元测试和代码覆盖率
- **Integration Tests**: 集成测试（需手动触发或添加 `integration-test` 标签）
- **Cross-platform**: macOS 和 Windows 兼容性测试

### 本地开发

#### 代码质量检查

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行代码质量检查
black --check src/aylm
isort --check-only src/aylm
ruff check src/aylm
mypy src/aylm
```

#### 运行测试

```bash
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 带覆盖率
pytest --cov=aylm --cov-report=html
```

#### 使用预提交钩子

```bash
# 安装预提交钩子
pre-commit install

# 手动运行
pre-commit run --all-files
```

### CI 环境说明

CI 配置经过优化以处理复杂的依赖关系：

- **PyTorch**: 使用 CPU 版本避免 CUDA 复杂性
- **Open3D**: 支持 pip 和 conda 安装的 fallback 机制
- **缓存**: 使用 GitHub Actions 缓存加速依赖安装
- **分层测试**: 轻量级代码质量检查 + 完整单元测试 + 可选集成测试

### 故障排除

如果 CI 失败：

1. **依赖安装失败**: 检查 `scripts/install_deps.sh` 脚本
2. **测试失败**: 本地运行 `pytest` 复现问题
3. **代码质量失败**: 运行 `pre-commit run --all-files` 检查

### 手动触发集成测试

在 Pull Request 上添加 `integration-test` 标签或使用 workflow_dispatch 手动触发。
