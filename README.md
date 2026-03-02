# A.YLM v2

A.YLM 是一个面向具身智能场景的 3D 几何处理与安全评估项目。

它的主流程是：
- 从图像或视频生成 3D 点云（依赖 Apple SHARP 模型）
- 对点云做体素化、语义融合、跟踪等处理
- 生成可用于安全评估的结构化结果
- 支持命令行和 HTTP/WebSocket 服务调用

## 1. 当前代码范围

本仓库主要包含以下能力：
- `run.sh` 一键脚本（环境检查、演示、自动处理）
- `aylm` CLI（`setup/predict/voxelize/process/pipeline/video/demo/serve/fuse`）
- 宪法安全评估模块（`src/aylm/constitution`）
- FastAPI 服务接口（`src/aylm/api`）
- 单元测试与 CI 工作流（`tests/`, `.github/workflows/ci.yml`）

## 2. 环境要求

### 必需项
- Python：建议 3.11 或 3.12
  - 说明：`run.sh` 已做版本检查，默认限制在 3.11/3.12
- Git 子模块：必须完整拉取 `ml-sharp`
- 首次安装需要联网访问 GitHub / PyPI / 模型下载地址

### 建议硬件
- 能运行 PyTorch 的 CPU 环境即可
- 有 CUDA 或 Apple MPS 可提升速度（非必需）

## 3. 安装

```bash
# 1) 拉取仓库（必须带子模块）
git clone --recursive https://github.com/appergb/A.YLM.git
cd A.YLM

# 2) 创建虚拟环境
python3.11 -m venv aylm_env
source aylm_env/bin/activate

# 3) 安装核心依赖
pip install -e .
pip install -e ml-sharp/

# 4) 可选依赖
pip install -e ".[full]"      # Open3D 等
pip install -e ".[semantic]"  # Ultralytics 等语义检测依赖
pip install -e ".[api]"       # FastAPI/uvicorn
pip install -e ".[dev]"       # pytest/ruff/mypy 等
```

## 4. 快速开始

### 4.1 一键脚本（推荐）

```bash
# 只检查环境，不跑任务
./run.sh --check-only

# 跑演示，不依赖输入图片/视频
./run.sh --demo

# 自动模式（会从 inputs/input_images 和 inputs/videos 查找输入）
./run.sh
```

如果 `./run.sh` 没有发现输入文件，会给出明确提示并退出，不会误报崩溃。

### 4.2 指定输入

```bash
# 图像目录
./run.sh -i /path/to/images

# 强制流水线模式
./run.sh --pipeline -i /path/to/images

# 视频处理
./run.sh --video -i /path/to/video.mp4
```

## 5. CLI 使用

安装后可直接使用 `aylm` 命令：

```bash
# 环境检查/模型下载
aylm setup --download

# 图像推理
aylm predict -i inputs/input_images -o outputs/output_gaussians

# 体素化
aylm voxelize -i outputs/output_gaussians

# 完整流程
aylm process -i inputs/input_images

# 并行流水线
aylm pipeline -i inputs/input_images

# 视频
aylm video process -i inputs/videos/example.mp4
aylm video extract -i inputs/videos/example.mp4 -o outputs/extracted_frames
aylm video play -i outputs/video_output/voxelized

# 宪法评估演示
aylm demo

# API 服务
aylm serve --port 8000

# 多帧融合
aylm fuse -i outputs/video_output/voxelized
```

查看完整参数：

```bash
aylm --help
aylm pipeline --help
aylm video --help
./run.sh --help
```

## 6. 输入与输出约定

### 输入目录
- `inputs/input_images/`：图片输入
- `inputs/videos/`：视频输入

### 输出目录
- `outputs/output_gaussians/`：SHARP 预测输出
- `outputs/video_output/`：视频流程输出
- `outputs/extracted_frames/`：视频抽帧输出

## 7. API 服务

安装 API 依赖后启动：

```bash
pip install -e ".[api]"
aylm serve --host 0.0.0.0 --port 8000
```

常用接口：
- `GET /api/v1/health`
- `POST /api/v1/evaluate`
- `POST /api/v1/evaluate/batch`
- `PUT /api/v1/ego`
- `GET /api/v1/summary`
- `WS /api/v1/session`

## 8. 项目结构

```text
A.YLM/
├── .github/workflows/        # CI
├── configs/                  # 配置样例
├── inputs/                   # 输入占位目录（默认仅 .gitkeep）
├── ml-sharp/                 # SHARP 子模块
├── models/                   # 模型目录
├── outputs/                  # 输出目录
├── src/aylm/                 # 主代码
│   ├── api/
│   ├── constitution/
│   └── tools/
├── tests/                    # 测试
├── run.sh                    # 一键脚本
└── pyproject.toml
```

## 9. 本地私有文件规范（提交骨架，不提交内容）

仓库使用 `local_only/` 统一存放本地资料。当前策略是：
- 提交目录骨架（`README_LOCAL.md`、`.gitkeep`），保证团队目录结构一致
- 不提交目录内实际私有内容（论文、测试数据、运行产物等）

- `local_only/research/`：论文、笔记、草稿
- `local_only/test_data/`：本地测试数据
- `local_only/run_outputs/`：本地运行产物归档
- `local_only/agent_notes/`：过程性临时文档
- `local_only/benchmarks/`：临时基准脚本

如果你有不需要上传的资料，统一放到 `local_only/` 下即可；目录结构会同步给其他人，但你的本地内容不会被提交。

## 10. 开发与测试

```bash
# 运行测试
pytest tests/ -v

# 代码质量
black --check src/aylm tests
isort --check-only src/aylm tests
ruff check src/aylm tests
mypy src/aylm --ignore-missing-imports
```

CI 当前包含：
- 单元测试（Linux + macOS）
- 代码格式和静态检查
- `run.sh --check-only` 与 `run.sh --demo` 的 smoke test（Python 3.11/3.12）

## 11. 常见问题

| 问题 | 原因 | 处理方式 |
|---|---|---|
| `ml-sharp` 安装失败 | 子模块未初始化 | `git submodule update --init --recursive` |
| `run.sh` 提示 Python 版本不支持 | 当前解释器不是 3.11/3.12 | 切换到 Python 3.11/3.12 并重建虚拟环境 |
| `run.sh` 提示未找到输入 | `inputs/` 只有占位文件 | 放入真实图片/视频，或先运行 `./run.sh --demo` |
| 模型不存在 | 首次未下载模型 | `aylm setup --download` |
| 依赖安装失败 | 网络或镜像问题 | 检查网络后重试，必要时配置可用镜像源 |

## 12. 许可证

MIT License。详见 `LICENSE`。
