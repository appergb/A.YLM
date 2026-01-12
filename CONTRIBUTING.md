# 贡献指南

欢迎为 A.YLM (单图像3D重建与智能导航系统) 项目贡献代码！我们非常感谢您的帮助。

**项目声明**: 此项目由TRIP(appergb)进行个人研发，其中参与closer和true。

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone https://github.com/appergb/A.YLM.git
cd A.YLM

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -e ml-sharp/
pip install -r docs/requirements-dev.txt  # 如果有的话
```

### 2. 运行测试

```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_voxelizer.py

# 运行代码质量检查
python -m flake8 src/
python -m black --check src/
python -m isort --check-only src/
```

## 开发工作流程

### 1. 创建功能分支

```bash
# 从 main 分支创建新分支
git checkout -b feature/your-feature-name

# 或修复 bug
git checkout -b fix/issue-number-description
```

### 2. 编写代码

- 遵循现有的代码风格
- 添加适当的文档字符串
- 为新功能编写测试
- 更新相关文档

### 3. 提交更改

```bash
# 添加更改的文件
git add .

# 提交时使用清晰的提交信息
git commit -m "feat: add new voxelization algorithm

- Implement adaptive ground detection
- Add support for material-aware noise
- Improve processing speed by 20%

Closes #123"
```

#### 提交信息格式

我们使用 [Conventional Commits](https://conventionalcommits.org/) 格式：

- `feat:` 新功能
- `fix:` 修复bug
- `docs:` 文档更改
- `style:` 代码格式更改
- `refactor:` 代码重构
- `test:` 添加测试
- `chore:` 构建过程或工具更改

### 4. 创建 Pull Request

1. 推送到您的分支：`git push origin feature/your-feature-name`
2. 在 GitHub 上创建 Pull Request
3. 填写 PR 模板，提供详细描述
4. 请求代码审查

## 代码质量要求

### 代码风格

我们使用以下工具确保代码质量：

- **Black**: 代码格式化
- **isort**: import 语句排序
- **flake8**: 代码质量检查
- **mypy**: 类型检查 (可选)

```bash
# 自动格式化代码
python -m black src/
python -m isort src/

# 检查代码质量
python -m flake8 src/
```

### 测试要求

- 为所有新功能编写单元测试
- 保持测试覆盖率在 80% 以上
- 测试应放在 `tests/` 目录下

```bash
# 运行测试并生成覆盖率报告
python -m pytest --cov=src --cov-report=html
```

### 文档要求

- 为所有公共函数和类添加 docstring
- 使用 Google 风格的 docstring 格式
- 更新 README.md 和相关文档

## 项目结构

```text
A.YLM/
├── ml-sharp/              # SHARP 源码和模型
│   ├── src/sharp/         # 主要代码
│   └── tests/             # 测试文件
├── scripts/               # 自定义脚本
│   ├── pointcloud_voxelizer.py    # 体素化处理
│   ├── coordinate_utils.py        # 坐标转换
│   └── undistort_iphone.py        # 镜头校正
├── inputs/                # 输入文件目录
├── outputs/               # 输出文件目录
├── docs/                  # 文档
├── requirements.txt       # Python 依赖
├── run_sharp.sh          # 主启动脚本
└── README.md             # 项目说明
```

## 贡献类型

### 🐛 Bug 修复

- 修复已知问题
- 改进错误处理
- 修复兼容性问题

### ✨ 新功能

- 添加新的体素化算法
- 支持新的图像格式
- 改进导航功能

### 📚 文档

- 改进现有文档
- 添加使用示例
- 翻译文档

### 🧪 测试

- 增加测试覆盖率
- 修复测试用例
- 添加集成测试

### 🛠️ 工具和基础设施

- 改进构建过程
- 更新依赖版本
- 改进开发工具

## 行为准则

我们致力于提供一个友好、包容的环境。请：

- ✅ 尊重所有贡献者
- ✅ 提供建设性的反馈
- ✅ 接受善意的建议
- ✅ 专注于解决问题
- ❌ 避免人身攻击
- ❌ 不要发布不适当的内容

## 许可证

通过贡献代码，您同意您的贡献将根据项目的许可证进行许可。请确保您有权许可您的贡献。

## 联系方式

- 📧 **邮箱**: <your-email@example.com>
- 💬 **讨论区**: [GitHub Discussions](https://github.com/appergb/A.YLM/discussions)
- 🐛 **问题跟踪**: [GitHub Issues](https://github.com/appergb/A.YLM/issues)

感谢您的贡献！🚀
