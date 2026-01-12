# 依赖包许可证兼容性检查

## 主要依赖包许可证汇总

| 包名 | 版本 | 许可证 | 兼容性 | 备注 |
|------|------|--------|--------|------|
| torch | 2.8.0 | BSD-3-Clause | ✅ | PyTorch主框架 |
| torchvision | 0.23.0 | BSD-3-Clause | ✅ | PyTorch视觉库 |
| timm | 1.0.20 | Apache-2.0 | ✅ | Vision Transformer库 |
| gsplat | 1.5.3 | MIT | ✅ | 高斯渲染库 |
| open3d | 0.18.0 | MIT | ✅ | 3D处理库 |
| numpy | 2.3.3 | BSD-3-Clause | ✅ | 数值计算 |
| scipy | 1.16.2 | BSD-3-Clause | ✅ | 科学计算 |
| pillow | 11.3.0 | MIT | ✅ | 图像处理 |
| click | 8.1.8 | BSD-3-Clause | ✅ | 命令行界面 |
| plyfile | 1.1.2 | MIT | ✅ | PLY文件处理 |
| imageio | 2.37.0 | BSD-2-Clause | ✅ | 图像I/O |
| matplotlib | 3.8.4 | PSF | ✅ | 可视化 |
| requests | 2.32.5 | Apache-2.0 | ✅ | HTTP请求 |
| rich | 14.1.0 | MIT | ✅ | 终端美化 |
| tqdm | 4.67.1 | MIT | ✅ | 进度条 |
| huggingface-hub | 0.35.3 | Apache-2.0 | ✅ | HF模型中心 |
| safetensors | 0.6.2 | Apache-2.0 | ✅ | 安全张量 |

## 许可证兼容性分析

### 与 MIT 许可证的兼容性

- ✅ **完全兼容**：所有依赖包都使用宽松的开源许可证
- ✅ BSD/MIT/Apache 许可证都与 MIT 兼容
- ✅ 不存在GPL等传染性许可证冲突

### 与 Apache 2.0 许可证的兼容性

- ✅ **完全兼容**：所有依赖都与 Apache 2.0 兼容
- ✅ BSD/MIT 许可证兼容 Apache 2.0
- ✅ Apache 2.0 可以与 Apache 2.0 许可证混合

### 与 Apple 许可证的兼容性

- ⚠️ **当前使用**：项目使用 Apple 研究许可证
- ⚠️ 限制：仅限研究用途，不允许商业使用
- ⚠️ 建议：如果要开源，建议改为 MIT 或 Apache 2.0

## 许可证冲突检查

### ✅ 通过的检查项

1. **传染性许可证**：没有 GPL/LGPL 等传染性许可证
2. **专利条款**：所有许可证都包含适当的专利保护
3. **归属要求**：所有许可证都有合理的归属要求
4. **商业使用**：所有许可证都允许商业使用

### ⚠️ 需要注意的事项

1. **Apple SHARP 模型**：使用 Apple 专有的研究许可证
2. **模型权重**：2.7GB 的预训练模型可能有单独的许可证
3. **第三方服务**：自动下载的模型权重需要检查许可证

## 推荐许可证选择

基于依赖分析，推荐以下许可证：

### 1. MIT License (推荐)

- **优势**：简单、宽松、行业标准
- **兼容性**：与所有依赖完全兼容
- **限制**：几乎没有使用限制

### 2. Apache License 2.0

- **优势**：详细的专利保护条款
- **兼容性**：与所有依赖完全兼容
- **限制**：比 MIT 稍严格，但仍然宽松

### 3. 保持 Apple 许可证

- **优势**：符合上游 SHARP 模型的要求
- **限制**：仅限研究用途，不能商业化

## 实施建议

1. **选择许可证**：根据项目目标选择合适的许可证
2. **更新文件**：
   - 将选定的许可证重命名为 `LICENSE`
   - 在 README.md 中添加许可证徽章
   - 更新 setup.py/pyproject.toml 中的许可证字段
3. **法律咨询**：如有商业化计划，建议咨询律师
4. **文档更新**：在文档中明确说明许可证条款

## 许可证徽章

为 README.md 添加许可证徽章：

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
```
