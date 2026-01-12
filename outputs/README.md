# 输出文件目录

此目录将包含SHARP处理后的结果文件。

## 输出文件结构

```text
outputs/
├── output_gaussians/           # 主要输出目录
│   ├── *.ply                   # SHARP原始3D高斯模型
│   ├── cropped_pointclouds/    # 局部区域裁剪结果
│   │   └── cropped_*.ply       # 裁剪后的点云文件
│   └── voxelized_outputs/      # 体素化结果
│       └── voxelized_*.ply     # 1cm体素网格数据
```

## 文件说明

### SHARP原始模型 (*.ply)

- 格式：二进制PLY格式
- 大小：约63MB
- 点云数量：约117万个3D高斯点
- 包含属性：位置(x,y,z), 尺度, 旋转, 不透明度, 颜色(RGB)

### 局部裁剪模型 (cropped_*.ply)

- 格式：ASCII PLY格式
- 大小：约16MB
- 点云数量：约59万个点
- 空间范围：10m半圆范围，自适应距离阈值

### 体素化模型 (voxelized_*.ply)

- 格式：ASCII PLY格式
- 大小：约1.1MB
- 体素数量：约6万个占据体素
- 体素尺寸：5mm x 5mm x 5mm
- 坐标系：OpenCV标准(x右,y下,z前)
- 物理尺度：绝对尺度，单位为米

## 使用建议

- PLY文件可以使用Polycam、SuperSplat等3DGS渲染器查看
- 坐标系遵循OpenCV标准 (x右, y下, z前)
- voxelized文件适用于智能设备路径规划和空间感知
