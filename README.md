# A.YLM: Hybrid End-to-End Autonomous Driving Perception via 3D Gaussian Splatting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Abstract

A.YLM presents a novel hybrid end-to-end perception framework for autonomous driving that addresses the "black-box problem" inherent in conventional end-to-end models. By introducing **3D Gaussian Splatting (3DGS)** as an intermediate representation and transforming it into **Occupancy 2.0** voxel grids, our approach achieves interpretable spatial understanding while maintaining computational efficiency suitable for edge deployment.

The system pipeline follows: **Multi-view RGB → 3DGS Reconstruction → Voxelization (Occupancy 2.0) → 2D Detection with 3D Projection & Tracking → Structured Spatial Data (Position, Velocity, TTC)**.

Key contributions include: (1) a transparent intermediate representation that bridges the gap between raw sensor input and decision-making, (2) dimensionality reduction through voxelization enabling real-time vehicle-side inference, (3) integration of Apple's SHARP Vision Transformer for efficient single-image 3D reconstruction, and (4) multi-object tracking with motion vector estimation via ByteTrack and Kalman filtering.

---

## 1. Technical Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        A.YLM Perception Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐  │
│   │  RGB     │    │    SHARP     │    │   Voxel     │    │  Semantic    │  │
│   │  Input   │───▶│    3DGS      │───▶│   Grid      │───▶│  Fusion      │  │
│   │  (Multi- │    │  Reconstruct │    │  (Occupancy │    │  (YOLO +     │  │
│   │   view)  │    │              │    │    2.0)     │    │   3D Proj)   │  │
│   └──────────┘    └──────────────┘    └─────────────┘    └──────────────┘  │
│                                                                  │          │
│                                                                  ▼          │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                    Navigation Output Module                          │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│   │  │  Obstacle   │  │  ByteTrack  │  │   Kalman    │  │  Structured │  │  │
│   │  │  Clustering │  │    MOT      │  │   Filter    │  │  JSON/PLY   │  │  │
│   │  │  (DBSCAN)   │  │             │  │  (Velocity) │  │   Output    │  │  │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Processing Pipeline

| Stage | Module | Input | Output | Key Algorithm |
|-------|--------|-------|--------|---------------|
| 1 | 3D Reconstruction | RGB Image | 3D Gaussian Splatting | Apple SHARP (ViT) |
| 2 | Voxelization | 3DGS Point Cloud | Occupancy Grid | Spatial Hashing |
| 3 | Ground Removal | Voxel Grid | Filtered Grid | RANSAC Plane Fitting |
| 4 | Object Detection | RGB Image | 2D Bounding Boxes | YOLO11 Instance Segmentation |
| 5 | Semantic Fusion | 2D Detections + 3D Points | Labeled Point Cloud | Camera Projection Matrix |
| 6 | Obstacle Extraction | Semantic Point Cloud | 3D Bounding Boxes | DBSCAN Clustering |
| 7 | Motion Estimation | Multi-frame Detections | Velocity Vectors | ByteTrack + Kalman Filter |

---

## 2. Core Innovations

### 2.1 Intermediate Representation: 3DGS → Voxel Pathway

Traditional end-to-end autonomous driving models suffer from the **"black-box problem"** — the decision-making process lacks interpretability, making debugging and safety validation challenging. A.YLM addresses this by introducing an explicit **intermediate representation**:

```
Raw Sensor Data → 3D Gaussian Splatting → Voxel Occupancy Grid → Decision
                        ↑                        ↑
                   Interpretable            Geometric
                   3D Structure             Constraint
```

The 3DGS representation provides:
- **Geometric Constraint**: Explicit 3D structure enables physics-based reasoning
- **Interpretability**: Each Gaussian primitive corresponds to observable scene elements
- **Debugging Capability**: Intermediate outputs can be visualized and validated

### 2.2 Computational Optimization via Voxelization

The transformation from 3DGS to voxel grid achieves **dimensionality reduction**:

| Representation | Data Size (typical) | Inference Complexity |
|----------------|---------------------|----------------------|
| Raw 3DGS | ~500K Gaussians | O(n²) rendering |
| Voxel Grid (5cm) | ~50K voxels | O(n) lookup |
| Sparse Voxel | ~10K occupied | O(k) where k << n |

This compression enables **real-time vehicle-side deployment** on edge computing platforms while preserving essential spatial information for navigation.

### 2.3 Apple SHARP Integration

We leverage Apple's **SHARP (Single-image 3D Human And Room Perception)** model, a Vision Transformer-based architecture that achieves:

- **Single-image 3D reconstruction**: Eliminates multi-view capture requirements
- **Reduced rendering pressure**: Direct point cloud output vs. iterative NeRF rendering
- **Edge-optimized inference**: Designed for Apple Silicon (MPS acceleration)

### 2.4 Multi-Object Tracking with Motion Estimation

The tracking module implements a two-stage approach:

1. **ByteTrack Association**: Robust multi-object tracking handling occlusions
2. **Kalman Filter State Estimation**: Predicts velocity vectors and Time-To-Collision (TTC)

```python
# State vector: [x, y, z, vx, vy, vz, ax, ay, az]
# Observation: [x, y, z] from 3D detection
# Output: Position, Velocity, Acceleration, TTC
```

### 2.5 Sparse Voxel Optimization Interface

The architecture reserves interfaces for **Sparse Voxel** optimization:

```python
class SparseVoxelGrid:
    """Future optimization: Only store occupied voxels"""
    def __init__(self, resolution: float = 0.05):
        self.occupied_voxels: Dict[Tuple[int,int,int], VoxelData]
        self.spatial_hash: SpatialHashMap
```

### 2.6 Multi-Sensor Fusion Potential

The voxel-based representation naturally supports **LiDAR fusion**:

```
Camera Voxels ──┐
                ├──▶ Fused Occupancy Grid ──▶ Enhanced Robustness
LiDAR Voxels ───┘
```

---

## 3. System Requirements

### 3.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 16GB |
| GPU | - | CUDA 11.0+ / Apple MPS |
| Storage | 5GB | 20GB (with models) |

### 3.2 Software Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ (3.11 recommended) | Runtime |
| PyTorch | 2.0+ | Deep Learning Framework |
| Open3D | 0.17+ | Point Cloud Processing |
| Ultralytics | 8.0+ | YOLO Object Detection |
| NumPy | 1.24+ | Numerical Computing |

---

## 4. Installation

### 4.1 Quick Start

```bash
# Clone repository with submodules
git clone --recursive https://github.com/appergb/A.YLM.git
cd A.YLM

# Create virtual environment (Python 3.11 recommended for Open3D compatibility)
python3.11 -m venv aylm_env
source aylm_env/bin/activate

# Install core dependencies
pip install -e .
pip install -e ml-sharp/

# Install full feature set (Open3D + YOLO)
pip install -e ".[full]"
```

### 4.2 Model Setup

```bash
# SHARP model (~2.8GB) - auto-downloads on first use
aylm setup --download

# YOLO model (~6MB) - auto-downloads on first use
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')"
```

---

## 5. Usage

### 5.1 One-Click Execution

```bash
# Complete workflow: setup → predict → voxelize → semantic fusion
./run.sh

# Custom input directory
./run.sh --input /path/to/images

# Force pipeline mode for batch processing
./run.sh --pipeline
```

### 5.2 CLI Commands

```bash
# Environment setup and model download
aylm setup --download

# Stage 1: SHARP 3D Reconstruction
aylm predict -i inputs/input_images -o outputs/output_gaussians -v

# Stage 2: Voxelization (Occupancy 2.0)
aylm voxelize -i outputs/output_gaussians --voxel-size 0.005

# Stage 3: Complete Pipeline (Sequential)
aylm process -v

# Stage 4: Parallel Pipeline (Multi-image)
aylm pipeline -i inputs/input_images -o outputs/output_gaussians -v
```

### 5.3 Video Processing

```bash
# Frame extraction
aylm video extract -i video.mp4 -o frames/ --interval 1.0

# Full video pipeline (extract + inference + voxelize + tracking)
aylm video process -i video.mp4 -o output/ --use-gpu

# Visualization playback
aylm video play -i voxels/ --fps 10 --loop
```

### 5.4 Python API

```python
from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig
from aylm.tools.semantic_fusion import SemanticFusion
from aylm.tools.object_detector import ObjectDetector
from aylm.tools.pipeline_processor import PipelineProcessor, PipelineConfig

# Stage 1: Voxelization with Occupancy 2.0
config = VoxelizerConfig(voxel_size=0.005)  # 5mm resolution
voxelizer = PointCloudVoxelizer(config=config)
voxelizer.process(
    input_path="gaussians.ply",
    output_path="voxelized.ply",
    remove_ground=True  # RANSAC ground plane removal
)

# Stage 2: Semantic Fusion (2D → 3D Projection)
detector = ObjectDetector(model_name="yolo11n-seg.pt")
detections = detector.detect("image.png")

fusion = SemanticFusion()
semantic_pc = fusion.fuse_semantics(
    ply_path="gaussians.ply",
    image_path="image.png",
    detections=detections
)

# Stage 3: Navigation Mesh Generation
fusion.save_navigation_ply(semantic_pc, "navigation.ply", voxel_size=0.05)

# Full Pipeline Processing
pipeline_config = PipelineConfig(
    verbose=True,
    enable_semantic_fusion=True,
    output_navigation_ply=True
)
processor = PipelineProcessor(pipeline_config)
results = processor.run_pipeline(
    input_dir="inputs/input_images",
    output_dir="outputs"
)
```

---

## 6. Output Specification

### 6.1 File Formats

| File Pattern | Format | Description |
|--------------|--------|-------------|
| `*.ply` | PLY | Raw 3DGS point cloud from SHARP |
| `vox_*.ply` | PLY | Voxelized point cloud with semantic colors |
| `nav_*.ply` | PLY | Navigation mesh (5cm solid cubes) |
| `*_obstacles.json` | JSON | Structured obstacle data for planning |

### 6.2 Obstacle JSON Schema

```json
{
  "metadata": {
    "timestamp": "2026-02-27T10:30:00Z",
    "frame_id": 42,
    "coordinate_systems": {
      "cv": {"axes": "X-right, Y-down, Z-forward"},
      "robot": {"axes": "X-forward, Y-left, Z-up (ENU)"}
    }
  },
  "obstacles": [
    {
      "id": 1,
      "type": "dynamic",
      "category": "vehicle",
      "center_cv": [10.32, 0.06, 0.21],
      "center_robot": [0.21, -10.32, -0.06],
      "dimensions": [7.52, 3.36, 2.26],
      "velocity": [2.5, 0.0, 0.1],
      "ttc": 4.2,
      "confidence": 0.93
    }
  ]
}
```

### 6.3 Semantic Label Taxonomy

| Label ID | Category | Color (RGB) | Motion Type |
|----------|----------|-------------|-------------|
| 0 | GROUND | (139, 69, 19) | Static |
| 1 | VEHICLE | (0, 0, 255) | Dynamic |
| 2 | PEDESTRIAN | (255, 0, 0) | Dynamic |
| 3 | BICYCLE | (0, 255, 255) | Dynamic |
| 4 | BUILDING | (128, 128, 128) | Static |
| 5 | VEGETATION | (0, 255, 0) | Static |
| 255 | UNKNOWN | (255, 255, 255) | Unknown |

---

## 7. Configuration

### 7.1 Environment Variables

```bash
export AYLM_ROOT="/path/to/project"
export INPUT_DIR="/path/to/images"
export OUTPUT_DIR="/path/to/output"
export CUDA_VISIBLE_DEVICES="0"  # GPU selection
```

### 7.2 Voxelizer Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `voxel_size` | 0.005 | 0.001-0.1 | Voxel resolution (meters) |
| `statistical_nb_neighbors` | 20 | 5-50 | Outlier removal neighbors |
| `ransac_distance_threshold` | 0.02 | 0.01-0.1 | RANSAC plane threshold |
| `ground_normal_threshold` | 0.8 | 0.5-1.0 | Ground normal Y-component |

### 7.3 Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_VOXEL_SIZE` | 0.05 | Navigation voxel size (5cm) |
| `DEFAULT_SLICE_RADIUS` | 20.0 | Processing radius (meters) |
| `DEFAULT_FOV_DEGREES` | 60.0 | Camera field of view |
| `DEFAULT_DENSITY_THRESHOLD` | 3 | Minimum points per voxel |

---

## 8. Project Structure

```
A.YLM/
├── src/aylm/                        # Core Package
│   ├── cli.py                       # Command Line Interface
│   └── tools/                       # Processing Modules
│       ├── pointcloud_voxelizer.py  # Occupancy 2.0 Voxelization
│       ├── pipeline_processor.py    # Parallel Pipeline Orchestration
│       ├── video_pipeline.py        # Video Sequence Processing
│       ├── semantic_fusion.py       # 2D→3D Semantic Projection
│       ├── object_detector.py       # YOLO Instance Segmentation
│       ├── obstacle_marker.py       # DBSCAN Obstacle Clustering
│       ├── pointcloud_slicer.py     # Spatial ROI Extraction
│       └── coordinate_utils.py      # CV↔Robot Coordinate Transform
├── ml-sharp/                        # Apple SHARP Model (submodule)
├── models/                          # Model Checkpoints
├── inputs/                          # Input Data
│   ├── input_images/                # RGB Images
│   └── videos/                      # Video Files
├── outputs/                         # Output Data
│   ├── output_gaussians/            # 3DGS Point Clouds
│   ├── voxelized/                   # Occupancy Grids
│   ├── detections/                  # Detection Results
│   └── navigation/                  # Navigation Meshes
├── tests/                           # Test Suite
├── run.sh                           # One-Click Execution Script
└── pyproject.toml                   # Project Configuration
```

---

## 9. Future Work

### 9.1 Sparse Voxel Optimization
Implementation of sparse voxel data structures to further reduce memory footprint and enable higher resolution occupancy grids.

### 9.2 Multi-Sensor Fusion
Integration with LiDAR point clouds for enhanced depth accuracy and robustness in adverse lighting conditions.

### 9.3 Temporal Consistency
Incorporation of recurrent architectures (LSTM/Transformer) for improved temporal coherence in video sequences.

### 9.4 End-to-End Training
Joint optimization of the entire pipeline from raw sensor input to navigation output.

---

## 10. Development

### 10.1 Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest --cov=aylm --cov-report=html

# Run specific module tests
pytest tests/unit/test_semantic_fusion.py -v
```

### 10.2 Code Quality

```bash
# Format code
black src/aylm tests
isort src/aylm tests

# Linting
ruff check src/aylm tests

# Type checking
mypy src/aylm
```

---

## 11. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Open3D installation failed | Python 3.13 incompatibility | Use Python 3.11 |
| YOLO model not found | First-time download | Run `pip install ultralytics` |
| SHARP model download failed | Network issue | Manual download from Apple CDN |
| Out of memory | Large point cloud | Increase `voxel_size` or use `--slice-radius` |
| CUDA/MPS unavailable | Driver issue | Check `torch.cuda.is_available()` |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Apple Inc.** - SHARP Vision Transformer model and research
- **Ultralytics** - YOLO object detection framework
- **Open3D Community** - Point cloud processing library
- **PyTorch Team** - Deep learning framework

## Citation

If you use A.YLM in your research, please cite:

```bibtex
@software{aylm2026,
  title={A.YLM: Hybrid End-to-End Autonomous Driving Perception via 3D Gaussian Splatting},
  author={TRIP (appergb)},
  year={2026},
  url={https://github.com/appergb/A.YLM}
}
```

---

**Author**: TRIP (appergb)

**Contributors**: Claude Code, Junie

**Project Status**: Production Ready
