# A-YLM: A Geometric Safety Supervisor for E2E Driving

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Abstract

A-YLM is a **Geometric Safety Supervisor** for end-to-end autonomous driving systems. Unlike traditional perception frameworks, A-YLM serves as an **external safety validation module** — a "Physical Examiner for AI" that provides independent geometric verification of driving decisions.

The core philosophy: **AI needs a physical ground truth supervisor**. While end-to-end models excel at pattern recognition, they lack explicit geometric reasoning. A-YLM bridges this gap by constructing real-time 3D geometric representations via **3D Gaussian Splatting (3DGS)** and validating driving decisions against physical reality.

Key capabilities:
- **Edge-deployed Safety Validation**: Lightweight geometric verification running alongside E2E models
- **Self-Supervised Safety Learning**: Learns safety boundaries from geometric constraints without manual labeling
- **Physical Ground Truth**: Provides interpretable 3D spatial understanding as safety reference
- **AI Self-Evolution**: Continuous improvement through geometric feedback loops

The system pipeline: **Multi-view RGB → 3DGS Reconstruction → Voxelization (Occupancy 2.0) → Safety Validation → Geometric Feedback**.

### Multi-Purpose Foundation Framework

Beyond safety supervision, A-YLM serves as a **versatile foundation framework** for various 3D perception applications:

| Application | Description | Use Case |
|-------------|-------------|----------|
| **Safety Supervisor** | Geometric validation for E2E driving | Primary use case |
| **3D Perception** | Standalone 3D scene understanding | Robotics, AR/VR |
| **Occupancy Mapping** | Real-time voxel-based environment mapping | Navigation, SLAM |
| **Semantic Fusion** | 2D-to-3D semantic projection | Scene understanding |
| **Object Tracking** | Multi-object 3D tracking with motion estimation | Surveillance, analytics |

The modular architecture allows mixing and matching components for custom pipelines.

---

## 1. Core Concept: Physical Examiner for AI

### 1.1 The Safety Supervision Problem

End-to-end autonomous driving models face a fundamental challenge: **the black-box problem**. These models make decisions without explicit geometric reasoning, making safety validation difficult.

A-YLM addresses this by serving as an **independent safety supervisor**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    A-YLM: Geometric Safety Supervisor                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                              ┌──────────────────────┐   │
│   │   E2E Model  │──── Driving Decision ────▶  │   Safety Validator   │   │
│   │  (Black Box) │                              │  (Geometric Check)   │   │
│   └──────────────┘                              └──────────────────────┘   │
│          │                                               │                  │
│          │                                               ▼                  │
│          │                                      ┌──────────────────────┐   │
│          │                                      │   Physical Ground    │   │
│          │                                      │       Truth          │   │
│          │                                      │   (3DGS + Voxels)    │   │
│          │                                      └──────────────────────┘   │
│          │                                               │                  │
│          ▼                                               ▼                  │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                     Safety Decision Gate                              │ │
│   │   • Validate against geometric constraints                           │ │
│   │   • Override unsafe decisions                                        │ │
│   │   • Provide safety feedback for learning                             │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Self-Supervised Safety Learning

A-YLM implements **self-supervised safety learning** through geometric constraints:

1. **Geometric Constraint Extraction**: 3DGS provides explicit 3D structure
2. **Safety Boundary Learning**: Learn safe operating regions from physical geometry
3. **Feedback Loop**: Geometric violations trigger safety interventions and learning updates

---

## 2. Technical Architecture

### 2.1 Safety Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        A-YLM Safety Validation Pipeline                     │
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
│   │                    Safety Validation Module                          │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│   │  │  Obstacle   │  │  Collision  │  │   Safety    │  │  Geometric  │  │  │
│   │  │  Detection  │  │  Prediction │  │  Boundary   │  │  Feedback   │  │  │
│   │  │  (DBSCAN)   │  │   (TTC)     │  │  Validation │  │   Output    │  │  │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Processing Stages

| Stage | Module | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| 1 | 3D Reconstruction | RGB Image | 3D Gaussian Splatting | Physical Ground Truth |
| 2 | Voxelization | 3DGS Point Cloud | Occupancy Grid | Geometric Representation |
| 3 | Ground Removal | Voxel Grid | Filtered Grid | Obstacle Isolation |
| 4 | Object Detection | RGB Image | 2D Bounding Boxes | Semantic Understanding |
| 5 | Semantic Fusion | 2D + 3D | Labeled Point Cloud | Safety Context |
| 6 | Safety Validation | Semantic 3D | Safety Decisions | Geometric Verification |
| 7 | Feedback Generation | Validation Results | Learning Signals | Self-Evolution |

---

## 3. Core Innovations

### 3.1 Physical Ground Truth via 3DGS

The 3DGS representation provides **physical ground truth** for safety validation:

```
Raw Sensor Data → 3D Gaussian Splatting → Voxel Occupancy Grid → Safety Validation
                        ↑                        ↑
                   Physical                 Geometric
                   Reality                  Constraint
```

Benefits:
- **Interpretable**: Each Gaussian corresponds to physical scene elements
- **Verifiable**: Intermediate outputs can be visualized and validated
- **Geometric**: Explicit 3D structure enables physics-based safety reasoning

### 3.2 Edge-Deployed Safety Module

A-YLM is designed for **edge deployment** alongside E2E models:

| Representation | Data Size | Inference Complexity |
|----------------|-----------|----------------------|
| Raw 3DGS | ~500K Gaussians | O(n²) rendering |
| Voxel Grid (5cm) | ~50K voxels | O(n) lookup |
| Sparse Voxel | ~10K occupied | O(k) where k << n |

This compression enables **real-time safety validation** on edge computing platforms.

### 3.3 Self-Supervised Safety Learning

The system learns safety boundaries without manual labeling:

1. **Geometric Constraint Mining**: Extract safety rules from 3D structure
2. **Violation Detection**: Identify when E2E decisions violate geometric constraints
3. **Feedback Generation**: Produce learning signals for model improvement

### 3.4 Apple SHARP Integration

We leverage Apple's **SHARP** model for efficient 3D reconstruction:

- **Single-image 3D reconstruction**: Eliminates multi-view requirements
- **Edge-optimized inference**: Designed for Apple Silicon (MPS acceleration)
- **Real-time capable**: Suitable for safety-critical applications

---

## 4. System Requirements

### 4.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 16GB |
| GPU | - | CUDA 11.0+ / Apple MPS |
| Storage | 5GB | 20GB (with models) |

### 4.2 Software Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ (3.11 recommended) | Runtime |
| PyTorch | 2.0+ | Deep Learning Framework |
| Open3D | 0.17+ | Point Cloud Processing |
| Ultralytics | 8.0+ | YOLO Object Detection |
| NumPy | 1.24+ | Numerical Computing |

---

## 5. Installation

### 5.1 Quick Start

```bash
# Clone repository with submodules
git clone --recursive https://github.com/appergb/A.YLM.git
cd A.YLM

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv aylm_env
source aylm_env/bin/activate

# Install core dependencies
pip install -e .
pip install -e ml-sharp/

# Install full feature set
pip install -e ".[full]"
```

### 5.2 Model Setup

```bash
# SHARP model (~2.8GB) - auto-downloads on first use
aylm setup --download

# YOLO model (~6MB) - auto-downloads on first use
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')"
```

---

## 6. Usage

### 6.1 One-Click Execution

```bash
# Complete safety validation workflow
./run.sh

# Custom input directory
./run.sh --input /path/to/images

# Force pipeline mode for batch processing
./run.sh --pipeline
```

### 6.2 CLI Commands

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

### 6.3 Video Processing

```bash
# Frame extraction
aylm video extract -i video.mp4 -o frames/ --interval 1.0

# Full video pipeline (extract + inference + voxelize + tracking)
aylm video process -i video.mp4 -o output/ --use-gpu

# Visualization playback
aylm video play -i voxels/ --fps 10 --loop
```

### 6.4 Python API

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

# Stage 3: Safety Validation Output
fusion.save_navigation_ply(semantic_pc, "safety_validation.ply", voxel_size=0.05)

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

## 7. Output Specification

### 7.1 File Formats

| File Pattern | Format | Description |
|--------------|--------|-------------|
| `*.ply` | PLY | Raw 3DGS point cloud from SHARP |
| `vox_*.ply` | PLY | Voxelized point cloud with semantic colors |
| `nav_*.ply` | PLY | Safety validation mesh (5cm solid cubes) |
| `*_obstacles.json` | JSON | Structured obstacle data for safety validation |

### 7.2 Safety Validation JSON Schema

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
      "confidence": 0.93,
      "safety_status": "warning"
    }
  ],
  "safety_summary": {
    "overall_status": "safe",
    "collision_risk": 0.15,
    "geometric_violations": []
  }
}
```

### 7.3 Semantic Label Taxonomy

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

## 8. Configuration

### 8.1 Environment Variables

```bash
export AYLM_ROOT="/path/to/project"
export INPUT_DIR="/path/to/images"
export OUTPUT_DIR="/path/to/output"
export CUDA_VISIBLE_DEVICES="0"  # GPU selection
```

### 8.2 Voxelizer Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `voxel_size` | 0.005 | 0.001-0.1 | Voxel resolution (meters) |
| `statistical_nb_neighbors` | 20 | 5-50 | Outlier removal neighbors |
| `ransac_distance_threshold` | 0.02 | 0.01-0.1 | RANSAC plane threshold |
| `ground_normal_threshold` | 0.8 | 0.5-1.0 | Ground normal Y-component |

### 8.3 Safety Validation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_VOXEL_SIZE` | 0.05 | Safety validation voxel size (5cm) |
| `DEFAULT_SLICE_RADIUS` | 20.0 | Processing radius (meters) |
| `DEFAULT_FOV_DEGREES` | 60.0 | Camera field of view |
| `DEFAULT_DENSITY_THRESHOLD` | 3 | Minimum points per voxel |
| `TTC_WARNING_THRESHOLD` | 3.0 | Time-to-collision warning (seconds) |
| `TTC_CRITICAL_THRESHOLD` | 1.5 | Time-to-collision critical (seconds) |

---

## 9. Project Structure

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
│   └── safety_validation/           # Safety Validation Results
├── tests/                           # Test Suite
├── run.sh                           # One-Click Execution Script
└── pyproject.toml                   # Project Configuration
```

---

## 10. Future Work

### 10.1 Advanced Safety Learning
Implementation of reinforcement learning for adaptive safety boundary optimization.

### 10.2 Multi-Sensor Fusion
Integration with LiDAR for enhanced geometric accuracy in safety-critical scenarios.

### 10.3 Temporal Safety Reasoning
Incorporation of temporal models for predictive safety validation.

### 10.4 Federated Safety Learning
Distributed learning across vehicle fleets for collective safety improvement.

---

## 11. Development

### 11.1 Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest --cov=aylm --cov-report=html

# Run specific module tests
pytest tests/unit/test_semantic_fusion.py -v
```

### 11.2 Code Quality

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

## 12. Troubleshooting

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

If you use A-YLM in your research, please cite:

```bibtex
@software{aylm2026,
  title={A-YLM: A Geometric Safety Supervisor for End-to-End Autonomous Driving},
  author={TRIP (appergb)},
  year={2026},
  url={https://github.com/appergb/A.YLM}
}
```

---

**Author**: TRIP (appergb)

**Contributors**: Claude Code, Junie

**Project Status**: Production Ready
