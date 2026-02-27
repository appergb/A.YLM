# A-YLM: Geometric Constitutional AI for Embodied Intelligence

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org)

**Self-Supervised Safety Framework | Autonomous Driving | Robotics | Embodied AI**

*Extending Anthropic's Constitutional AI paradigm from language to the physical world*

[Paper](#) | [Documentation](#) | [Demo](#) | [中文文档](docs/paper_aylm_zh.md)

</div>

---

## Vision: Physical Constitution for AI

> *"Just as Constitutional AI teaches language models ethical boundaries through textual principles, A-YLM teaches embodied AI systems physical boundaries through geometric constraints."*

A-YLM represents a paradigm shift in AI safety — from **reactive safety checks** to **constitutional safety learning**. We believe that for AI to truly understand and safely navigate the physical world, it needs more than perception; it needs a **geometric constitution** that defines the fundamental laws of physical interaction.

---

## Abstract

A-YLM is a **Geometric Constitutional AI** framework for embodied intelligence systems. Inspired by Anthropic's Constitutional AI approach for language models, A-YLM extends this safety paradigm to the **physical world** — providing geometric constraints as a "Physical Constitution" that governs AI behavior in 3D space.

The core philosophy: **AI needs physical laws, not just language rules**. While Constitutional AI teaches language models ethical boundaries through textual principles, A-YLM teaches embodied AI systems physical boundaries through geometric constraints. This enables AI to **understand the world** through self-supervised learning from massive geometric data.

### Key Innovations

| Concept | Language AI (Constitutional AI) | Embodied AI (A-YLM) |
|---------|--------------------------------|---------------------|
| **Constitution** | Ethical principles in text | Geometric constraints in 3D |
| **Supervision** | RLHF with human feedback | Self-supervised geometric validation |
| **Learning** | Learn from text corrections | Learn from physical violations |
| **Safety** | Prevent harmful outputs | Prevent unsafe physical actions |
| **Evolution** | Iterative refinement | Continuous geometric feedback |

### Core Capabilities

- **Geometric Constitutional Supervision**: Physical laws as AI behavior boundaries
- **Self-Supervised World Understanding**: AI learns physics through geometric feedback — no human labeling required
- **Embodied Intelligence Training**: Continuous self-evolution from massive 3D data
- **Universal Applicability**: Not just autonomous driving — all embodied AI systems (robotics, drones, AR/VR, humanoids)
- **Edge-Deployable**: Real-time geometric supervision on resource-constrained devices

The system pipeline: **Sensor Input → 3DGS Reconstruction → Voxelization → Constitutional Validation → Geometric Feedback → Self-Learning**.

---

## Why Geometric Constitutional AI?

### The Problem with Current AI Safety

Current embodied AI systems face a fundamental challenge:

| Approach | Limitation |
|----------|------------|
| **Rule-based Safety** | Cannot generalize to novel situations |
| **Learned Safety** | Requires massive human-labeled data |
| **End-to-End Models** | Black-box decisions, uninterpretable |
| **Simulation-based** | Sim-to-real gap, limited coverage |

### Our Solution: Physical Constitution

A-YLM introduces **Geometric Constitutional AI** — a self-supervised framework where:

1. **Physical laws become the constitution**: Geometric constraints (collision avoidance, spatial boundaries) serve as inviolable principles
2. **AI learns from violations**: When AI decisions violate geometric constraints, the system generates training signals automatically
3. **No human labeling needed**: The physical world itself provides ground truth
4. **Continuous evolution**: AI improves its world understanding through every interaction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Geometric Constitutional AI Loop                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐         │
│    │   AI        │ ──────▶ │  Geometric  │ ──────▶ │  Physical   │         │
│    │   Decision  │         │  Validation │         │  Execution  │         │
│    └─────────────┘         └─────────────┘         └─────────────┘         │
│           ▲                       │                       │                 │
│           │                       │ Violation?            │                 │
│           │                       ▼                       │                 │
│    ┌─────────────┐         ┌─────────────┐               │                 │
│    │   World     │ ◀────── │  Feedback   │ ◀─────────────┘                 │
│    │   Model     │         │  Generation │   Physical Outcome              │
│    │   Update    │         └─────────────┘                                 │
│    └─────────────┘                                                         │
│           │                                                                 │
│           └──────────────── Self-Evolution ────────────────────────────────│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Geometric Constitutional AI: A New Safety Paradigm

### 1.1 From Language Constitution to Physical Constitution

Anthropic's Constitutional AI revolutionized language model safety by embedding ethical principles directly into the training process. A-YLM extends this paradigm to **embodied intelligence**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Constitutional AI Paradigm Comparison                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Language AI (Anthropic)              Embodied AI (A-YLM)                  │
│   ┌─────────────────────┐              ┌─────────────────────┐              │
│   │  Text Constitution  │              │ Geometric Constitution│             │
│   │  "Be helpful,       │              │ "Respect physical    │             │
│   │   harmless, honest" │              │  boundaries, avoid   │             │
│   └─────────────────────┘              │  collisions, obey    │             │
│            │                           │  spatial constraints"│             │
│            ▼                           └─────────────────────┘              │
│   ┌─────────────────────┐                       │                           │
│   │   RLHF Training     │                       ▼                           │
│   │   (Human Feedback)  │              ┌─────────────────────┐              │
│   └─────────────────────┘              │  Self-Supervised    │              │
│            │                           │  Geometric Learning │              │
│            ▼                           └─────────────────────┘              │
│   ┌─────────────────────┐                       │                           │
│   │   Safe Language     │                       ▼                           │
│   │   Outputs           │              ┌─────────────────────┐              │
│   └─────────────────────┘              │  Safe Physical      │              │
│                                        │  Actions            │              │
│                                        └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Self-Supervised World Understanding (Embodied Intelligence)

A-YLM enables AI to **understand the physical world** through continuous geometric learning — the essence of **embodied intelligence**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Self-Supervised World Understanding                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Stage 1: Geometric Data Collection                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Camera → 3DGS Reconstruction → Voxel Grid → Physical Ground Truth  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   Stage 2: Constitutional Validation                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  AI Decision → Geometric Check → Collision? Boundary? TTC?          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   Stage 3: Violation Detection & Feedback                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Violation Detected → Generate Training Signal → Update World Model │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   Stage 4: AI Self-Evolution                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Improved World Understanding → Better Decisions → Safer Actions    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This creates a **self-improving loop** where AI learns physics without human labeling — true embodied intelligence.

### 1.3 Universal Embodied Intelligence Applications

While autonomous driving is our primary demonstration, A-YLM's geometric constitutional approach applies to **all embodied AI**:

| Domain | Application | Geometric Constitution | Self-Learning Signal |
|--------|-------------|------------------------|----------------------|
| **Autonomous Driving** | Vehicle safety | Collision avoidance, lane boundaries | Near-miss detection |
| **Robotics** | Manipulation safety | Workspace limits, force constraints | Contact detection |
| **Drones/UAV** | Flight safety | Obstacle avoidance, no-fly zones | Proximity alerts |
| **AR/VR** | Spatial interaction | Physical object boundaries | Occlusion conflicts |
| **Humanoid Robots** | Navigation safety | Human proximity, obstacle clearance | Social distance violations |
| **Industrial Automation** | Workspace safety | Equipment boundaries, safety zones | Zone intrusion |

---

## 2. Technical Architecture

### 2.1 Geometric Safety Supervisor

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
  title={A-YLM: Geometric Constitutional AI for Embodied Intelligence},
  author={TRIP (appergb)},
  year={2026},
  url={https://github.com/appergb/A.YLM},
  note={Self-supervised safety framework extending Constitutional AI to physical world}
}
```

---

## Acknowledgments

- **Anthropic** - Constitutional AI paradigm that inspired this work
- **Apple Inc.** - SHARP Vision Transformer model and research
- **Ultralytics** - YOLO object detection framework
- **Open3D Community** - Point cloud processing library
- **PyTorch Team** - Deep learning framework

---

## Contact & Community

- **Author**: TRIP (appergb)
- **Contributors**: Claude Code, Junie
- **Project Status**: Production Ready
- **Issues**: [GitHub Issues](https://github.com/appergb/A.YLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/appergb/A.YLM/discussions)

---

<div align="center">

**Star us on GitHub if you find this project useful!**

*A-YLM: Teaching AI to understand the physical world through geometric constitution*

</div>
