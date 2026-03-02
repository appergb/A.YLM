# A-YLM: Geometric Constitutional AI for Embodied Intelligence

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11/3.12](https://img.shields.io/badge/python-3.11%2F3.12-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org)

**Self-Supervised Safety Framework | Autonomous Driving | Robotics | Embodied AI**

*Extending Anthropic's Constitutional AI paradigm from language to the physical world*

[Paper](#) | [Documentation](#) | [Demo](#) | [中文文档](#)

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

The system pipeline: **Sensor Input → 3DGS Reconstruction → Voxelization → Semantic Fusion → Object Tracking → Motion Estimation → Constitutional Validation → Training Signal → Self-Learning**.

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

### Our Solution: Bidirectional Fusion with E2E Systems

A-YLM introduces **Geometric Constitutional AI** with **bidirectional fusion** — fully compatible with existing end-to-end driving systems (Tesla FSD, Huawei ADS, etc.):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Bidirectional Fusion Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    End-to-End Driving AI (FSD/ADS)                   │   │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │   │
│   │   │  Video   │───▶│  Neural  │───▶│ Decision │───▶│  Control │     │   │
│   │   │  Input   │    │  Network │    │  Output  │    │  Signal  │     │   │
│   │   └──────────┘    └────┬─────┘    └────┬─────┘    └──────────┘     │   │
│   │                        │               │                            │   │
│   │                        │ 3D Input      │ Decision                   │   │
│   │                        ▼               ▼                            │   │
│   └────────────────────────┼───────────────┼────────────────────────────┘   │
│                            │               │                                │
│   ┌────────────────────────┼───────────────┼────────────────────────────┐   │
│   │                    A-YLM Edge Module (Lightweight)                   │   │
│   │                        │               │                            │   │
│   │   ┌──────────┐    ┌────▼─────┐    ┌────▼─────┐    ┌──────────┐     │   │
│   │   │  Camera  │───▶│  3D GS   │───▶│  Safety  │───▶│ Training │     │   │
│   │   │  Input   │    │  Recon   │    │  Scoring │    │  Signal  │     │   │
│   │   └──────────┘    └────┬─────┘    └──────────┘    └──────────┘     │   │
│   │                        │                                            │   │
│   │                        │ Point Cloud + Color                        │   │
│   │                        ▼                                            │   │
│   │               ┌──────────────┐                                      │   │
│   │               │  3D Scene    │ ──▶ Available for E2E AI Input       │   │
│   │               │  with Color  │                                      │   │
│   │               └──────────────┘                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Direction 1: Safety Supervision (A-YLM → E2E AI)**
- A-YLM monitors and validates E2E AI decisions in real-time
- Provides safety scores and violation labels
- Generates training signals for AI self-improvement (no human labeling)

**Direction 2: 3D Input Enhancement (A-YLM → E2E AI)**
- A-YLM provides 3D point cloud as additional input to E2E AI
- Includes color information from 3D Gaussian Splatting
- Enables E2E AI to perceive 3D geometry directly
- Fully compatible with existing video-based E2E architectures

**Key Advantages:**
1. **Full Compatibility**: Works with existing E2E systems (FSD, ADS) without modification
2. **Enhanced Perception**: E2E AI gains 3D geometric understanding
3. **Local Safety Guarantee**: Edge module provides real-time safety validation
4. **Lightweight & Edge-Optimized**: Runs on Jetson, Apple MPS, and other edge devices
5. **Self-Supervised Learning**: AI evolves through geometric feedback, no human annotation needed

### Local Safety Decision Module

A-YLM provides a **standalone local safety decision module** that can be called as an independent component:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    A-YLM Local Safety Decision Module                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Camera Frame                                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │  Camera  │───▶│  3D GS   │───▶│  Point   │───▶│  Depth   │            │
│   │  Frame   │    │  Recon   │    │  Cloud   │    │  Info    │            │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│                                                         │                   │
│                                                         ▼                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Safety Scoring Module (Our Contribution)          │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│   │   │  Collision   │  │     TTC      │  │   Boundary   │              │  │
│   │   │   Detection  │  │  Calculation │  │  Validation  │              │  │
│   │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │  │
│   │          │                 │                 │                       │  │
│   │          └─────────────────┼─────────────────┘                       │  │
│   │                            ▼                                         │  │
│   │                   ┌──────────────┐                                   │  │
│   │                   │ Safety Score │ ──▶ 0.0 (Dangerous) ~ 1.0 (Safe) │  │
│   │                   └──────────────┘                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                         │                   │
│   Output:                                               ▼                   │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │  - Safety Score (0.0 ~ 1.0)                                          │ │
│   │  - Violation Labels (collision, ttc_warning, boundary)               │ │
│   │  - 3D Obstacle Positions with Depth                                  │ │
│   │  - Recommended Action (safe/warning/emergency_stop)                  │ │
│   │  - Training Signal for Cloud AI                                      │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Module Features:**
- **3D Decision Making**: With point cloud and depth information, enables 3D-based safety decisions locally
- **Safety Scoring**: Our proposed scoring module evaluates AI decisions against geometric constraints
- **Modular Design**: Can be called as an independent module by any system
- **Real-time**: Optimized for edge deployment with <50ms latency
- **API-Ready**: Simple Python API for integration

```python
# Example 1: CommandValidator — 验证外部指令安全性
from aylm.constitution import CommandValidator

validator = CommandValidator()

# 验证 JSON 轨迹
result = validator.validate(
    command={"type": "trajectory", "points": [[5, 0, 0, 0.5]]},
    ego_speed=10.0,
    obstacles=[{"center_robot": [5, 0, 0], "dimensions_robot": [1, 1, 1],
                "_label": "PERSON", "confidence": 0.9}],
)
print(f"Approved: {result.approved}")        # True/False
print(f"Safety Score: {result.safety_score}") # 0.0 ~ 1.0
print(f"Action: {result.recommended_action}") # 'emergency_stop'
print(f"Reason: {result.reason}")             # 人类可读原因

# 验证自然语言指令（中/英文）
result = validator.validate(
    command="向左转弯30度",
    ego_speed=10.0,
    obstacles=[...],
)
if not result.approved:
    print(f"否决: {result.reason}")
    print(f"安全替代: {result.alternative_decision}")

# Example 2: ConstitutionSession — 有状态的多帧时序评估
from aylm.api import ConstitutionSession

session = ConstitutionSession(ego_speed=10.0)

# 单帧评估
result = session.evaluate(obstacles=[...])
print(f"Safety: {result['safety_score']}, Trend: {result['trend']}")

# 动态修改自车速度
session.update_ego(speed=15.0, heading=0.1)

# 批量时序评估
results = session.evaluate_batch([
    {"obstacles": [...], "ego_speed": 10.0, "timestamp": 0.0},
    {"obstacles": [...], "ego_speed": 12.0, "timestamp": 0.5},
])

# 会话趋势分析
print(session.trend)    # "improving" / "declining" / "stable"
print(session.summary)  # 统计摘要
```

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
| 6 | Object Tracking | Multi-frame Obstacles | Tracked Obstacles (IDs) | Temporal Association |
| 7 | Motion Estimation | Tracked Positions | Velocity / Heading | Dynamic Prediction |
| 8 | Constitution Evaluation | Scene + Decision | Safety Score + Violations | Geometric Safety Verification |
| 9 | Feedback Generation | Validation Results | Training Signals | Self-Evolution |

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
| Voxel Grid (voxel representation) | ~50K voxels | O(n) lookup |
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
| Python | 3.11/3.12 (for `run.sh`) | Runtime |
| PyTorch | 2.0+ | Deep Learning Framework |
| Open3D | 0.17+ | Point Cloud Processing |
| Ultralytics | 8.0+ | YOLO Object Detection |
| NumPy | 1.24+ | Numerical Computing |
| SciPy | 1.10+ | Hungarian Algorithm (Tracking) |
| FastAPI | 0.100+ (optional) | HTTP/WebSocket API Server |
| uvicorn | 0.20+ (optional) | ASGI Server |

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

# Install API server (FastAPI + uvicorn) for HTTP/WebSocket interface
pip install -e ".[api]"

# Install development tools (pytest, black, ruff, etc.)
pip install -e ".[dev]"
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

Before running `./run.sh`, ensure:

- Clone with submodules (`ml-sharp` is required):
  - `git clone --recursive https://github.com/appergb/A.YLM.git`
  - or `git submodule update --init --recursive`
- Python 3.11 or 3.12 is available (`run.sh` enforces this range)
- First-time dependency/model install can access network
- Default auto mode needs real files in `inputs/input_images` or `inputs/videos`

```bash
# Environment preflight only
./run.sh --check-only

# Constitution demo (no image/video required)
./run.sh --demo

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

# Stage 5: Constitution Evaluation Demo (独立宪法评估演示)
aylm demo --ego-speed 10.0

# Stage 6: Constitution API Server (宪法评估 HTTP/WebSocket 服务)
aylm serve --port 8000 --ego-speed 10.0
```

### 6.3 Video Processing

```bash
# Frame extraction
aylm video extract -i video.mp4 -o frames/ --interval 1.0

# Full video pipeline (extract + inference + voxelize + tracking + constitution)
aylm video process -i video.mp4 -o output/ --use-gpu --ego-speed 10.0

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

### 6.5 Constitution Module API

A-YLM provides a **plugin-based Constitutional AI module** with 5 built-in safety principles. External systems (LLM, end-to-end models, ROS) can validate their decisions against geometric safety constraints.

#### 6.5.1 CommandValidator — 指令安全验证

```python
from aylm.constitution import CommandValidator

validator = CommandValidator()

# ── JSON 轨迹验证 ──
result = validator.validate(
    command={"type": "trajectory", "points": [[5, 0, 0, 0.5]]},
    ego_speed=10.0,
    obstacles=[{
        "center_robot": [5, 0, 0],
        "dimensions_robot": [1, 1, 1],
        "_label": "PERSON",
        "confidence": 0.9,
    }],
)
print(result.approved)           # False (collision risk)
print(result.safety_score)       # 0.61
print(result.recommended_action) # "emergency_stop"
print(result.reason)             # 人类可读原因
print(result.alternative_decision)  # 安全替代方案

# ── 控制指令验证 ──
result = validator.validate(
    command={"type": "control", "steering": 0.1, "throttle": 0.5, "brake": 0.0},
    ego_speed=10.0,
)

# ── 航点验证 ──
result = validator.validate(
    command={"type": "waypoint", "target": [10, 3, 0], "speed": 5.0},
    ego_speed=10.0,
)

# ── 自然语言验证（中文 + 英文） ──
result = validator.validate(command="向左转弯30度", ego_speed=10.0, obstacles=[...])
result = validator.validate(command="emergency brake", ego_speed=15.0)
result = validator.validate(command="加速到60km/h", ego_speed=10.0)
result = validator.validate(command="change lane right", ego_speed=12.0)
```

**Supported Command Types:**

| Type | Format | Examples |
|------|--------|---------|
| **Trajectory** | `{"type": "trajectory", "points": [[x,y,z,t], ...]}` | 轨迹点序列 |
| **Control** | `{"type": "control", "steering": 0.1, "throttle": 0.5}` | 控制信号 |
| **Waypoint** | `{"type": "waypoint", "target": [x,y,z], "speed": 5.0}` | 目标点 |
| **Natural Language** | `"向左转弯"` / `"emergency brake"` / `"加速到60"` | 中英文自然语言 |

#### 6.5.2 ConstitutionSession — 有状态时序评估

```python
from aylm.api import ConstitutionSession

session = ConstitutionSession(ego_speed=10.0)

# 单帧评估
result = session.evaluate(
    obstacles=[{"center_robot": [5,0,0], "dimensions_robot": [1,1,1],
                "_label": "VEHICLE", "confidence": 0.9}],
)

# 动态修改速度/航向
session.update_ego(speed=15.0, heading=0.1)

# 带指令的评估
result = session.evaluate(
    command={"type": "trajectory", "points": [[3,0,0,0.5]]},
    obstacles=[...],
)

# 批量时序评估
results = session.evaluate_batch([
    {"obstacles": [...], "ego_speed": 10.0, "timestamp": 0.0},
    {"obstacles": [...], "ego_speed": 12.0, "timestamp": 0.5},
])

# 安全趋势分析
print(session.trend)    # "improving" / "declining" / "stable" / "unknown"
print(session.summary)  # {"avg_score": 0.7, "approval_rate": 0.8, ...}
```

#### 6.5.3 Custom Principle Plugin — 自定义宪法原则

```python
from aylm.constitution import ConstitutionPrinciple, ViolationResult, Severity
from aylm.constitution import ConstitutionRegistry

@ConstitutionRegistry.register_principle("my_custom_rule")
class MyCustomPrinciple(ConstitutionPrinciple):
    """自定义安全规则。"""

    @property
    def name(self) -> str:
        return "my_custom_rule"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    def evaluate(self, scene, decision) -> ViolationResult:
        # 实现您的安全检查逻辑
        return ViolationResult(violated=False, severity=self.severity, ...)
```

#### 6.5.4 Built-in Safety Principles

| Principle | Description | Severity |
|-----------|-------------|----------|
| `NoCollisionPrinciple` | 碰撞检测：检查轨迹点与障碍物的最小距离 | CRITICAL |
| `SafeFollowingPrinciple` | 安全跟车距离：基于速度的动态跟车距离验证 | HIGH |
| `TTCSafetyPrinciple` | TTC (Time-to-Collision) 安全：碰撞时间估算 | HIGH |
| `LaneCompliancePrinciple` | 车道合规：车道偏移检测 | MEDIUM |
| `SpeedLimitPrinciple` | 速度限制：超速检测 | MEDIUM |

### 6.6 HTTP API Server

Install API dependencies:

```bash
pip install -e ".[api]"   # or: pip install fastapi uvicorn
```

Start the server:

```bash
aylm serve --port 8000 --ego-speed 10.0
# API docs: http://localhost:8000/docs
```

#### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/evaluate` | 单帧安全评估 |
| `POST` | `/api/v1/evaluate/batch` | 批量时序评估 |
| `PUT` | `/api/v1/ego` | 动态修改自车速度/航向 |
| `GET` | `/api/v1/summary` | 会话统计摘要 |
| `GET` | `/api/v1/config` | 查看当前配置 |
| `WS` | `/api/v1/session` | WebSocket 实时流式评估 |

#### Example: Single-frame Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "obstacles": [
      {"center_robot": [5,0,0], "dimensions_robot": [1,1,1],
       "_label": "PERSON", "confidence": 0.9}
    ],
    "command": "向左转弯",
    "ego_speed": 10.0
  }'
```

#### Example: Dynamic Speed Update

```bash
curl -X PUT http://localhost:8000/api/v1/ego \
  -H "Content-Type: application/json" \
  -d '{"speed": 15.0, "heading": 0.1}'
```

### 6.7 WebSocket Real-time Streaming

```python
import asyncio
import json
import websockets

async def realtime_evaluation():
    async with websockets.connect("ws://localhost:8000/api/v1/session") as ws:
        # 发送评估帧
        await ws.send(json.dumps({
            "obstacles": [{"center_robot": [5,0,0], "dimensions_robot": [1,1,1],
                           "_label": "VEHICLE", "confidence": 0.9}],
            "ego_speed": 10.0,
        }))
        result = json.loads(await ws.recv())
        print(f"Score: {result['safety_score']}, Approved: {result['approved']}")

        # 动态修改速度
        await ws.send(json.dumps({"action": "update_ego", "speed": 15.0}))
        ack = json.loads(await ws.recv())

        # 获取会话摘要
        await ws.send(json.dumps({"action": "summary"}))
        summary = json.loads(await ws.recv())

        # 重置会话
        await ws.send(json.dumps({"action": "reset"}))

asyncio.run(realtime_evaluation())
```

---

## 7. Output Specification

### 7.1 File Formats

| File Pattern | Format | Description |
|--------------|--------|-------------|
| `*.ply` | PLY | Raw 3DGS point cloud from SHARP |
| `vox_*.ply` | PLY | Voxelized point cloud with semantic colors |
| `nav_*.ply` | PLY | Safety validation mesh (voxel cubes) |
| `*_obstacles.json` | JSON | Structured obstacle data for safety validation |

### 7.2 Safety Validation JSON Schema

```json
{
  "frame_id": 2,
  "timestamp": 1.984,
  "coordinate_systems": {
    "cv": {"description": "OpenCV/相机坐标系", "axes": "X右, Y下, Z前"},
    "robot": {"description": "机器人/ROS坐标系", "axes": "X前, Y左, Z上"},
    "transform": "X_robot=Z_cv, Y_robot=-X_cv, Z_robot=-Y_cv"
  },
  "total_count": 2,
  "movable_count": 2,
  "static_count": 0,
  "tracked_count": 2,
  "obstacles": [
    {
      "type": "可运动障碍物",
      "category": "行人",
      "movable": true,
      "center_cv": [x, y, z],
      "center_robot": [x, y, z],
      "dimensions_cv": [w, h, d],
      "dimensions_robot": [w, h, d],
      "confidence": 0.875,
      "point_count": 56521,
      "_label": "PERSON",
      "track_id": 1,
      "track_age": 3,
      "track_hits": 3
    }
  ],
  "constitution_evaluation": {
    "safety_score": {
      "overall": 0.6087,
      "scores": {"collision": 0.1, "ttc": 1.0, "boundary": 1.0},
      "violations": ["no_collision"],
      "recommended_action": "emergency_stop",
      "confidence": 1.0
    },
    "violations": [
      {
        "violated": true,
        "severity": "CRITICAL",
        "description": "检测到碰撞风险，最小距离 -0.95m",
        "metrics": {"min_distance": -0.95, "safety_margin": 0.5},
        "correction_hint": {"action": "avoid", "obstacle_position": [x, y, z]}
      }
    ],
    "principle_results": {
      "no_collision": {"violated": true, "severity": "CRITICAL", "...": "..."},
      "safe_following": {"violated": false, "severity": "HIGH", "...": "..."},
      "ttc_safety": {"violated": false, "severity": "HIGH", "...": "..."},
      "lane_compliance": {"violated": false, "severity": "MEDIUM", "...": "..."},
      "speed_limit": {"violated": false, "severity": "MEDIUM", "...": "..."}
    },
    "training_signal": {
      "signal_type": "correction",
      "scene_context": {"ego_state": {"speed": 10.0}, "obstacles": ["..."]},
      "ai_decision": {"trajectory": ["..."], "target_speed": 10.0},
      "correction_target": {"corrections": [{"action": "avoid"}]}
    }
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
| `DEFAULT_VOXEL_SIZE` | 0.05 | Safety validation voxel size (voxel) |
| `DEFAULT_SLICE_RADIUS` | 20.0 | Processing radius (meters) |
| `DEFAULT_FOV_DEGREES` | 60.0 | Camera field of view |
| `DEFAULT_DENSITY_THRESHOLD` | 3 | Minimum points per voxel |
| `TTC_WARNING_THRESHOLD` | 3.0 | Time-to-collision warning (seconds) |
| `TTC_CRITICAL_THRESHOLD` | 1.5 | Time-to-collision critical (seconds) |

### 8.4 Constitution Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approval_threshold` | 0.6 | Safety score threshold for approving decisions |
| `collision_safety_margin` | 0.5 | Minimum safe distance to obstacles (meters) |
| `following_time_headway` | 2.0 | Minimum following time headway (seconds) |
| `speed_limit_default` | 13.89 | Default speed limit (m/s, ~50 km/h) |
| `lane_max_offset` | 1.75 | Maximum lane deviation (meters) |

### 8.5 Constitution Config File (YAML)

Constitution evaluation can be configured via YAML or JSON:

```yaml
# configs/constitution_example.yaml
principles:
  no_collision:
    enabled: true
    params:
      safety_margin: 0.5
  safe_following:
    enabled: true
    params:
      time_headway: 2.0
  ttc_safety:
    enabled: true
    params:
      warning_threshold: 3.0
      critical_threshold: 1.5
  lane_compliance:
    enabled: true
  speed_limit:
    enabled: true
    params:
      default_limit_kmh: 50

scorer:
  type: weighted
  weights:
    collision: 1.0
    following: 0.8
    ttc: 0.8
    lane: 0.5
    speed: 0.5
```

---

## 9. Project Structure

```
A.YLM/
├── src/aylm/                           # Core Package
│   ├── cli.py                          # CLI (setup, predict, voxelize, process, pipeline, video, demo, serve)
│   ├── api/                            # External API Module
│   │   ├── session.py                  # ConstitutionSession (stateful evaluation)
│   │   └── app.py                      # FastAPI HTTP + WebSocket server
│   ├── constitution/                   # Constitutional AI Module
│   │   ├── base.py                     # ConstitutionPrinciple abstract base
│   │   ├── evaluator.py               # ConstitutionEvaluator orchestrator
│   │   ├── validator.py               # CommandValidator (top-level API)
│   │   ├── command_parser.py           # CommandParser + JSON/NL parsers
│   │   ├── types.py                    # SceneState, AIDecision, EgoState
│   │   ├── config.py                   # ConstitutionConfig (YAML/JSON)
│   │   ├── registry.py                # ConstitutionRegistry plugin system
│   │   ├── scorer.py                   # SafetyScorer abstract base
│   │   ├── weighted_scorer.py          # WeightedSafetyScorer implementation
│   │   ├── training.py                # TrainingSignalGenerator abstract base
│   │   ├── default_generator.py        # DefaultTrainingSignalGenerator
│   │   ├── adapter.py                  # ConstitutionObstacle adapter
│   │   └── principles/                 # Built-in safety principles
│   │       ├── collision.py            # NoCollisionPrinciple
│   │       ├── ttc.py                  # TTCSafetyPrinciple
│   │       ├── following.py            # SafeFollowingPrinciple
│   │       ├── lane.py                 # LaneCompliancePrinciple
│   │       └── speed.py               # SpeedLimitPrinciple
│   └── tools/                          # Processing Modules
│       ├── pipeline_processor.py       # Parallel Pipeline Orchestration
│       ├── video_pipeline.py           # Video Sequence Processing + Tracking
│       ├── pointcloud_voxelizer.py     # Occupancy 2.0 Voxelization
│       ├── semantic_fusion.py          # 2D→3D Semantic Projection
│       ├── object_detector.py          # YOLO Instance Segmentation
│       ├── obstacle_marker.py          # DBSCAN Obstacle Clustering
│       ├── object_tracker.py           # ByteTrack Multi-Object Tracking
│       ├── motion_estimator.py         # Kalman-filtered Motion Estimation
│       ├── pointcloud_slicer.py        # Spatial ROI Extraction
│       ├── coordinate_utils.py         # CV↔Robot Coordinate Transform
│       ├── json_utils.py               # numpy-safe JSON serialization
│       ├── constitution_integration.py # Constitution ↔ Pipeline bridge
│       └── constitution_demo.py        # Interactive constitution demo
├── ml-sharp/                           # Apple SHARP Model (submodule)
├── local_only/                         # Local-only workspace (skeleton tracked, content ignored)
├── models/                             # Model Checkpoints
├── configs/                            # Configuration Files
│   └── constitution_example.yaml       # Example constitution config
├── inputs/                             # Input Data
│   ├── input_images/                   # RGB Images
│   └── videos/                         # Video Files
├── outputs/                            # Output Data
│   ├── output_gaussians/               # 3DGS Point Clouds
│   ├── voxelized/                      # Occupancy Grids + Obstacle JSONs
│   ├── detections/                     # Detection Results
│   └── video_output/                   # Video Pipeline Output
├── tests/                              # Test Suite (378+ tests)
├── run.sh                              # One-Click Execution Script
└── pyproject.toml                      # Project Configuration
```

### 9.1 Local-only workspace convention

Use `local_only/` for files that should stay local and not be uploaded:

- Research drafts and personal notes
- Local test datasets and media
- Temporary benchmark scripts/results
- Local run outputs and archived artifacts

Current policy:
- Track directory skeleton (`README_LOCAL.md`, `.gitkeep`) for team consistency
- Ignore actual private content via `.gitignore`

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

### 11.3 CI Checks

Current CI includes:

- Unit tests (`ubuntu-latest`, `macos-latest`, Python 3.11)
- Lint and format checks (`black`, `isort`, `ruff`, `mypy`)
- `run.sh` smoke checks on clean checkout:
  - `./run.sh --check-only`
  - `./run.sh --demo`
  - Python matrix: 3.11 / 3.12

---

## 12. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `pip install -e ml-sharp/` failed in first run | Submodule not initialized | Run `git submodule update --init --recursive` |
| `run.sh` reports unsupported Python version | Using Python outside 3.11/3.12 | Use Python 3.11 or 3.12 and recreate venv |
| `./run.sh` found no image/video in auto mode | No real files in `inputs/input_images` or `inputs/videos` | Add input files, or run `./run.sh --demo` first |
| Open3D installation failed | Python version incompatibility | Use Python 3.11 or 3.12 |
| YOLO model not found | First-time download | Run `pip install ultralytics` |
| SHARP model download failed | Network issue | Manual download from Apple CDN |
| Dependency install fails on first setup | Network / DNS / mirror issue | Check connectivity to GitHub/PyPI and retry |
| Out of memory | Large point cloud | Increase `voxel_size` or use `--slice-radius` |
| CUDA/MPS unavailable | Driver issue | Check `torch.cuda.is_available()` |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Anthropic** - Constitutional AI paradigm that inspired this work
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
