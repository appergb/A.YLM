# A.YLM

Single-image 3D reconstruction and intelligent navigation system based on Apple SHARP model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Single-Image 3D Reconstruction**: Convert a single RGB image into a 3D Gaussian Splatting model using Apple's SHARP Vision Transformer
- **Semantic Fusion**: YOLO-based object detection with 3D point cloud semantic labeling
- **Obstacle Detection**: Automatic identification of vehicles, pedestrians, bicycles, and other obstacles
- **Navigation Mesh Generation**: Generate robot-navigable 3D voxel meshes with obstacle information
- **Pipeline Processing**: Parallel processing for multiple images (inference + voxelization)
- **Video Processing**: Extract frames from video and process as image sequence
- **Ground Detection**: RANSAC-based ground plane detection with normal vector validation
- **Coordinate Transformation**: OpenCV to Robot/ENU coordinate system conversion
- **Multi-Format Support**: 68+ image formats including HEIC, JPEG, PNG, WEBP, AVIF, TIFF, PSD

## System Requirements

- Python 3.9+ (Python 3.11 recommended for Open3D compatibility)
- PyTorch 2.0+
- 4GB+ RAM
- GPU recommended (supports CUDA, MPS for Apple Silicon)

### Optional Dependencies

| Feature | Dependency | Installation |
|---------|------------|--------------|
| Fast voxelization | Open3D | `pip install open3d` |
| Object detection | Ultralytics YOLO | `pip install ultralytics` |
| Video processing | OpenCV | `pip install opencv-python` |

## Installation

### Quick Start

```bash
# Clone repository with submodules
git clone --recursive https://github.com/appergb/A.YLM.git
cd A.YLM

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv aylm_env
source aylm_env/bin/activate

# Install dependencies
pip install -e .
pip install -e ml-sharp/

# Install with all features (Open3D + YOLO)
pip install -e ".[full]"
```

### YOLO Model Setup

For semantic fusion and obstacle detection, download YOLO model:

```bash
# Auto-download on first use, or manually:
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

### One-Click Run

```bash
# Run complete workflow: setup + predict + voxelize
./run.sh

# Or with custom input
./run.sh --input /path/to/images

# Force pipeline mode for multiple images
./run.sh --pipeline
```

### CLI Commands

```bash
# Environment setup and model download
aylm setup --download

# Run SHARP prediction (image to 3D Gaussian)
aylm predict -i inputs/input_images -o outputs/output_gaussians -v

# Run voxelization (3D Gaussian to voxel grid)
aylm voxelize -i outputs/output_gaussians --voxel-size 0.005

# Run complete pipeline (sequential)
aylm process -v

# Run parallel pipeline (for multiple images)
aylm pipeline -i inputs/input_images -o outputs/output_gaussians -v
```

### Video Processing

```bash
# Extract frames from video
aylm video extract -i video.mp4 -o frames/ --interval 1.0

# Process video (extract + inference + voxelize)
aylm video process -i video.mp4 -o output/ --use-gpu

# Play voxel sequence
aylm video play -i voxels/ --fps 10 --loop
```

### Python API

```python
from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig
from aylm.tools.semantic_fusion import SemanticFusion
from aylm.tools.object_detector import ObjectDetector

# Basic voxelization
config = VoxelizerConfig(voxel_size=0.005)  # 5mm voxels
processor = PointCloudVoxelizer(config=config)
processor.process(
    input_path="output.ply",
    output_path="voxelized.ply",
    remove_ground=True
)

# Semantic fusion with obstacle detection
detector = ObjectDetector()
detections = detector.detect("image.png")

fusion = SemanticFusion()
semantic_pc = fusion.fuse_semantics(
    ply_path="gaussians.ply",
    image_path="image.png",
    detections=detections
)

# Save navigation mesh (5cm voxels)
fusion.save_navigation_ply(semantic_pc, "navigation.ply", voxel_size=0.05)
```

### Pipeline Processing

```python
from aylm.tools.pipeline_processor import PipelineProcessor, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    verbose=True,
    enable_semantic_fusion=True,
    output_navigation_ply=True
)

# Process multiple images in parallel
processor = PipelineProcessor(config)
results = processor.run_pipeline(
    input_dir="inputs/input_images",
    output_dir="outputs"
)
```

## Project Structure

```
A.YLM/
├── src/aylm/                    # Main package
│   ├── cli.py                   # Command line interface
│   └── tools/                   # Processing tools
│       ├── pointcloud_voxelizer.py  # Voxelization module
│       ├── pipeline_processor.py    # Parallel pipeline
│       ├── video_pipeline.py        # Video processing
│       ├── semantic_fusion.py       # Semantic labeling
│       ├── object_detector.py       # YOLO detection
│       ├── obstacle_marker.py       # Obstacle marking
│       ├── pointcloud_slicer.py     # Point cloud slicing
│       └── coordinate_utils.py      # Coordinate transforms
├── ml-sharp/                    # Apple SHARP model (submodule)
├── models/                      # Model checkpoints
├── inputs/
│   ├── input_images/            # Input images
│   └── videos/                  # Input videos
├── outputs/
│   ├── output_gaussians/        # 3D Gaussian PLY files
│   ├── voxelized/               # Voxelized point clouds
│   ├── detections/              # Detection results (JSON)
│   └── navigation/              # Navigation meshes
├── tests/                       # Test suite
├── run.sh                       # One-click run script
└── pyproject.toml               # Project configuration
```

## Output Files

| File | Description |
|------|-------------|
| `*.ply` | 3D Gaussian Splatting model from SHARP |
| `vox_*.ply` | Voxelized point cloud with semantic colors |
| `vox_*_obstacles.json` | Obstacle detection results |
| `nav_*.ply` | Navigation mesh (5cm solid cubes) |

### Obstacle JSON Format

```json
{
  "coordinate_systems": {
    "cv": {"axes": "X右, Y下, Z前"},
    "robot": {"axes": "X前, Y左, Z上"}
  },
  "obstacles": [
    {
      "type": "可运动障碍物",
      "category": "车辆",
      "center_cv": [10.32, 0.06, 0.21],
      "center_robot": [0.21, -10.32, -0.06],
      "dimensions_cv": [7.52, 3.36, 2.26],
      "confidence": 0.93
    }
  ]
}
```

## Configuration

### Environment Variables

```bash
export AYLM_ROOT="/path/to/project"      # Project root directory
export INPUT_DIR="/path/to/images"        # Input images directory
export OUTPUT_DIR="/path/to/output"       # Output directory
```

### Voxelizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_size` | 0.005 | Voxel size in meters (5mm) |
| `statistical_nb_neighbors` | 20 | Neighbors for outlier removal |
| `ransac_distance_threshold` | 0.02 | RANSAC plane distance threshold |
| `ground_normal_threshold` | 0.8 | Ground normal Y-component threshold |

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_VOXEL_SIZE` | 0.05 | Navigation voxel size (5cm) |
| `DEFAULT_SLICE_RADIUS` | 20.0 | Point cloud slice radius (meters) |
| `DEFAULT_FOV_DEGREES` | 60.0 | Camera field of view |
| `DEFAULT_DENSITY_THRESHOLD` | 3 | Minimum points per voxel |

### Video Config (YAML)

```yaml
# inputs/videos/video_config.yaml
frame_extraction_method: interval  # interval/uniform/keyframe
frame_interval: 1.0                # seconds
gpu_acceleration: auto             # auto/cuda/mps/none
output_format: png
```

## Semantic Labels

| Label | Color | Description |
|-------|-------|-------------|
| GROUND | Brown | Ground plane (removed for navigation) |
| VEHICLE | Blue | Cars, trucks, buses |
| PEDESTRIAN | Red | People |
| BICYCLE | Cyan | Bicycles, motorcycles |
| BUILDING | Gray | Static structures |
| VEGETATION | Green | Trees, plants |
| UNKNOWN | White | Unclassified points |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/unit/test_semantic_fusion.py -v

# Run with coverage
pytest --cov=aylm --cov-report=html
```

### Code Quality

```bash
# Format code
black src/aylm tests
isort src/aylm tests

# Linting
ruff check src/aylm tests

# Type checking
mypy src/aylm
```

## Troubleshooting

### Common Issues

1. **Open3D Installation Failed**
   - Use Python 3.11 (Open3D may not support Python 3.13)
   - The system will automatically fall back to numpy implementation

2. **YOLO Model Not Found**
   ```bash
   pip install ultralytics
   # Model auto-downloads on first use (~6MB)
   ```

3. **Model Download Failed**
   - Check network connection
   - Model URL: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
   - Model size: ~2.8GB

4. **Out of Memory**
   - Reduce input image resolution
   - Increase voxel size (e.g., 0.01 for 1cm voxels)
   - Use `--slice-radius` to limit processing area

5. **CUDA/MPS Issues**
   ```bash
   # Check PyTorch device availability
   python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Apple Inc.** - SHARP model and research
- **Ultralytics** - YOLO object detection
- **Open3D Community** - Point cloud processing library
- **PyTorch Team** - Deep learning framework

## Author

**TRIP** (appergb)

**Contributors**: claude code,junie

---

**Project Status**: Production Ready
