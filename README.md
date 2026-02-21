# A.YLM

Single-image 3D reconstruction and intelligent navigation system based on Apple SHARP model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Single-Image 3D Reconstruction**: Convert a single RGB image into a 3D Gaussian Splatting model using Apple's SHARP Vision Transformer
- **Intelligent Voxelization**: 5mm precision point cloud voxelization for robot navigation
- **Ground Detection**: RANSAC-based ground plane detection and removal
- **Coordinate Transformation**: OpenCV to Robot/ENU coordinate system conversion
- **Multi-Format Support**: 68+ image formats including HEIC, JPEG, PNG, WEBP, AVIF, TIFF, PSD
- **Dual Implementation**: Open3D acceleration with numpy fallback for compatibility

## System Requirements

- Python 3.9+ (Python 3.11 recommended for Open3D compatibility)
- PyTorch 2.0+
- 4GB+ RAM
- GPU recommended (supports CUDA, MPS for Apple Silicon)

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

# Install with Open3D (optional, for faster voxelization)
pip install -e ".[full]"
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
```

### CLI Commands

```bash
# Environment setup and model download
aylm setup --download

# Run SHARP prediction (image to 3D Gaussian)
aylm predict -i inputs/input_images -o outputs/output_gaussians -v

# Run voxelization (3D Gaussian to voxel grid)
aylm voxelize -i outputs/output_gaussians --voxel-size 0.005

# Run complete pipeline
aylm process -v
```

### Python API

```python
from aylm.tools.pointcloud_voxelizer import PointCloudVoxelizer, VoxelizerConfig
from aylm.tools.coordinate_utils import transform_for_navigation

# Configure voxelizer
config = VoxelizerConfig(voxel_size=0.005)  # 5mm voxels
processor = PointCloudVoxelizer(config=config)

# Process point cloud
processor.process(
    input_path="output.ply",
    output_path="voxelized.ply",
    remove_ground=True,
    transform_coords=False
)
```

## Project Structure

```
A.YLM/
├── src/aylm/                    # Main package
│   ├── cli.py                   # Command line interface
│   └── tools/                   # Processing tools
│       ├── pointcloud_voxelizer.py  # Voxelization module
│       └── coordinate_utils.py      # Coordinate transforms
├── ml-sharp/                    # Apple SHARP model (submodule)
├── models/                      # Model checkpoints
├── inputs/input_images/         # Input images
├── outputs/output_gaussians/    # Output PLY files
├── tests/                       # Test suite
├── run.sh                       # One-click run script
└── pyproject.toml               # Project configuration
```

## Output Files

| File | Description |
|------|-------------|
| `*.ply` | 3D Gaussian Splatting model from SHARP |
| `vox_*.ply` | Voxelized point cloud (navigation-ready) |

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
| `statistical_std_ratio` | 2.0 | Standard deviation threshold |
| `ransac_distance_threshold` | 0.02 | RANSAC plane distance threshold |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=aylm --cov-report=html
```

### Code Quality

```bash
# Format code
black src/aylm tests
isort src/aylm tests

# Type checking
mypy src/aylm

# Linting
ruff check src/aylm tests
```

## Troubleshooting

### Common Issues

1. **Open3D Installation Failed**
   - Use Python 3.11 (Open3D may not support Python 3.13)
   - The system will automatically fall back to numpy implementation

2. **Model Download Failed**
   - Check network connection
   - Model URL: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
   - Model size: ~2.8GB

3. **Out of Memory**
   - Reduce input image resolution
   - Increase voxel size (e.g., 0.01 for 1cm voxels)

4. **CUDA/MPS Issues**
   ```bash
   # Check PyTorch device availability
   python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Apple Inc.** - SHARP model and research
- **Open3D Community** - Point cloud processing library
- **PyTorch Team** - Deep learning framework

## Author

**TRIP** (appergb)

**Contributors**: closer, true

---

**Project Status**: Production Ready
