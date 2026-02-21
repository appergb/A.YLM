# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A.YLM v2 - Single-image 3D reconstruction system based on Apple SHARP model. Converts a single photo into a 3D Gaussian Splatting model, then voxelizes it for robot navigation.

## Common Commands

```bash
# Activate virtual environment
source aylm_env/bin/activate

# Run complete workflow (setup + predict + voxelize)
./run.sh

# Or use CLI commands
python -m aylm.cli process          # Full workflow
python -m aylm.cli setup --download # Environment setup + download model
python -m aylm.cli predict -v       # SHARP prediction (verbose)
python -m aylm.cli voxelize -v      # Voxelization (verbose)

# Run tests
pytest tests/

# Install in development mode
pip install -e .
pip install -e ml-sharp/
```

## Architecture

```
A.YLM-v2/
├── src/aylm/
│   ├── cli.py                      # CLI entry point (setup/predict/voxelize/process)
│   └── tools/
│       ├── pointcloud_voxelizer.py # Point cloud processing (Open3D + numpy fallback)
│       └── coordinate_utils.py     # OpenCV → robot/ENU coordinate transforms
├── ml-sharp/                       # Apple SHARP model (Git submodule)
├── models/                         # Model checkpoint (sharp_2572gikvuh.pt, 2.8GB)
├── inputs/input_images/            # Input images (HEIC, JPG, PNG, etc.)
├── outputs/output_gaussians/       # Output PLY files
│   └── voxelized/                  # Voxelized PLY files
└── aylm_env/                       # Python 3.11 virtual environment
```

## Key Components

- **SHARP Model**: Vision Transformer for single-image 3D Gaussian prediction. Called via `sharp predict` CLI.
- **PointCloudVoxelizer**: Loads PLY, removes outliers (statistical), detects ground (RANSAC), voxel downsamples, transforms coordinates.
- **VoxelizerConfig**: Controls voxel_size (default 0.005m = 5mm), RANSAC params, outlier removal params.

## Dependencies

- Python 3.11 (required for Open3D compatibility)
- PyTorch with MPS (Apple Silicon) support
- Open3D for point cloud processing (optional, has numpy fallback)
- sharp package from ml-sharp/

## Notes

- Model checkpoint auto-downloads from Apple CDN if missing
- Supports 95+ image formats including HEIC
- Coordinate transform: OpenCV (Y-down, Z-forward) → Robot (Z-up, X-forward)
