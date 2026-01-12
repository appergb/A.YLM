# A.YLM

Single-image 3D reconstruction and intelligent navigation system based on Apple SHARP model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- Single-image 3D reconstruction using Vision Transformer
- Intelligent voxelization with 1cm precision for navigation
- Support for 95+ image formats including professional RAW
- GPU acceleration with real-time processing (<1s inference)
- Ground detection, coordinate transformation and path planning

## Requirements

- Python 3.9+
- PyTorch 2.8.0+
- Open3D 0.18.0+
- 4GB+ RAM (GPU recommended)

## Installation

```bash
git clone https://github.com/appergb/A.YLM.git
cd A.YLM

pip install -r requirements.txt
pip install -e ml-sharp/
```

## Usage

```bash
# Run complete pipeline
./run_sharp.sh

# Or run individual steps
./run_sharp.sh --setup      # Environment check
./run_sharp.sh --predict    # 3D reconstruction
./run_sharp.sh --voxelize   # Voxelization
```

## Model Preloading (Recommended)

```bash
# Background model preloading
python3 scripts/preload_sharp_model.py --background

# Check preload status
python3 scripts/preload_sharp_model.py --status
```

## Output Files

- `*.ply`: 3D Gaussian splatting model
- `cropped_*.ply`: Local region cropping results
- `voxelized_*.ply`: 1cm voxel grid (navigation ready)

## Configuration

### Environment Variables

```bash
export INPUT_DIR="/path/to/images"
export OUTPUT_DIR="/path/to/output"
export SHARP_MODEL_PATH="/path/to/model.pt"
```

### Custom Parameters

```bash
# Adjust voxel size and range
python3 scripts/pointcloud_voxelizer.py input.ply --voxel-size 0.01 --range 10.0

# Enable visualization
python3 scripts/pointcloud_voxelizer.py input.ply --visualize
```

## Troubleshooting

1. **Python version error**

   ```bash
   python3 --version  # Ensure >= 3.9
   ```

2. **Dependency installation failure**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory insufficient**
   - Close other applications
   - Use `--voxel-size 0.02` for lower precision

4. **GPU unavailable**

   ```bash
   python3 scripts/preload_sharp_model.py --device cpu
   ```

## Project Structure

```text
A.YLM/
├── scripts/                    # Python scripts
│   ├── preload_sharp_model.py  # Model preloading
│   ├── pointcloud_voxelizer.py # Voxelization
│   └── coordinate_utils.py     # Coordinate utilities
├── ml-sharp/                   # SHARP model code
├── src/aylm/                  # Main package
├── inputs/                     # Input images
├── outputs/                    # Output results
├── models/                     # Model weights
└── run_sharp.sh               # Main script
```

## Contributing

Issues and pull requests are welcome.

**Developer**: TRIP (appergb)
**Contributors**: closer, true

## License

This project is licensed under the MIT License.
