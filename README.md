# AYLM

Advanced YLM for 3D Gaussian Splatting - Image to 3D Gaussian conversion pipeline.

## Installation

```bash
pip install -e .
```

With full dependencies (including Open3D):
```bash
pip install -e ".[full]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

```bash
aylm --help
```

## Project Structure

```
src/aylm/          # Main package
  tools/           # Utility tools
tests/             # Test files
inputs/            # Input images
outputs/           # Output gaussians
models/            # Model weights
```
