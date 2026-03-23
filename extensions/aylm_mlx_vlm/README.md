# A-YLM MLX-VLM Training Demo (Optional Module)

This module is an **optional extension** for A-YLM. It runs the offline navigation
demo with the MLX-VLM provider and exports **training signals** in JSONL format.

## Install (Optional)

```bash
cd /Users/lvbaiqing/TRUE\ 开发/03_Python项目/A.YLM-v2-d/extensions/aylm_mlx_vlm
pip install -e .
```

## Usage

```bash
# 1) Produce video artifacts with the base pipeline
aylm video process -i demo.mp4 -o outputs/video_demo --use-gpu --ego-speed 1.0

# 2) Run MLX-VLM demo + training signal export
aylm-mlx-train -i outputs/video_demo -o outputs/nav_demo
```

Outputs:
- `commands.jsonl` from the base navigation demo
- `training_signals.jsonl` exported by this module (one signal per line)

## Notes
- This module depends on `mlx-vlm` and is meant to be installed only when needed.
- It uses the base `aylm.navigation_demo` runner and extracts training signals from
  `proposal_evaluation.evaluation_detail.training_signal`.
