# Repository Guidelines

## Project Structure & Module Organization

This repository implements Sparse Voxels Rasterization using Python, PyTorch, and custom CUDA kernels. Core model and camera code lives in `src/`, dataset readers in `src/dataloader/`, and shared helpers in `src/utils/`. The CUDA extension has Python wrappers in `cuda/svraster_cuda/`, kernels in `cuda/src/`, and bindings in `cuda/binding.cpp`.

Root scripts provide training, rendering, evaluation, mesh extraction, and visualization. Dataset presets live in `cfg/`; benchmark and preprocessing scripts live in `scripts/`. See `articles/` and `notebooks/` for tutorials and `asset/` for documentation images. Keep local datasets in `data/` and generated runs in `output/`; both are ignored by Git.

## Build, Test, and Development Commands

Install PyTorch first with a compatible CUDA toolkit; README-tested combinations include PyTorch 1.13.1/CUDA 11.7 and PyTorch 2.5.0/CUDA 12.4.

- `pip install -r requirements.txt`: install Python dependencies.
- `pip install -e cuda/`: build and install the CUDA extension; rerun after kernel or binding changes.
- `python train.py --eval --source_path "$DATA_PATH" --model_path "$OUTPUT_PATH"`: train with held-out evaluation views.
- `python render.py "$OUTPUT_PATH" --skip_train`: render test views.
- `python eval.py "$OUTPUT_PATH"`: compute image-quality metrics for rendered results.
- `python render.py "$OUTPUT_PATH" --eval_fps`: benchmark rendering speed.
- `python viz.py "$OUTPUT_PATH"`: launch the interactive viewer.

## Coding Style & Naming Conventions

Follow surrounding code: four-space indentation in Python, `snake_case` functions and modules, and `PascalCase` classes. Preserve local C++/CUDA formatting and existing copyright headers. No repository-wide formatter or linter is configured. Define configuration defaults in `src/config.py` and dataset-specific overrides in `cfg/*.yaml`; command-line values override these settings.

## Testing Guidelines

No automated test framework or coverage threshold is configured, and `tests/` currently contains no test sources. Validate changes on a representative scene using training, rendering, and evaluation above. Record the configuration, dataset, GPU, quality metrics, and relevant timing comparisons. For new automated tests, use descriptive `test_*.py` filenames and document their dependencies and invocation.

## Commit & Pull Request Guidelines

History uses short, plain-language subjects such as `Fix input resize bug` and `Concurrent dataset loading`; no strict prefix convention is evident. Keep commits focused. Pull requests should explain the affected workflow, link relevant issues, and report validation commands and results. Include comparable rendered images for visual changes and performance measurements for CUDA optimizations.
