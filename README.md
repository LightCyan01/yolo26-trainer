# YOLO26 Fine-Tuner

A terminal-based interactive tool for fine-tuning YOLO26 models on custom datasets. Navigate training, validation, and hyperparameter configuration entirely through keyboard-driven menus — no config files needed.

## Features

- **Train** — configure epochs, batch size, optimizer, LR scheduler, and more through interactive menus
- **Validate** — run validation with a custom trained or official model; all settings optional (model remembers training config)
- **Hyperparameter tuning** — manual overrides or automatic tuning via `model.tune()`
- **Augmentation settings** — full control over mosaic, mixup, flips, HSV shifts, and more
- **Dataset support** — any dataset with a YOLO-format `.yaml` file; point the menu at it and go

## Requirements

- Python 3.14+
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

```bash
git clone https://github.com/LightCyan01/yolo26-trainer.git
cd yolo26-trainer
uv sync
```

Official model weights (e.g. `yolo26n.pt`) are downloaded automatically by Ultralytics on first use.

## Usage

```bash
uv run main.py
```

Use arrow keys to navigate, Enter to select. The main menu offers:

- **Train** — select task type → pick an official model → set dataset YAML → configure settings → start
- **Validate** — select task type → pick official or custom trained model → optionally override val settings → start

## Project Structure

```
src/
  menu/         # Interactive menus (train, val)
  training/     # run_train(), run_val() wrappers
  utils/        # Settings dataclasses, validators, file dialogs, model lists
dataset/
  NEU-DET-YOLO/ # Example: NEU surface defect dataset (YOLO format)
  MVTecAD/      # Example: MVTec anomaly detection dataset
models/         # Auto-downloaded weights are cached here
```

## Tech Stack

- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)
- [questionary](https://github.com/tmbo/questionary) — terminal menus
- [PyTorch](https://pytorch.org/) + CUDA 12.8
