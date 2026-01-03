# Project Overview

This repository contains the code, datasets, environments, and experimental outputs for the project. It is organized by purpose to support clarity and reproducibility.

## Contents

- [Directory structure](#directory-structure)
- [Environments](#environments)
- [Data](#data)
- [Models and experiments](#models-and-experiments)
- [Evaluation outputs](#evaluation-outputs)
- [Results (predictions)](#results-predictions)
- [Timing experiments](#timing-experiments)
- [External dependencies](#external-dependencies)
- [Reproducibility notes](#reproducibility-notes)

## Directory structure

```text
Data/
└── ...                      # Datasets and splits

environments/
├── main-env-locked.yml
├── rexomni_env.yml
└── semidetr_env.bash

Evaluations/
├── Experiment_1/
├── Experiment_2/
└── evaluation.ipynb

External/
└── ...

Models/
├── MLLM/
├── SEMI-DETR/
├── VLM/
└── YOLO/

Notebooks/
└── ...

Results/
├── Experiment_1/
└── Experiment_2.1/

src/
└── annotation_methods/
    ├── __init__.py
    ├── budget_splits.py
    ├── coco_utils.py
    ├── io_utils.py
    ├── vlm_helpers.py
    └── yolo_helpers.py

Timing_Experiments/
├── Timing_Experiments_Data/
└── timing_experiments.ipynb
```

## Environments

All Python dependencies are managed with Conda using version-locked environment files.

- `environments/main-env-locked.yml` — primary environment used for most experiments.
- `environments/rexomni_env.yml` — environment used for REX Omni experiments (mirrors the official REX Omni repository).
- `environments/semidetr_env.bash` — helper script to set up the SEMI-DETR environment.

To create the main environment:

```bash
conda env create -f environments/main-env-locked.yml
conda activate <main-env-name>
```

To reproduce REX Omni experiments, create and activate the environment from `rexomni_env.yml` instead.

## Data

The `Data/` directory contains all datasets and predefined train/test splits. A more detailed description of what this directory looks like when data is added is provided in `Data/README.md`.

You must add data yourself; follow the instructions in the `dataset_split.ipynb` notebook under `Notebooks/`.

## Models and experiments

Model families are organized under `Models/`:

- `Models/MLLM/` — multimodal large language model experiments (e.g., `rexomni.ipynb`).
- `Models/SEMI-DETR/` — SEMI-DETR experiments (environment prepared; work in progress).
- `Models/VLM/` — vision–language model experiments (`vlms.ipynb`).
- `Models/YOLO/` — YOLO-based detectors and training:
  - `runs/` — YOLO training runs and checkpoints.
  - `yolo_outputs/` — raw YOLO-style outputs (converted to COCO before being placed in `Results/`).
  - `yolo11m.pt` — example YOLO checkpoint.
  - `yolo.ipynb` — yolo splits/training/inference notebook.

Additional utility notebooks are stored in `Notebooks/`:

- `dataset_split.ipynb` — create train/test splits.
- `dataset_statistics.ipynb` — compute basic dataset statistics.
- `display_predictions.ipynb` — visualize model predictions.

## Evaluation outputs

The `Evaluations/` directory contains JSON log files with evaluation statistics for each experiment:

- `Evaluations/Experiment_1/`
- `Evaluations/Experiment_2/`

Filenames follow the pattern:

```text
{dataset}_{split}_{model-and-repeat}_evaluation.json
```

For example:

```text
apples_test_gd_b_evaluation.json
```

Use `Evaluations/evaluation.ipynb` to load these logs and compute aggregated metrics and tables for reporting.

## Results (predictions)

The `Results/` directory contains COCO-format prediction files produced by the models, grouped by experiment:

- `Results/Experiment_1/`
- `Results/Experiment_2.1/`
- …

Filenames follow the pattern:

```text
{dataset}_{split}_{model(-and-repeat)}_predictions.json
```

For example:

```text
apples_test_gd_b_predictions.json

apples_test_rexomni_transformers_r0_predictions.json
```

These files are the main input for the evaluation notebooks and for further analysis.

## Timing experiments

Timing-related data and analyses live under `Timing_Experiments/`:

- `Timing_Experiments/Timing_Experiments_Data/` — timing measurements (e.g., per dataset such as `apples/`, `tomatoes/`).
- `Timing_Experiments/timing_experiments.ipynb` — notebook that aggregates timing data and computes derived quantities (e.g., per-operation time, total annotation time, compute cost).

## External dependencies

The `External/` directory is intended to hold external repositories that some experiments may depend on (for example, cloned upstream code for specific models). It is empty by default.

If a notebook requires an external repo, clone it into `External/` and update any paths in the notebook as needed.

## Reproducibility notes

- Environments are version-locked via the YAML files in `environments/`.
- Raw predictions are stored in `Results/` in COCO JSON format.
- Evaluation logs and statistics are stored in `Evaluations/` as JSON and can be recomputed with `evaluation.ipynb`.
- Timing measurements and cost estimates are reproducible from the data and notebooks in `Timing_Experiments/`.

Refer to the individual notebooks and scripts for experiment-specific details.