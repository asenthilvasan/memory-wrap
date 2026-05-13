# Changed Line Guide vs. Original `memory-wrap`

This guide compares this repo (`memory-wrap-SENN`) against the sibling original repo at `../memory-wrap`.

**Scope:** source, config, infrastructure, and Markdown documentation files. Generated artifacts such as datasets, model outputs, curated images, `.DS_Store`, and run-output directories are intentionally excluded because they do not have meaningful source-code line numbers.

**Line numbers:** all line numbers refer to the `memory-wrap-SENN` version of the file.

**Purpose labels:** descriptions are intentionally broad. They identify the general role of each changed block without explaining the exact implementation.

## Summary of changed files

| File | Change type | General purpose |
|---|---:|---|
| `paper/pretrain_supcon.py` | Added | Encoder pretraining pipeline |
| `paper/train.py` | Modified | Pretrained-encoder training support and logging |
| `paper/utils/utils.py` | Modified | Evaluation loader stability |
| `paper/utils/datasets.py` | Modified | Deterministic CINIC10 test loading |
| `paper/scripts/generate_heatmaps.py` | Modified | Visualization sampling/output handling |
| `paper/scripts/generate_memory_images.py` | Modified | Visualization sampling/output handling |

## Modified files

### `paper/train.py`

| Line(s) | General purpose |
|---:|---|
| 14 | Runtime/data-loading environment configuration |
| 19-24 | CLI flags for optional pretrained-encoder workflow |
| 83-85 | Per-epoch memory-model progress logging |
| 132-134 | Per-epoch standard-model progress logging |
| 163-174 | Checkpoint output-directory naming for training variants |
| 200-206 | Optional pretrained model initialization and parameter-freezing setup |
| 209 | Optimizer parameter selection for trainable parameters only |
| 255 | Final run-summary logging behavior |

### `paper/utils/utils.py`

| Line(s) | General purpose |
|---:|---|
| 216-222 | Memory-loader iterator setup for evaluation stability |
| 227-231 | Memory-loader batch refresh behavior during evaluation |

### `paper/utils/datasets.py`

| Line(s) | General purpose |
|---:|---|
| 275 | Formatting/alignment cleanup inside CINIC10 test transform block |
| 276 | Deterministic CINIC10 test-loader ordering |

### `paper/scripts/generate_heatmaps.py`

| Line(s) | General purpose |
|---:|---|
| 19-25 | CLI controls for limiting and sampling visualization queries |
| 121-125 | Original-image panel rendering behavior |
| 181-188 | Optional deterministic shuffled test-query loader setup |
| 200-206 | Visualization output-directory routing by checkpoint variant |
| 215-220 | Optional early-stop behavior for visualization generation |

### `paper/scripts/generate_memory_images.py`

| Line(s) | General purpose |
|---:|---|
| 17-22 | CLI controls for limiting and sampling visualization queries |
| 62-69 | Optional deterministic shuffled test-query loader setup |
| 73-79 | Visualization output-directory routing by checkpoint variant |
| 93-98 | Optional early-stop behavior for visualization generation |

## Added code/infrastructure files

### `paper/pretrain_supcon.py`

| Line(s) | General purpose |
|---:|---|
| 1-32 | File-level purpose and usage documentation |
| 33-48 | Imports, runtime setup, and existing-project integration |
| 51-91 | CLI configuration for the pretraining job |
| 94-123 | Dataset-specific configuration table |
| 126-143 | Pretraining loss function interface and documentation |
| 144-155 | Loss-function setup for one kind of positive-pair structure |
| 156-162 | Loss-function setup for label-aware positive grouping |
| 164-184 | Loss-function tensor operations and numerical-stability logic |
| 186-195 | Loss-function aggregation and return value |
| 198-208 | Two-view augmentation helper |
| 211-217 | Main-program device and runtime setup |
| 219-243 | Image augmentation pipeline construction |
| 245-263 | Dataset and DataLoader construction |
| 265-285 | Model, optimizer, scheduler, and mixed-precision setup |
| 287-329 | Pretraining loop |
| 331-343 | Checkpoint output setup |
| 346-347 | CLI entrypoint |

## Files checked with no source changes detected

These files appeared in both repos and were not part of the source/config/doc diff found for this guide:

| Path group | Notes |
|---|---|
| `memory.py` | Core Memory Wrap implementation appears unchanged |
| `paper/architectures/*.py` | Backbone implementations appear unchanged |
| `paper/config/train.yaml` | Training config appears unchanged |
| `paper/eval.py`, `paper/eval_dir_mv.py`, `paper/explanation_accuracy.py` | Evaluation scripts appear unchanged |
| `paper/scripts/run_*.py`, `paper/scripts/train_aes_svhn.py` | Experiment helper scripts appear unchanged |
| `paper/scripts/wrappers/*.py` | Wrapper implementations appear unchanged |
| `paper/VIT/*.py` | ViT-related scripts appear unchanged |
