# vit-compare: software requirements specification

Reverse-engineered from the code at commit `2f1e8f3` (2026-11-08) on 2026-11-18. It states
what the code does, not what it was meant to do; where the two differ, the difference is
listed under 3.6 or 9. Documentation hub: [`docs/README.md`](docs/README.md).

<!-- jira-sync:start -->
Jira: KAN-86

## Tasks

This list is one ticket in Jira project KAN, [KAN-86](https://igorsvolohovs.atlassian.net/browse/KAN-86). Each line is a checkbox there: tick it in Jira and the sync ticks it here; add a checkbox there and it is appended here. Done items are the features described in section 3; open items are what the project declares but the code does not do yet.

- [x] ViT comparison (`compare_vit_train.py`) (section 3.1) <!-- spec-vit-comparison-comparevittrain-py -->
- [x] ConvNext training (`task1_convnext_train.py`) (section 3.2) <!-- spec-convnext-training-task1convnexttrain-py -->
- [x] Inference comparison (`inference_compare.py`) (section 3.3) <!-- spec-inference-comparison-inferencecompare-py -->
- [x] ConvNext inference (`inference_convnext.py`) (section 3.4) <!-- spec-convnext-inference-inferenceconvnext-py -->
- [x] Common (section 3.5) <!-- spec-common -->
- [ ] Every script exits 0, also when a model fails (declared, not implemented) <!-- spec-every-script-exits-0-also-when-a-model-fails -->
- [ ] `data/` and `models/` are gitignored (declared, not implemented) <!-- spec-data-and-models-are-gitignored -->
- [ ] Classes `DataPipeline`, `ModelManager`, `Trainer` (declared, not implemented) <!-- spec-classes-datapipeline-modelmanager-trainer -->
- [ ] Train each model "on GPU" (declared, not implemented) <!-- spec-train-each-model-on-gpu -->
- [ ] Train, val and test splits prepared for the comparison (declared, not implemented) <!-- spec-train-val-and-test-splits-prepared-for-the-compari -->
- [ ] Parameter counts and a ConvNext confidence "estimated from test accuracy" (declared, not implemented) <!-- spec-parameter-counts-and-a-convnext-confidence-estimat -->
- [ ] Manual tests (declared, not implemented) <!-- spec-manual-tests -->

<!-- jira-sync:end -->

## 1. Purpose and scope

**In scope.** Four standalone Python scripts that fine-tune pretrained image classifiers
from the Hugging Face Hub on CIFAR-10 for one epoch and measure them
(`README.md:3-8`):

- `compare_vit_train.py`: three Vision Transformers, training time and average loss;
- `task1_convnext_train.py`: ConvNext-Base, training time, average loss, validation accuracy;
- `inference_compare.py`: per-model prediction, confidence and latency on one test image,
  plus an annotated PNG per model;
- `inference_convnext.py`: throughput and accuracy of the fine-tuned ConvNext over the
  whole CIFAR-10 test set.

**Out of scope** (no code for it): hyperparameter search, more than one epoch,
command-line options, model serving, a library API for other projects (the wheel only
carries the four scripts as top-level modules, `pyproject.toml:46`).

## 2. Context

| Item | Description | Source |
| :--- | :--- | :--- |
| Operator | A person running the scripts from the repository root; who that is is not stated (see 9) | `docs/manual.md:3` |
| Hugging Face Hub | Supplies pretrained weights and image processors through `from_pretrained` | `compare_vit_train.py:47,118` |
| torchvision CIFAR-10 | Downloads CIFAR-10 into `./data` on first use (`download=True`) | `compare_vit_train.py:68-72` |
| Runtime | Python 3.13, torch 2.11.0; CUDA when available, CPU otherwise | `pyproject.toml:22,27`, `compare_vit_train.py:168` |
| Lab tooling | `recipe.ps1` verbs and Gitea workflows from the lab standard | `recipe.ps1`, `.gitea/workflows/` |

## 3. Functional requirements

### 3.1 ViT comparison (`compare_vit_train.py`)

- **FR-1.** The system shall fine-tune, in this order, `microsoft/cvt-21`,
  `microsoft/swin-base-patch4-window7-224` and `google/vit-base-patch16-224-in21k`.
  Source: `compare_vit_train.py:162-179`. Verification:
  `test_main_trains_every_model_and_reports_a_failure_without_stopping`.
- **FR-2.** The system shall preprocess CIFAR-10 for each model by resizing to a square of
  the processor's `size["height"]` (224 when absent), converting to a tensor and
  normalising with the processor's `image_mean` and `image_std`.
  Source: `compare_vit_train.py:47,58-66`. Verification:
  `test_the_loaders_split_the_test_set_in_half_and_resize_to_the_processor`.
- **FR-3.** The system shall load the CIFAR-10 training set into a shuffled loader of batch
  size 32 with 2 workers, and split the CIFAR-10 test set into a validation half
  (`len // 2`) and a test half (the rest) without a seed.
  Source: `compare_vit_train.py:38,68-83`. Verification:
  `test_the_loaders_split_the_test_set_in_half_and_resize_to_the_processor`.
- **FR-4.** The system shall load each model with `num_labels=10` and
  `ignore_mismatched_sizes=True` and train it for one epoch with AdamW (learning rate
  `5e-5`) and cross-entropy loss, printing the loss every 100 batches.
  Source: `compare_vit_train.py:118-145`. Verification:
  `test_one_epoch_trains_measures_and_saves`.
- **FR-5.** The system shall report the duration of the batch loop measured with
  `time.perf_counter()` and the average loss as the sum of batch losses divided by the
  number of batches. Source: `compare_vit_train.py:129-149`. Verification:
  `test_one_epoch_trains_measures_and_saves` (checks only `>= 0` and `> 0`).
- **FR-6.** The system shall save each fine-tuned model with `save_pretrained` to
  `./models/<model id with "/" replaced by "_">`. Source: `compare_vit_train.py:152-155`.
  Verification: `test_one_epoch_trains_measures_and_saves`.
- **FR-7.** When preparing or training a model raises, the system shall print
  `Failed to train <model>: <reason>` and continue with the next model.
  Source: `compare_vit_train.py:175-182`. Verification:
  `test_main_trains_every_model_and_reports_a_failure_without_stopping`.
- **FR-8.** The system shall print the table
  `ViT Model Training Performance (1 Epoch on CIFAR-10)` with the columns Model Name,
  Time (s), Time (min) and Avg Loss, one row per model that finished.
  Source: `compare_vit_train.py:185-199`. Verification:
  `test_main_trains_every_model_and_reports_a_failure_without_stopping`.

### 3.2 ConvNext training (`task1_convnext_train.py`)

- **FR-9.** The system shall fine-tune `facebook/convnext-base-224-22k-1k` for one epoch,
  batch size 8, with AdamW (`5e-5`) and cross-entropy, showing a rich progress bar and
  printing the loss every 100 batches. Source: `task1_convnext_train.py:42,135-167,203`.
  Verification: `test_one_epoch_trains_validates_and_saves`.
- **FR-10.** The system shall enable autocast and gradient scaling only when the device is
  CUDA. Source: `task1_convnext_train.py:139,154`. Verification: no test (only the CPU
  path runs in `test_one_epoch_trains_validates_and_saves`).
- **FR-11.** The system shall split the CIFAR-10 test set into halves with a generator
  seeded with 42, so that validation and test hold the same images on every run.
  Source: `task1_convnext_train.py:80-84`. Verification:
  `test_the_loaders_split_the_test_set_the_same_way_every_time`.
- **FR-12.** After training, the system shall compute validation accuracy in eval mode
  without gradients as `100 * correct / total`. Source: `task1_convnext_train.py:174-185`.
  Verification: `test_one_epoch_trains_validates_and_saves`.
- **FR-13.** The system shall save the model to `./models/convnext_cifar10`.
  Source: `task1_convnext_train.py:188-191`. Verification:
  `test_one_epoch_trains_validates_and_saves`.
- **FR-14.** The system shall print the table `Task 1: ConvNext Performance Result`
  (columns Property and Value) with the rows Model, Time (seconds), Time (minutes), Avg Loss
  and Val Accuracy.
  Source: `task1_convnext_train.py:219-229`. Verification: `test_main_prints_the_result_table`.
- **FR-15.** On any exception in the run, the system shall print
  `An error occurred: <reason>` instead of raising. Source: `task1_convnext_train.py:211,231-232`.
  Verification: `test_main_reports_an_error_instead_of_raising`.
- **FR-16.** On CUDA the system shall empty the CUDA cache before starting.
  Source: `task1_convnext_train.py:206-207`. Verification: no test (`pyproject.toml:78-79`).

### 3.3 Inference comparison (`inference_compare.py`)

- **FR-17.** The system shall take CIFAR-10 test image 42 and name its true class from the
  CIFAR-10 class list. Source: `inference_compare.py:18,93-96`. Verification:
  `test_main_runs_every_model_folder_and_reports_a_failing_one`.
- **FR-18.** The system shall run inference for every directory directly under `./models`
  and skip entries that are not directories. Source: `inference_compare.py:107-113`.
  Verification: `test_main_runs_every_model_folder_and_reports_a_failing_one`.
- **FR-19.** The system shall load the model from the folder and the processor from the
  Hub id mapped from the folder name, using `google/vit-base-patch16-224-in21k` for any
  name outside the three-entry map. Source: `inference_compare.py:51-59`. Verification:
  `test_inference_names_the_class_with_the_highest_probability`,
  `test_an_unknown_folder_takes_the_vit_processor`.
- **FR-20.** The system shall run one warm-up pass and ten timed passes without gradients,
  report the mean of the ten times, and report the class with the highest softmax
  probability and that probability. Source: `inference_compare.py:65-84`. Verification:
  `test_inference_names_the_class_with_the_highest_probability`.
- **FR-21.** The system shall write the image, enlarged to 448 x 448 (LANCZOS), with
  `<label> (<confidence>)` in white on a black box at (10, 10), Arial 24 or Pillow's
  default font, to `./docs/assets/inference_<folder>.png`, creating the directory.
  Source: `inference_compare.py:21-41,99-117`. Verification:
  `test_the_visual_is_saved_with_a_dark_label_box`,
  `test_the_visual_falls_back_to_the_default_font`.
- **FR-22.** When a folder fails, the system shall print `Error with <folder>: <reason>`,
  write no image for it and continue. Source: `inference_compare.py:112-128`.
  Verification: `test_main_runs_every_model_folder_and_reports_a_failing_one`.
- **FR-23.** The system shall print the table
  `Inference Performance Comparison (Target: <true class>)` with Model Folder, Prediction,
  Confidence (one decimal, percent) and Latency (ms, two decimals).
  Source: `inference_compare.py:131-140`. Verification:
  `test_main_runs_every_model_folder_and_reports_a_failing_one`.

### 3.4 ConvNext inference (`inference_convnext.py`)

- **FR-24.** When `./models/convnext_cifar10` does not exist, the system shall print that it
  does not exist, ask to train the model first, and stop without loading anything.
  Source: `inference_convnext.py:92-97`. Verification:
  `test_main_stops_when_no_model_was_trained`.
- **FR-25.** The system shall load the processor of `facebook/convnext-base-224-22k-1k`
  from the Hub and the model from `./models/convnext_cifar10`, in eval mode.
  Source: `inference_convnext.py:24-34,89`. Verification: helper `inferencer_with` in
  `tests/test_inference_convnext.py:13-27`.
- **FR-26.** The system shall run the whole CIFAR-10 test set, preprocessed as in FR-2,
  in unshuffled batches of 32. Source: `inference_convnext.py:36-47,100`. Verification:
  `test_the_test_loader_resizes_every_cifar_image_to_the_processor_size`.
- **FR-27.** The system shall make ten warm-up passes on a random `1x3x224x224` tensor, then
  time one pass over the loader without gradients and report accuracy
  (`100 * correct / total`), throughput (`total / duration`), duration and image count.
  Source: `inference_convnext.py:49-84`. Verification:
  `test_accuracy_counts_the_images_whose_class_the_model_names`.
- **FR-28.** The system shall print the table
  `ConvNext Inference Performance (CIFAR-10 Test Set)` (columns Property and Value) with the
  rows Total Images, Total Time (s), Throughput (img/s) and Test Accuracy.
  Source: `inference_convnext.py:105-114`.
  Verification: `test_main_measures_the_trained_model_and_prints_the_table`.

### 3.5 Common

- **FR-29.** Each script shall use the CUDA device when `torch.cuda.is_available()` and the
  CPU otherwise. Source: `compare_vit_train.py:168`, `task1_convnext_train.py:205`,
  `inference_compare.py:89`, `inference_convnext.py:90`. Verification: no test.

### 3.6 Declared but not implemented

| Claim | Where | What the code does |
| :--- | :--- | :--- |
| Every script exits 0, also when a model fails | `docs/manual.md:17` | Code reading, not run: `inference_compare.py:93,107` are outside the `try`; `inference_convnext.py` has no handler |
| `data/` and `models/` are gitignored | `docs/install.md:8` | `.gitignore` has no such lines since `b67f28b`; `git check-ignore` matches nothing |
| Classes `DataPipeline`, `ModelManager`, `Trainer` | `docs/compare_vit_train.md:26-34` | `VisionDataPipeline` and `PerfTrainer`; model loading sits in `PerfTrainer.train_one_epoch` |
| Train each model "on GPU" | `docs/compare_vit_train.md:15` | Falls back to the CPU (FR-29) |
| Train, val and test splits prepared for the comparison | `docs/compare_vit_train.md:13` | Prepared, never used (`compare_vit_train.py:177`) |
| Parameter counts and a ConvNext confidence "estimated from test accuracy" | `README.md:14-30` | No code computes either |
| Manual tests | `tests/manual/tests.json` | The list is empty |

## 4. Interfaces

### 4.1 Command line

No script takes arguments (`docs/manual.md:3`).

| Command | Reads | Writes | Prints |
| :--- | :--- | :--- | :--- |
| `.\recipe.ps1 run` or `... run compare_vit_train.py` | Hub, `./data` | `./models/<org>_<name>/` x3 | FR-8 table |
| `.\recipe.ps1 run task1_convnext_train.py` | Hub, `./data` | `./models/convnext_cifar10/` | FR-14 table |
| `.\recipe.ps1 run inference_compare.py` | `./models/*/`, Hub processors, `./data` | `./docs/assets/inference_<folder>.png` | FR-23 table |
| `.\recipe.ps1 run inference_convnext.py` | `./models/convnext_cifar10/`, Hub processor, `./data` | nothing | FR-28 table |

`recipe.ps1 run` executes `uv run --frozen python <arguments>` and defaults to
`compare_vit_train.py` (`recipe.ps1:60-65`).

### 4.2 Files

| File | Format | Producer | Consumer |
| :--- | :--- | :--- | :--- |
| `data/` | torchvision CIFAR-10 layout | `datasets.CIFAR10(download=True)` | all four scripts |
| `models/<folder>/` | transformers `save_pretrained` output, model only (no processor) | FR-6, FR-13 | FR-18, FR-25 |
| `docs/assets/inference_<folder>.png` | PNG, 448 x 448, Git LFS (`.gitattributes:9`) | FR-21 | `README.md:32-34` |
| `tests/manual/tests.json` | JSON `{"version": 1, "tests": [...]}`, entries with `id`, `title`, `expected`, `steps` | hand-written | `tests/test_manual_list.py` |

## 5. Data

| Entity | Definition | Source |
| :--- | :--- | :--- |
| `TrainingMetrics` (ViT) | frozen dataclass: `model_name`, `duration_seconds`, `avg_loss` | `compare_vit_train.py:26-32` |
| `TrainingMetrics` (ConvNext) | as above plus `accuracy` (percent) | `task1_convnext_train.py:29-36` |
| Inference result | tuple `(label, confidence, mean latency in s)` | `inference_compare.py:44-84` |
| ConvNext measurement | dict `duration`, `accuracy`, `throughput`, `total_images` | `inference_convnext.py:79-84` |
| Class list | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (index = CIFAR-10 label) | `inference_compare.py:18` |

Nothing is stored apart from the files in 4.2; there is no database.

## 6. Non-functional requirements

- **NFR-1.** Python 3.13 only. Source: `pyproject.toml:22`, `uv.lock:3`.
- **NFR-2.** Runtime and development dependencies are pinned to exact versions and resolved
  from `uv.lock` with `--frozen`. Source: `pyproject.toml:27-31,37-43`, `recipe.ps1:63,101`.
- **NFR-3.** The four scripts and `tools/` pass `mypy --strict`. Source: `pyproject.toml:91-94`.
- **NFR-4.** The project's code passes ruff 0.16.1 with line length 100, rule sets
  E, W, F, I, UP, B, BLE, SIM, C90, S, RUF, and LF line endings.
  Source: `pyproject.toml:52-71`.
- **NFR-5.** The test suite holds branch coverage at 97 % or more. Source: `pyproject.toml:80,85`.
- **NFR-6.** No test downloads weights or data: `HF_HUB_OFFLINE` is set before
  `huggingface_hub` is imported, and fakes replace models, processors and CIFAR-10.
  Source: `tests/conftest.py:6`, `tests/fake_vision_models_and_datasets.py:1`.
  Verification: `test_the_hub_is_offline_for_the_whole_run`.
- **NFR-7.** The scripts must run with the repository root as working directory.
  Source: `docs/manual.md:3`, `recipe.ps1:45`.
- **NFR-8.** The repository carries the lab tooling of
  [repo-quality-template](https://github.com/IGORSVOLOHOVS/repo-quality-template); its
  verdict is `.\recipe.ps1 test`, which also fails when `pyproject.toml`'s version differs
  from `$Version`. Source: `recipe.ps1:84-111`, `README.md:53`.

No performance target is set anywhere in code or config; the timings in `README.md:12-30`
are measurements, not requirements.

## 7. Constraints and dependencies

| Package | Version | Role | Source |
| :--- | :--- | :--- | :--- |
| torch | 2.11.0 | training, inference | `pyproject.toml:27` |
| torchvision | 0.26.0 | CIFAR-10, transforms | `pyproject.toml:28` |
| transformers | 5.5.4 | Auto model and processor classes | `pyproject.toml:29` |
| pillow | 12.2.0 | PNG drawing | `pyproject.toml:30` |
| rich | 15.0.0 | console tables, progress, tracebacks | `pyproject.toml:31` |
| huggingface-hub | 1.33.0 (transitive) | read directly by `tests/test_no_weights_are_downloaded.py` | `uv.lock` |
| ruff, mypy, pytest, pytest-cov, coverage | 0.16.1, 2.3.0, 8.4.2, 5.0.0, 7.15.2 | dev group | `pyproject.toml:37-43` |
| setuptools | 80.9.0 | build backend | `pyproject.toml:10` |

`uv.lock` resolves 68 packages, the virtual `vit-compare` entry included. On Windows the locked torch is the CPU build
(`pyproject.toml:24-26`, `docs/install.md:6`).

## 8. Acceptance

1. `.\recipe.ps1 test` exits 0 (`recipe.ps1:84-111`). This covers every FR whose
   verification names a test, NFR-3 to NFR-6 and NFR-8.
2. With network access, ideally on a CUDA GPU, run the four commands of 4.1 in the order
   shown and check: the four tables and their columns or rows (FR-8, FR-14, FR-23, FR-28); the
   folders under `models/` (FR-6, FR-13); one PNG per model folder (FR-21).
3. FR-10, FR-16 and FR-29 have no test; they hold only on a CUDA run or by code review.

## 9. Open questions

1. Who the scripts are for and what decision the comparison is meant to support. Partly
   answered by the history: they began as a course exercise, whose brief `docs/task.md`
   was removed in `2833f98` (fine-tune the three ViTs, time one epoch each, save them for
   the next task). Who reads the results now is unknown.
2. Which GPU, driver and operating system produced the figures in `README.md:12-30`.
3. Where the "Parameters" column and the ConvNext inference row (latency 5.67 ms,
   confidence marked "estimated from test accuracy") come from. Partly answered: the three
   ViT counts (~31M, ~87M, ~86M) match the removed brief (`git show 2833f98^:docs/task.md`,
   lines 5-7); nothing in the repository gives ~88M for ConvNext. No code estimates a
   confidence from test accuracy. `inference_compare.py` would print a latency for a
   `models/convnext_cifar10` folder (FR-18), but no PNG from such a run is committed, so
   whether 5.67 ms came from it is unknown.
4. Whether `docs/manual.md:17` (exit 0 always) is the intended behaviour, in which case the
   inference scripts do not meet it.
5. Whether `data/` and `models/` were meant to leave `.gitignore` in `b67f28b`, or
   `docs/install.md:8` is right.
6. Whether `models/convnext_cifar10` should get its own processor in `inference_compare.py`
   instead of the ViT fallback.
7. Whether the ViTs were meant to be evaluated on the validation or test half that
   `compare_vit_train.py` builds and discards. The removed brief asks to prepare train,
   validation and test parts and to time one epoch (`2833f98^:docs/task.md:26,28`), and
   says accuracy will be needed later (line 3), but asks for no evaluation in this task.
8. Inference, not measured: CUDA latency in `inference_compare.py` is taken without
   `torch.cuda.synchronize()`; whether the README latencies are affected is unknown.
9. What lab requirements R-37, R-41 and R-42 demand; the repository names them
   (`README.md:53`) but their text is on a local server. Partly answered: commit `503dd6b`
   calls `README.md:53` itself "the R-42 line", so why R-42 is still listed as an exception
   is unknown; what R-37 and R-41 demand is unknown.
10. Whether `.agent/` and `.vscode/` (C++, CMake and generic Python agent and editor
    material from the first commit) are meant to stay in this Python-only repository.

## Screenshots of requirements

### FR-1 - The system shall fine-tune, in this order, microsoft/cvt-21,

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-2 - The system shall preprocess CIFAR-10 for each model by resizing to a square of

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-3 - The system shall load the CIFAR-10 training set into a shuffled loader of batch

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-4 - The system shall load each model with numlabels=10 and

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-5 - The system shall report the duration of the batch loop measured with

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-6 - The system shall save each fine-tuned model with savepretrained to

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-7 - When preparing or training a model raises, the system shall print

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-8 - The system shall print the table

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-9 - The system shall fine-tune facebook/convnext-base-224-22k-1k for one epoch,

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-10 - The system shall enable autocast and gradient scaling only when the device is

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-11 - The system shall split the CIFAR-10 test set into halves with a generator

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-12 - After training, the system shall compute validation accuracy in eval mode

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-13 - The system shall save the model to ./models/convnextcifar10

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-14 - The system shall print the table Task 1: ConvNext Performance Result

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-15 - On any exception in the run, the system shall print

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-16 - On CUDA the system shall empty the CUDA cache before starting

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-17 - The system shall take CIFAR-10 test image 42 and name its true class from the

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-18 - The system shall run inference for every directory directly under ./models

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-19 - The system shall load the model from the folder and the processor from the

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-20 - The system shall run one warm-up pass and ten timed passes without gradients,

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-21 - The system shall write the image, enlarged to 448 x 448 (LANCZOS), with

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-22 - When a folder fails, the system shall print Error with <folder>: <reason>,

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-23 - The system shall print the table

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-24 - When ./models/convnextcifar10 does not exist, the system shall print that it

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-25 - The system shall load the processor of facebook/convnext-base-224-22k-1k

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-26 - The system shall run the whole CIFAR-10 test set, preprocessed as in FR-2,

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-27 - The system shall make ten warm-up passes on a random 1x3x224x224 tensor, then

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-28 - The system shall print the table

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-29 - Each script shall use the CUDA device when torch.cuda.isavailable() and the

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.
