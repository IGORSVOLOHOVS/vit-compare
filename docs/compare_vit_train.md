# Task: ViT Performance Comparison Training

## Objective
Implement a training script `compare_vit_train.py` to compare the performance (training time for 1 epoch) of three Vision Transformer architectures on the CIFAR-10 dataset.

## Requirements
- **Models**:
    - `microsoft/cvt-21`
    - `microsoft/swin-base-patch4-window7-224`
    - `google/vit-base-patch16-224-in21k`
- **Dataset**: CIFAR-10 (torchvision).
- **Functionality**:
    - Load CIFAR-10 and prepare splits (train, val, test).
    - Universal loading for models using `AutoImageProcessor` and `AutoModelForImageClassification`.
    - Train each model for 1 epoch on GPU.
    - Measure and log training time for each model using `time`.
    - Save fine-tuned models to `./models/{model_name_slug}`.
- **Constraints**:
    - Python 3.10+ typing (`mypy --strict` compliant).
    - Clean Architecture (Modular structure).
    - Use `rich` for UI and logging.

## Implementation Plan

### 1. Data Module
- Define a `DataPipeline` class to handle transformations and `DataLoaders`.
- Transformations should be dynamic based on the model's `AutoImageProcessor`.

### 2. Model Module
- Define a `ModelManager` class to load models and processors.
- Support for `microsoft/cvt-21`, `swin-base`, and `vit-base`.

### 3. Training Module
- Define a `Trainer` component.
- Implement training loop for 1 epoch.
- Integrate timing logic.

### 4. Main Script (`compare_vit_train.py`)
- Orchestrate the training of all three models.
- Print a comparison table at the end.

## Verification Criteria
- [ ] Script executes successfully.
- [ ] 1 epoch training completed for each model.
- [ ] Models saved in separate directories.
- [ ] Comparison table displayed with training times.
- [ ] Passes `ruff format` and `mypy --strict`.
