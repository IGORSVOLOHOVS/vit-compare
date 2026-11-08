# Manual

Run each script from the root of the repository, through `.\recipe.ps1 run <script>`: the paths `data/`, `models/` and `docs/assets/` are relative. No script takes options; the models, the batch sizes and the learning rate `5e-5` are fixed in the code.

## Train

`compare_vit_train.py` (the default of `.\recipe.ps1 run`) fine-tunes the three Vision Transformers in turn for one epoch, batch size 32, and prints the loss every 100 batches. Each fine-tuned model is saved to `models/<org>_<name>`. At the end it prints the table `ViT Model Training Performance (1 Epoch on CIFAR-10)` with the time in seconds and minutes and the average loss. A model that fails prints `Failed to train <model>: <reason>` and is left out of the table.

`task1_convnext_train.py` fine-tunes ConvNext-Base, batch size 8, with a progress bar, then measures its accuracy on the validation half of the CIFAR-10 test set, split with seed 42. It saves the model to `models/convnext_cifar10` and prints the table `Task 1: ConvNext Performance Result`. Any failure prints `An error occurred: <reason>`.

## Infer

`inference_compare.py` classifies CIFAR-10 test image 42 with every folder in `models/` and prints `Inference Performance Comparison (Target: <true class>)` with each model's prediction, confidence and average latency over 10 passes after one warm-up. It writes the image, enlarged to 448 x 448 with the prediction drawn on it, to `docs/assets/inference_<folder>.png`. A folder that fails prints `Error with <folder>: <reason>`.

`inference_convnext.py` needs `models/convnext_cifar10`, from `task1_convnext_train.py`; without it, it says so and stops. It warms up with 10 passes, runs the whole CIFAR-10 test set and prints `ConvNext Inference Performance (CIFAR-10 Test Set)` with the image count, total time, throughput and accuracy.

Every script exits 0, also when a model fails.
