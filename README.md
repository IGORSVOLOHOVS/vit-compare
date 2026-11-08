# vit-compare

Four console scripts that fine-tune three Vision Transformers and ConvNext on CIFAR-10 for one epoch each and compare how long they train, how low their loss falls and how fast they answer.

- `compare_vit_train.py` fine-tunes `microsoft/cvt-21`, `microsoft/swin-base-patch4-window7-224` and `google/vit-base-patch16-224-in21k`, saves each under `models/` and prints a table of training time and average loss. A model that fails to load is reported and skipped.
- `task1_convnext_train.py` fine-tunes `facebook/convnext-base-224-22k-1k` the same way, with mixed precision on CUDA, and also reports its accuracy on half of the CIFAR-10 test set.
- `inference_compare.py` loads every folder in `models/`, classifies CIFAR-10 test image 42 with each, prints prediction, confidence and latency, and draws the prediction into `docs/assets/inference_<folder>.png`.
- `inference_convnext.py` measures the fine-tuned ConvNext's throughput and accuracy over the whole CIFAR-10 test set.

## Results

All models were fine-tuned for one epoch on CIFAR-10 on an NVIDIA GPU.

| Model | Training time (min) | Avg loss | Parameters |
| :--- | :--- | :--- | :--- |
| microsoft/cvt-21 | 11.82 | 0.6569 | ~31M |
| google/vit-base-patch16 | 29.65 | 0.2203 | ~86M |
| microsoft/swin-base-patch4 | 70.54 | 0.1119 | ~87M |
| facebook/convnext-base | 9.42 | 0.0964 | ~88M |

Inference on one test image, averaged over 10 passes:

| Model | Prediction | Confidence | Latency (ms) |
| :--- | :--- | :--- | :--- |
| ViT-Base | `dog` | 98.0% | 5.82 |
| CvT-21 | `dog` | 98.7% | 22.09 |
| Swin-Base | `dog` | 99.8% | 22.28 |
| ConvNext-Base | `dog` | 97.8%* | 5.67 |

\*Estimated from test accuracy.

![ViT inference](docs/assets/inference_google_vit-base-patch16-224-in21k.png)
![CvT inference](docs/assets/inference_microsoft_cvt-21.png)
![Swin inference](docs/assets/inference_microsoft_swin-base-patch4-window7-224.png)

## Use

```powershell
.\recipe.ps1 run                         # compare_vit_train.py
.\recipe.ps1 run task1_convnext_train.py
.\recipe.ps1 run inference_compare.py
.\recipe.ps1 run inference_convnext.py
```

`docs/install.md` says what to install and `docs/manual.md` what each script prints. `.\recipe.ps1 test` runs the tests, mypy and ruff with the models, the processors and CIFAR-10 replaced: no test downloads weights or data.

## Licence

MIT, see `LICENSE`.

## Lab requirements

This repository complies with [revision 4.0.0](http://127.0.0.1:3000/igors-lab/software-requirements/src/tag/v4.0.0/BEST_REQUIREMENTS.md) of the lab requirements, except R-02 (#8), R-09 (#9), R-37 (#10), R-41 (#11), R-42 (#12).
