# Install

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13. The pins are locked for 3.13 only.
2. Clone this repository and run `uv sync --frozen` at its root.

On Windows the locked `torch` is the CPU build from PyPI, so the scripts run on the CPU there. They were written for a CUDA GPU: one epoch of Swin-Base took 70 minutes on one, and takes far longer on a CPU.

The first run of a training script downloads CIFAR-10, about 170 MB, into `data/`, and the pretrained weights of each model, a few hundred megabytes apiece, into the Hugging Face cache. It needs a network connection and disk space. `data/` and `models/` are gitignored. `.\recipe.ps1 test` needs none of that: `tests/conftest.py` sets `HF_HUB_OFFLINE`, the tests replace the models, the processors and CIFAR-10, and a test that reached for the weights would fail instead of fetching them.

`.\recipe.ps1 build` packs the four scripts into a wheel in `dist\`.
