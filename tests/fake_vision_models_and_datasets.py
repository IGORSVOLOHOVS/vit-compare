"""Stand-ins for the Hugging Face model, the processor and CIFAR-10: nothing is downloaded."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from PIL import Image
from torch import nn


class FakeProcessor:
    """What AutoImageProcessor.from_pretrained returns, with a small image size."""

    def __init__(self, height: int = 8) -> None:
        self.size = {"height": height}
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> "FakeInputs":
        self.calls.append(kwargs)
        return FakeInputs()


class FakeInputs(dict[str, Any]):
    """A processor's output: a mapping that can be moved to a device."""

    def __init__(self) -> None:
        super().__init__(pixel_values=torch.zeros((1, 3, 8, 8)))
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> "FakeInputs":
        self.device = device
        return self


class TinyClassifier(nn.Module):
    """A real, trainable model small enough for a test: mean colour, then one linear layer."""

    def __init__(self, num_labels: int = 10) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.head = nn.Linear(3, num_labels)
        self.saved_to: Path | None = None
        self.calls = 0

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(logits=self.head(pixel_values.mean(dim=(2, 3))))

    def save_pretrained(self, path: Path) -> None:
        self.saved_to = Path(path)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")


class FixedLogitsModel(nn.Module):
    """A model that always answers with the same logits, so the winning class is known."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = torch.tensor([logits])
        self.calls = 0
        self.grad_enabled: list[bool] = []

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return SimpleNamespace(logits=self.logits.expand(pixel_values.shape[0], -1))


class FakeCifar10:
    """datasets.CIFAR10 with a handful of 32x32 images and no download."""

    sizes: ClassVar[dict[bool, int]] = {True: 6, False: 5}
    created: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self, root: str, train: bool = True, download: bool = False, transform: Any = None
    ) -> None:
        FakeCifar10.created.append({"root": root, "train": train, "download": download})
        self.transform = transform
        self.length = self.sizes[train]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index >= self.length:
            raise IndexError(index)
        image = Image.new("RGB", (32, 32), (index * 20, 100, 200))
        label = index % 10
        return (self.transform(image) if self.transform else image), label


class SampleCifar10:
    """datasets.CIFAR10 for the inference script: only sample 42 is read."""

    def __init__(self, root: str, train: bool = True, download: bool = False) -> None:
        self.train = train

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        return Image.new("RGB", (32, 32), (10, 20, 30)), 5
