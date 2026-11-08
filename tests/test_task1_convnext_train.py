"""task1_convnext_train.py with the model, the processor and CIFAR-10 replaced."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from fake_vision_models_and_datasets import FakeCifar10, FakeProcessor, TinyClassifier

import task1_convnext_train


def pipeline_with(processor: FakeProcessor) -> task1_convnext_train.ConvNextPipeline:
    with patch.object(
        task1_convnext_train.AutoImageProcessor, "from_pretrained", return_value=processor
    ) as load:
        pipeline = task1_convnext_train.ConvNextPipeline("org/convnext", batch_size=2)
    load.assert_called_once_with("org/convnext")
    return pipeline


def test_the_loaders_split_the_test_set_the_same_way_every_time() -> None:
    pipeline = pipeline_with(FakeProcessor(height=8))
    with patch.object(task1_convnext_train.datasets, "CIFAR10", FakeCifar10):
        first = pipeline.get_dataloaders()
        second = pipeline.get_dataloaders()
    train, val, test = first
    assert len(train.dataset) == 6
    assert len(val.dataset) == 2 and len(test.dataset) == 3
    assert train.num_workers == 2
    # The split is seeded with 42, so validation and test hold the same images on every run.
    assert list(val.dataset.indices) == list(second[1].dataset.indices)
    assert list(test.dataset.indices) == list(second[2].dataset.indices)
    images, _ = next(iter(val))
    assert images.shape == (2, 3, 8, 8)


def test_one_epoch_trains_validates_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = TinyClassifier()
    before = model.head.weight.detach().clone()
    torch.manual_seed(2)
    train = [(torch.rand(2, 3, 8, 8), torch.tensor([0, 1])) for _ in range(3)]
    val = [
        (torch.rand(2, 3, 8, 8), torch.tensor([0, 1])),
        (torch.rand(1, 3, 8, 8), torch.tensor([2])),
    ]
    trainer = task1_convnext_train.ConvNextTrainer(torch.device("cpu"))
    with patch.object(
        task1_convnext_train.AutoModelForImageClassification, "from_pretrained", return_value=model
    ) as load:
        metrics = trainer.train_one_epoch("org/convnext", train, val)  # type: ignore[arg-type]
    load.assert_called_once_with("org/convnext", num_labels=10, ignore_mismatched_sizes=True)
    # Three training batches, then two validation batches.
    assert model.calls == 5
    assert not torch.equal(before, model.head.weight)
    assert metrics.avg_loss > 0
    assert 0.0 <= metrics.accuracy <= 100.0
    assert metrics.accuracy * 3 / 100 == pytest.approx(round(metrics.accuracy * 3 / 100))
    assert model.saved_to == Path("models") / "convnext_cifar10"
    assert not model.training


def test_main_prints_the_result_table(capsys: pytest.CaptureFixture[str]) -> None:
    class Pipeline:
        def __init__(self, model_name: str) -> None:
            assert model_name == "facebook/convnext-base-224-22k-1k"

        def get_dataloaders(self) -> tuple[list[Any], list[Any], list[Any]]:
            return [], [], []

    def train(self: Any, model_name: str, train: Any, val: Any) -> Any:
        return task1_convnext_train.TrainingMetrics(model_name, 120.0, 0.125, 87.5)

    with (
        patch.object(task1_convnext_train, "ConvNextPipeline", Pipeline),
        patch.object(task1_convnext_train.ConvNextTrainer, "train_one_epoch", train),
    ):
        task1_convnext_train.main()
    out = capsys.readouterr().out
    assert "120.00" in out and "2.00" in out and "0.1250" in out and "87.50%" in out


def test_main_reports_an_error_instead_of_raising(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(
        task1_convnext_train, "ConvNextPipeline", side_effect=RuntimeError("offline")
    ):
        task1_convnext_train.main()
    assert "An error occurred: offline" in capsys.readouterr().out
