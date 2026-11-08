"""compare_vit_train.py with the model, the processor and CIFAR-10 replaced."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from fake_vision_models_and_datasets import FakeCifar10, FakeProcessor, TinyClassifier

import compare_vit_train


def pipeline_with(
    processor: FakeProcessor, batch_size: int = 2
) -> compare_vit_train.VisionDataPipeline:
    with patch.object(
        compare_vit_train.AutoImageProcessor, "from_pretrained", return_value=processor
    ) as load:
        pipeline = compare_vit_train.VisionDataPipeline("org/model", batch_size=batch_size)
    load.assert_called_once_with("org/model")
    return pipeline


def test_the_pipeline_keeps_the_name_the_batch_size_and_the_processor() -> None:
    processor = FakeProcessor()
    pipeline = pipeline_with(processor, batch_size=4)
    assert pipeline.model_name == "org/model"
    assert pipeline.batch_size == 4
    assert pipeline.processor is processor


def test_the_loaders_split_the_test_set_in_half_and_resize_to_the_processor() -> None:
    pipeline = pipeline_with(FakeProcessor(height=8))
    FakeCifar10.created.clear()
    with patch.object(compare_vit_train.datasets, "CIFAR10", FakeCifar10):
        train, val, test = pipeline.get_dataloaders()
    assert [entry["train"] for entry in FakeCifar10.created] == [True, False]
    assert all(entry["root"] == "./data" for entry in FakeCifar10.created)
    assert len(train.dataset) == 6
    assert len(val.dataset) == 2 and len(test.dataset) == 3
    assert train.num_workers == 2 and train.batch_size == 2
    images, labels = next(iter(val))
    assert images.shape == (2, 3, 8, 8)
    assert labels.shape == (2,)
    # Normalised with mean 0.5 and std 0.5, every value lies in [-1, 1].
    assert images.min() >= -1.0 and images.max() <= 1.0


def batches(count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(1)
    return [(torch.rand(2, 3, 8, 8), torch.tensor([0, 1])) for _ in range(count)]


def test_one_epoch_trains_measures_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = TinyClassifier()
    before = model.head.weight.detach().clone()
    trainer = compare_vit_train.PerfTrainer(torch.device("cpu"))
    with patch.object(
        compare_vit_train.AutoModelForImageClassification, "from_pretrained", return_value=model
    ) as load:
        metrics = trainer.train_one_epoch("org/model", batches(3), num_labels=10)
    load.assert_called_once_with("org/model", num_labels=10, ignore_mismatched_sizes=True)
    assert model.calls == 3
    assert not torch.equal(before, model.head.weight)
    assert metrics.model_name == "org/model"
    assert metrics.duration_seconds >= 0
    assert metrics.avg_loss > 0
    assert model.saved_to == Path("models") / "org_model"
    assert (tmp_path / "models" / "org_model" / "config.json").is_file()


def test_main_trains_every_model_and_reports_a_failure_without_stopping(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trained: list[str] = []

    class Pipeline:
        def __init__(self, model_name: str) -> None:
            if model_name == "microsoft/cvt-21":
                raise RuntimeError("no weights here")
            self.model_name = model_name

        def get_dataloaders(self) -> tuple[list[Any], list[Any], list[Any]]:
            return [], [], []

    def train(self: Any, model_name: str, loader: Any) -> compare_vit_train.TrainingMetrics:
        trained.append(model_name)
        return compare_vit_train.TrainingMetrics(model_name, 90.0, 0.25)

    with (
        patch.object(compare_vit_train, "VisionDataPipeline", Pipeline),
        patch.object(compare_vit_train.PerfTrainer, "train_one_epoch", train),
    ):
        compare_vit_train.main()
    assert trained == [
        "microsoft/swin-base-patch4-window7-224",
        "google/vit-base-patch16-224-in21k",
    ]
    out = capsys.readouterr().out
    assert "Failed to train microsoft/cvt-21: no weights here" in out
    assert "90.00" in out and "1.50" in out and "0.2500" in out
