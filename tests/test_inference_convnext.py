"""inference_convnext.py with the fine-tuned model, the processor and CIFAR-10 replaced."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fake_vision_models_and_datasets import FakeCifar10, FakeProcessor, FixedLogitsModel

import inference_convnext


def inferencer_with(model: FixedLogitsModel) -> inference_convnext.ConvNextInference:
    with (
        patch.object(
            inference_convnext.AutoImageProcessor, "from_pretrained", return_value=FakeProcessor()
        ) as load_processor,
        patch.object(
            inference_convnext.AutoModelForImageClassification,
            "from_pretrained",
            return_value=model,
        ) as load_model,
    ):
        inferencer = inference_convnext.ConvNextInference("models/x", torch.device("cpu"))
    load_processor.assert_called_once_with("facebook/convnext-base-224-22k-1k")
    load_model.assert_called_once_with("models/x")
    return inferencer


def test_the_test_loader_resizes_every_cifar_image_to_the_processor_size() -> None:
    inferencer = inferencer_with(FixedLogitsModel([0.0] * 10))
    with patch.object(inference_convnext.datasets, "CIFAR10", FakeCifar10):
        loader = inferencer.get_test_loader(batch_size=2)
    assert len(loader.dataset) == 5
    images, _ = next(iter(loader))
    assert images.shape == (2, 3, 8, 8)


def test_accuracy_counts_the_images_whose_class_the_model_names() -> None:
    logits = [0.0] * 10
    logits[1] = 3.0
    model = FixedLogitsModel(logits)
    inferencer = inferencer_with(model)
    loader = [
        (torch.zeros(2, 3, 8, 8), torch.tensor([1, 1])),
        (torch.zeros(2, 3, 8, 8), torch.tensor([1, 4])),
    ]
    results = inferencer.measure_performance(loader)  # type: ignore[arg-type]
    assert results["total_images"] == 4.0
    assert results["accuracy"] == pytest.approx(75.0)
    assert results["duration"] > 0
    assert results["throughput"] == pytest.approx(4 / results["duration"])
    # Ten warm-up passes, then one per batch, all without gradients.
    assert model.calls == 12
    assert not any(model.grad_enabled)


def test_main_stops_when_no_model_was_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(inference_convnext, "ConvNextInference") as inferencer:
        inference_convnext.main()
    inferencer.assert_not_called()
    assert "Please train the model" in capsys.readouterr().out


def test_main_measures_the_trained_model_and_prints_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "convnext_cifar10").mkdir(parents=True)
    logits = [0.0] * 10
    logits[0] = 2.0
    with (
        patch.object(
            inference_convnext.AutoImageProcessor, "from_pretrained", return_value=FakeProcessor()
        ),
        patch.object(
            inference_convnext.AutoModelForImageClassification,
            "from_pretrained",
            return_value=FixedLogitsModel(logits),
        ),
        patch.object(inference_convnext.datasets, "CIFAR10", FakeCifar10),
    ):
        inference_convnext.main()
    out = capsys.readouterr().out
    # FakeCifar10's five test labels are 0..4 and the model always says 0: one in five.
    assert "20.00%" in out
    assert "Total Images" in out
