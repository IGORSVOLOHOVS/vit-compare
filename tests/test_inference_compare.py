"""inference_compare.py with the fine-tuned models, the processor and CIFAR-10 replaced."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from fake_vision_models_and_datasets import FakeProcessor, FixedLogitsModel, SampleCifar10
from PIL import Image

import inference_compare


def test_the_visual_is_saved_with_a_dark_label_box(tmp_path: Path) -> None:
    target = tmp_path / "visual.png"
    inference_compare.create_inference_visual(
        Image.new("RGB", (200, 100), "red"), "cat", 0.5, target
    )
    saved = Image.open(target).convert("RGB")
    assert saved.size == (200, 100)
    colours = {colour for _, colour in saved.getcolors(maxcolors=100_000) or []}
    # White text on a black box near the top left; the far corner is untouched.
    assert (0, 0, 0) in colours and (255, 255, 255) in colours
    assert saved.getpixel((190, 90)) == (255, 0, 0)


def test_the_visual_falls_back_to_the_default_font(tmp_path: Path) -> None:
    target = tmp_path / "visual.png"
    real_truetype = inference_compare.ImageFont.truetype

    def no_arial(font: Any, *args: Any, **kwargs: Any) -> Any:
        # load_default() itself opens Pillow's bundled font through truetype().
        if font == "arial.ttf":
            raise OSError("no arial")
        return real_truetype(font, *args, **kwargs)

    with patch.object(inference_compare.ImageFont, "truetype", side_effect=no_arial) as truetype:
        inference_compare.create_inference_visual(
            Image.new("RGB", (200, 100), "red"), "cat", 0.5, target
        )
    assert truetype.call_args_list[0].args == ("arial.ttf", 24)
    assert target.is_file()


def infer(folder: str, logits: list[float]) -> tuple[tuple[str, float, float], Any, Any, Any]:
    processor = FakeProcessor()
    model = FixedLogitsModel(logits)
    with (
        patch.object(
            inference_compare.AutoImageProcessor, "from_pretrained", return_value=processor
        ) as load_processor,
        patch.object(
            inference_compare.AutoModelForImageClassification, "from_pretrained", return_value=model
        ) as load_model,
    ):
        result = inference_compare.run_inference(
            Path("models") / folder, Image.new("RGB", (32, 32)), torch.device("cpu")
        )
    return result, load_processor, load_model, model


def test_inference_names_the_class_with_the_highest_probability() -> None:
    logits = [0.0] * 10
    logits[3] = 5.0
    (label, confidence, latency), load_processor, load_model, model = infer(
        "microsoft_cvt-21", logits
    )
    assert label == "cat"
    expected = torch.softmax(torch.tensor(logits), dim=-1)[3].item()
    assert confidence == pytest.approx(expected)
    assert latency >= 0
    load_processor.assert_called_once_with("microsoft/cvt-21")
    load_model.assert_called_once_with(Path("models") / "microsoft_cvt-21")
    # One warm-up and ten timed passes, none of them tracking gradients.
    assert model.calls == 11
    assert model.grad_enabled == [False] * 11


def test_an_unknown_folder_takes_the_vit_processor() -> None:
    _, load_processor, _, _ = infer("my_own_model", [1.0] + [0.0] * 9)
    load_processor.assert_called_once_with("google/vit-base-patch16-224-in21k")


def test_main_runs_every_model_folder_and_reports_a_failing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "good").mkdir(parents=True)
    (tmp_path / "models" / "broken").mkdir()
    (tmp_path / "models" / "notes.txt").write_text("not a model", encoding="utf-8")

    def run(model_path: Path, image: Image.Image, device: torch.device) -> tuple[str, float, float]:
        if model_path.name == "broken":
            raise RuntimeError("bad weights")
        return "dog", 0.9, 0.0025

    with (
        patch.object(inference_compare.datasets, "CIFAR10", SampleCifar10),
        patch.object(inference_compare, "run_inference", run),
    ):
        inference_compare.main()
    out = capsys.readouterr().out
    assert "Error with broken: bad weights" in out
    assert "90.0%" in out and "2.50" in out
    assert "Target: dog" in out
    assert (tmp_path / "docs" / "assets" / "inference_good.png").is_file()
    assert not (tmp_path / "docs" / "assets" / "inference_broken.png").exists()
