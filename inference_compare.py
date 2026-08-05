"""Module for running inference and comparing speed of fine-tuned ViT models."""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.table import Table
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification

CONSOLE = Console()

# CIFAR-10 Classes
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def create_inference_visual(
    image: Image.Image, label: str, confidence: float, output_path: Path
) -> None:
    """Create and save a visualization of the inference result."""
    draw = ImageDraw.Draw(image)

    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    text = f"{label} ({confidence:.1%})"

    # Draw background for text
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle(bbox, fill="black")
    draw.text((10, 10), text, font=font, fill="white")

    image.save(output_path)


def run_inference(
    model_path: Path, image: Image.Image, device: torch.device
) -> tuple[str, float, float]:
    """Run inference with a specific model and measure latency."""
    # Load processor from the original model hub to ensure correct settings
    # For speed comparison, we use the saved weights
    # Determining original model name from path
    model_name_map = {
        "microsoft_cvt-21": "microsoft/cvt-21",
        "microsoft_swin-base-patch4-window7-224": "microsoft/swin-base-patch4-window7-224",
        "google_vit-base-patch16-224-in21k": "google/vit-base-patch16-224-in21k",
    }

    orig_name = model_name_map.get(model_path.name, "google/vit-base-patch16-224-in21k")
    processor = AutoImageProcessor.from_pretrained(orig_name)
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        _ = model(**inputs)

    # Measure latency over 10 runs
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            outputs = model(**inputs)
            latencies.append(time.perf_counter() - start)

    avg_latency = sum(latencies) / len(latencies)

    # Get prediction
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)

    return CLASSES[idx.item()], conf.item(), avg_latency


def main() -> None:
    """Main execution point for inference comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONSOLE.print(f"Using device: [bold cyan]{device}[/bold cyan]")

    # Get a sample image from CIFAR-10 test set
    test_set = datasets.CIFAR10(root="./data", train=False, download=True)
    sample_idx = 42  # Chosen arbitrarily
    sample_img, true_label_idx = test_set[sample_idx]
    true_label = CLASSES[true_label_idx]

    # Resize sample for better visualization (though inference uses original/processed size)
    display_img = sample_img.resize((448, 448), Image.Resampling.LANCZOS)

    models_dir = Path("./models")
    assets_dir = Path("./docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_path in models_dir.iterdir():
        if not model_path.is_dir():
            continue

        CONSOLE.print(f"Running inference for [bold green]{model_path.name}[/bold green]...")
        try:
            pred_label, confidence, latency = run_inference(model_path, sample_img, device)

            # Save visual
            vis_path = assets_dir / f"inference_{model_path.name}.png"
            create_inference_visual(display_img.copy(), pred_label, confidence, vis_path)

            results.append(
                {
                    "name": model_path.name,
                    "label": pred_label,
                    "conf": confidence,
                    "latency": latency,
                }
            )
        except Exception as e:
            CONSOLE.print(f"[bold red]Error with {model_path.name}: {e}[/bold red]")

    # Output comparison table
    table = Table(title=f"Inference Performance Comparison (Target: {true_label})")
    table.add_column("Model Folder", style="magenta")
    table.add_column("Prediction", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency (ms)", justify="right", style="cyan")

    for r in results:
        table.add_row(r["name"], r["label"], f"{r['conf']:.1%}", f"{r['latency'] * 1000:.2f}")

    CONSOLE.print(table)


if __name__ == "__main__":
    main()
