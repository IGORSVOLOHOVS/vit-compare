"""Module for measuring ConvNext inference performance on CIFAR-10."""

import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

CONSOLE = Console()


class ConvNextInference:
    """Handles inference and performance measurement for ConvNext."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        """Initialize with a saved model path and device."""
        self.device = device
        CONSOLE.print(f"[bold blue]Loading model from {model_path}...[/bold blue]")

        self.processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def get_test_loader(self, batch_size: int = 32) -> DataLoader[Any]:
        """Prepare the CIFAR-10 test set."""
        size = self.processor.size.get("height", 224)
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
            ]
        )
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        return DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def measure_performance(self, dataloader: DataLoader[Any]) -> dict[str, float]:
        """Measure inference time and accuracy."""
        correct = 0
        total = 0

        # Warm-up (standard for inference benchmarks)
        CONSOLE.print("[yellow]Warming up GPU...[/yellow]")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            for _ in range(10):
                _ = self.model(dummy_input)

        start_time = time.perf_counter()

        with torch.no_grad(), Progress() as progress:
            task = progress.add_task("[cyan]Running inference...", total=len(dataloader))
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).logits
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                progress.update(task, advance=1)

        end_time = time.perf_counter()

        duration = end_time - start_time
        accuracy = 100 * correct / total
        throughput = total / duration

        return {
            "duration": duration,
            "accuracy": accuracy,
            "throughput": throughput,
            "total_images": float(total),
        }


def main() -> None:
    """Main execution point for ConvNext inference."""
    model_path = "./models/convnext_cifar10"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(model_path).exists():
        CONSOLE.print(
            f"[bold red]Model path {model_path} does not exist. Please train the model first.[/bold red]"
        )
        return

    inferencer = ConvNextInference(model_path, device)
    test_loader = inferencer.get_test_loader(batch_size=32)

    results = inferencer.measure_performance(test_loader)

    # Report results
    table = Table(title="ConvNext Inference Performance (CIFAR-10 Test Set)")
    table.add_column("Property", style="magenta")
    table.add_column("Value", style="cyan")

    table.add_row("Total Images", f"{results['total_images']:.0f}")
    table.add_row("Total Time (s)", f"{results['duration']:.4f}")
    table.add_row("Throughput (img/s)", f"{results['throughput']:.2f}")
    table.add_row("Test Accuracy", f"{results['accuracy']:.2f}%")

    CONSOLE.print(table)


if __name__ == "__main__":
    main()
