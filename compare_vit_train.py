"""Module for comparing ViT model training performance on CIFAR-10."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

# Install rich traceback for better error reporting
install(show_locals=True)

CONSOLE = Console()


@dataclass(frozen=True)
class TrainingMetrics:
    """Dataclass to store training performance metrics."""

    model_name: str
    duration_seconds: float
    avg_loss: float


class VisionDataPipeline:
    """Handles dataset loading and preprocessing for vision models."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        """Initialize the data pipeline.

        Args:
            model_name: The HuggingFace model identifier.
            batch_size: Number of samples per batch.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]

    def get_dataloaders(
        self,
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        """Download CIFAR-10 and create training/validation/test DataLoaders.

        Returns:
            A tuple of (train_loader, val_loader, test_loader).
        """
        # CIFAR-10 is 32x32, most ViTs expect 224x224
        size = self.processor.size.get("height", 224)

        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
            ]
        )

        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        # We split the CIFAR test set into val and test for the exercise
        full_test_set = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        # Split test set 50/50 for val/test
        val_size = len(full_test_set) // 2
        test_size = len(full_test_set) - val_size
        val_set, test_set = torch.utils.data.random_split(full_test_set, [val_size, test_size])

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


class PerfTrainer:
    """Trains a model and measures performance metrics."""

    def __init__(self, device: torch.device) -> None:
        """Initialize with a target device.

        Args:
            device: The torch device (e.g., 'cuda' or 'cpu').
        """
        self.device = device

    def train_one_epoch(
        self,
        model_name: str,
        dataloader: DataLoader[Any],
        num_labels: int = 10,
    ) -> TrainingMetrics:
        """Load, fine-tune for one epoch, and measure time.

        Args:
            model_name: HuggingFace model identifier.
            dataloader: Training data loader.
            num_labels: Number of output classes.

        Returns:
            TrainingMetrics containing time and loss.
        """
        CONSOLE.print(f"\n[bold blue]Starting training for: {model_name}[/bold blue]")

        # Load model with specific number of labels
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        model.to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        start_time = time.perf_counter()
        total_loss = 0.0
        num_batches = len(dataloader)

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                CONSOLE.print(f"Batch {i}/{num_batches} - Loss: {loss.item():.4f}")

        end_time = time.perf_counter()
        duration = end_time - start_time
        avg_loss = total_loss / num_batches

        # Save model
        save_path = Path("./models") / model_name.replace("/", "_")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        CONSOLE.print(f"[green]Model saved to {save_path}[/green]")

        return TrainingMetrics(model_name=model_name, duration_seconds=duration, avg_loss=avg_loss)


def main() -> None:
    """Main execution point for model comparison."""
    models_to_compare = [
        "microsoft/cvt-21",
        "microsoft/swin-base-patch4-window7-224",
        "google/vit-base-patch16-224-in21k",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONSOLE.print(f"Using device: [bold cyan]{device}[/bold cyan]")

    trainer = PerfTrainer(device)
    all_metrics: list[TrainingMetrics] = []

    for model_name in models_to_compare:
        try:
            pipeline = VisionDataPipeline(model_name)
            train_loader, _, _ = pipeline.get_dataloaders()

            metrics = trainer.train_one_epoch(model_name, train_loader)
            all_metrics.append(metrics)
        except Exception as e:
            CONSOLE.print(f"[bold red]Failed to train {model_name}: {e}[/bold red]")

    # Report results
    table = Table(title="ViT Model Training Performance (1 Epoch on CIFAR-10)")
    table.add_column("Model Name", style="magenta")
    table.add_column("Time (s)", justify="right", style="cyan")
    table.add_column("Time (min)", justify="right", style="cyan")
    table.add_column("Avg Loss", justify="right", style="green")

    for m in all_metrics:
        table.add_row(
            m.model_name,
            f"{m.duration_seconds:.2f}",
            f"{m.duration_seconds / 60:.2f}",
            f"{m.avg_loss:.4f}",
        )

    CONSOLE.print(table)


if __name__ == "__main__":
    main()
