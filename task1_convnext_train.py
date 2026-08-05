"""Module for fine-tuning ConvNext on CIFAR-10 and measuring performance."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.traceback import install
from torch import nn
from torch.utils.data import DataLoader, random_split
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
    accuracy: float


class ConvNextPipeline:
    """Handles dataset loading and preprocessing for ConvNext."""

    def __init__(self, model_name: str, batch_size: int = 8) -> None:
        """Initialize the data pipeline.

        Args:
            model_name: The HuggingFace model identifier.
            batch_size: Number of samples per batch.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        CONSOLE.print(f"[bold blue]Loading processor for {model_name}...[/bold blue]")
        self.processor = AutoImageProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]

    def get_dataloaders(
        self,
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        """Download CIFAR-10 and create training/validation/test DataLoaders.

        Returns:
            A tuple of (train_loader, val_loader, test_loader).
        """
        # CIFAR-10 is 32x32, ConvNext-base expects 224x224
        size = self.processor.size.get("height", 224)

        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
            ]
        )

        CONSOLE.print("[yellow]Downloading CIFAR-10 dataset...[/yellow]")
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        full_test_set = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        # Split test set 50/50 for val/test as requested (prepare train, test, val parts)
        val_size = len(full_test_set) // 2
        test_size = len(full_test_set) - val_size
        val_set, test_set = random_split(
            full_test_set, [val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        CONSOLE.print(f"Dataset sizes: Train={len(train_set)}, Val={val_size}, Test={test_size}")
        return train_loader, val_loader, test_loader


class ConvNextTrainer:
    """Trains ConvNext and measures performance metrics."""

    def __init__(self, device: torch.device) -> None:
        """Initialize with a target device.

        Args:
            device: The torch device (e.g., 'cuda' or 'cpu').
        """
        self.device = device

    def train_one_epoch(
        self,
        model_name: str,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        num_labels: int = 10,
    ) -> TrainingMetrics:
        """Load, fine-tune for one epoch, and measure time.

        Args:
            model_name: HuggingFace model identifier.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_labels: Number of output classes.

        Returns:
            TrainingMetrics containing time, loss, and accuracy.
        """
        CONSOLE.print(f"\n[bold green]Starting experiment: {model_name}[/bold green]")

        # Load model with specific number of labels
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        # Use Mixed Precision to save memory
        scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

        start_time = time.perf_counter()
        total_loss = 0.0
        num_batches = len(train_loader)

        model.train()
        with Progress() as progress:
            task = progress.add_task("[cyan]Training epoch 1...", total=num_batches)

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=(self.device.type == "cuda")
                ):
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress.update(task, advance=1, description=f"Loss: {loss.item():.4f}")

                # Explicit logging for background monitoring
                if i % 100 == 0:
                    print(f"Batch {i}/{num_batches} - Loss: {loss.item():.4f}")

        end_time = time.perf_counter()
        duration = end_time - start_time
        avg_loss = total_loss / num_batches

        # Validation for sanity check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images).logits
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # Save model
        save_path = Path("./models") / "convnext_cifar10"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        CONSOLE.print(f"[bold green]Model saved successfully to {save_path}[/bold green]")

        return TrainingMetrics(
            model_name=model_name,
            duration_seconds=duration,
            avg_loss=avg_loss,
            accuracy=accuracy,
        )


def main() -> None:
    """Main execution point for ConvNext fine-tuning."""
    model_name = "facebook/convnext-base-224-22k-1k"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    CONSOLE.print(f"Executing on device: [bold cyan]{device}[/bold cyan]")

    try:
        pipeline = ConvNextPipeline(model_name)
        train_loader, val_loader, _ = pipeline.get_dataloaders()

        trainer = ConvNextTrainer(device)
        metrics = trainer.train_one_epoch(model_name, train_loader, val_loader)

        # Report results
        table = Table(title="Task 1: ConvNext Performance Result")
        table.add_column("Property", style="magenta")
        table.add_column("Value", style="cyan")

        table.add_row("Model", metrics.model_name)
        table.add_row("Time (seconds)", f"{metrics.duration_seconds:.2f}")
        table.add_row("Time (minutes)", f"{metrics.duration_seconds / 60:.2f}")
        table.add_row("Avg Loss", f"{metrics.avg_loss:.4f}")
        table.add_row("Val Accuracy", f"{metrics.accuracy:.2f}%")

        CONSOLE.print(table)

    except Exception as e:
        CONSOLE.print(f"[bold red]An error occurred: {e}[/bold red]")


if __name__ == "__main__":
    main()
