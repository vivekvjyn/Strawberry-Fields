import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

import time
import mlflow
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import SiameseNetwork, NxLoss
from model.dataset import TripletDataset
from libs.utils import load_config
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

console = Console()


def main():
    ROOT, cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print()
    console.print(Rule("[bold cyan]Training", style="cyan"))
    console.print()

    dataset_dir = ROOT / cfg["dataset_dir"]

    if not dataset_dir.exists():
        console.print(f"[red]Corpus not found at {dataset_dir}[/]")
        sys.exit(1)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"run_{int(time.time())}"):
        mlflow.log_params({
            "sample_rate": cfg["sample_rate"],
            "segment_length": cfg["segment_length"],
            "embedding_dim": cfg["embedding_dim"],
            "n_mels": cfg["n_mels"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "temperature": cfg["temperature"],
            "device": device,
        })

        model = SiameseNetwork(
            embedding_dim=cfg["embedding_dim"],
            sample_rate=cfg["sample_rate"],
            n_mels=cfg["n_mels"],
        ).to(device)

        dataset = TripletDataset(
            data_dir=str(dataset_dir),
            sample_rate=cfg["sample_rate"],
            segment_length=cfg["segment_length"],
        )

        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )

        criterion = NxLoss(temperature=cfg["temperature"])
        optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

        checkpoint_dir = ROOT / cfg["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            train_loss = 0.0

            with Progress(
                TextColumn(f"[bold cyan]Epoch {epoch}/{cfg['epochs']}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TextColumn("[green]loss={task.fields[loss]}[/]"),
                console=console,
            ) as progress:
                task = progress.add_task("train", total=len(train_loader), loss="...")

                for corpus, pos_key, neg_key in train_loader:
                    corpus = corpus.to(device)
                    pos_key = pos_key.to(device)
                    neg_key = neg_key.to(device)

                    emb_corpus = model(corpus, augment=False)
                    emb_pos = model(pos_key, augment=False)
                    emb_neg = model(neg_key, augment=False)

                    keys = torch.stack([emb_pos, emb_neg], dim=1)
                    loss = criterion(emb_corpus, keys)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    progress.update(task, advance=1, loss=f"{loss.item():.4f}")

            scheduler.step()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for corpus, pos_key, neg_key in val_loader:
                    corpus = corpus.to(device)
                    pos_key = pos_key.to(device)
                    neg_key = neg_key.to(device)

                    emb_corpus = model(corpus, augment=False)
                    emb_pos = model(pos_key, augment=False)
                    emb_neg = model(neg_key, augment=False)

                    keys = torch.stack([emb_pos, emb_neg], dim=1)
                    loss = criterion(emb_corpus, keys)
                    val_loss += loss.item()

            val_loss /= max(len(val_loader), 1)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                }, ckpt_path)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            console.print(f"  train=[green]{train_loss:.4f}[/]  val=[yellow]{val_loss:.4f}[/]  best=[bold]{best_val_loss:.4f}[/]")

        mlflow.log_metric("best_val_loss", best_val_loss)

        console.print()
        console.print(Panel(f"[bold green]Training complete! Best val_loss: {best_val_loss:.4f}", border_style="green"))
        console.print(f"  Checkpoint: {checkpoint_dir / 'best_model.pt'}")
        console.print(f"  MLflow: {cfg['mlflow']['tracking_uri']}")
        console.print()


if __name__ == "__main__":
    main()
