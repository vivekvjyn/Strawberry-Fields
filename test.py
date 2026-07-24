import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalMRR,
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG,
)

from libs.utils import load_config, load_model, load_audio_chunk, load_audio_sliding

console = Console()


def embed_file(model, path, sr, n_samples, device):
    audio = load_audio_chunk(path, sr, n_samples)
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.embed(tensor)
    return emb.cpu()


def embed_corpus(model, path, sr, n_samples, device, hop=None):
    chunks = load_audio_sliding(path, sr, n_samples, hop)
    tensors = torch.tensor(chunks, dtype=torch.float32).to(device)
    with torch.no_grad():
        embs = model.embed(tensors)
    return embs.cpu()


def main():
    ROOT, cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print()
    console.print(Rule("[bold cyan]Evaluation", style="cyan"))
    console.print()

    model, ckpt = load_model(cfg, device)

    sr = cfg["sample_rate"]
    n_samples = int(cfg["segment_length"] * sr)
    bs = cfg["batch_size"]

    corpus_dir = ROOT / cfg["test"]["corpus_dir"]
    query_dir = ROOT / cfg["test"]["queries_dir"]
    query_csv = ROOT / cfg["test"]["queries_csv"]

    corpus_files = sorted(corpus_dir.glob("*.wav"))

    query_to_song = {}
    with open(query_csv) as f:
        for row in csv.DictReader(f):
            query_to_song[row["Filename"]] = int(row["Song ID"])
    query_files = sorted(query_dir.glob("*.wav"))
    query_files = [f for f in query_files if f.name in query_to_song]

    corpus_embeddings = []
    corpus_song_ids = []
    all_scores = []
    all_targets = []
    top1_correct = 0

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        embed_task = progress.add_task("Embedding corpus", total=len(corpus_files))
        query_task = progress.add_task("Querying", total=len(query_files), top1="0/0", visible=False)

        with torch.no_grad():
            for f in corpus_files:
                song_id = int(f.stem)
                embs = embed_corpus(model, f, sr, n_samples, device)
                corpus_embeddings.append(embs)
                corpus_song_ids.extend([song_id] * embs.size(0))
                progress.update(embed_task, advance=1)

        corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
        progress.update(query_task, visible=True)

        unique_song_ids = sorted(set(corpus_song_ids))
        song_id_to_idx = {sid: i for i, sid in enumerate(unique_song_ids)}
        n_corpus = len(unique_song_ids)

        for f in query_files:
            emb = embed_file(model, f, sr, n_samples, device)
            sims = torch.mm(emb, corpus_embeddings.to(device).t()).squeeze(0).cpu()

            max_sims = torch.full((n_corpus,), -float("inf"))
            for i, sid in enumerate(corpus_song_ids):
                if sims[i] > max_sims[song_id_to_idx[sid]]:
                    max_sims[song_id_to_idx[sid]] = sims[i]

            gt = query_to_song[f.name]
            target = torch.tensor([1.0 if s == gt else 0.0 for s in unique_song_ids])
            all_scores.append(max_sims)
            all_targets.append(target)
            if unique_song_ids[max_sims.argmax().item()] == gt:
                top1_correct += 1

            progress.update(query_task, advance=1, top1=f"{top1_correct}/{len(all_scores)}")

    scores = torch.stack(all_scores)
    targets = torch.stack(all_targets)
    n_queries = len(all_scores)

    indexes = torch.arange(n_queries).unsqueeze(1).expand_as(scores)

    map_val = RetrievalMAP()(scores, targets, indexes=indexes).item()
    mrr_val = RetrievalMRR()(scores, targets, indexes=indexes).item()
    p1_val = RetrievalPrecision(top_k=1)(scores, targets, indexes=indexes).item()
    p5_val = RetrievalPrecision(top_k=5)(scores, targets, indexes=indexes).item()
    p10_val = RetrievalPrecision(top_k=10)(scores, targets, indexes=indexes).item()
    r5_val = RetrievalRecall(top_k=5)(scores, targets, indexes=indexes).item()
    r10_val = RetrievalRecall(top_k=10)(scores, targets, indexes=indexes).item()
    ndcg_val = RetrievalNormalizedDCG()(scores, targets, indexes=indexes).item()

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_row("Queries", f"[green]{n_queries}[/]")
    metrics.add_row("Corpus", f"[green]{n_corpus}[/] songs")
    metrics.add_row("Top-1", f"[green]{top1_correct / n_queries:.3f}[/] ({top1_correct}/{n_queries})")
    metrics.add_row("MAP", f"[green]{map_val:.3f}[/]")
    metrics.add_row("MRR", f"[green]{mrr_val:.3f}[/]")
    metrics.add_row("P@1", f"[green]{p1_val:.3f}[/]")
    metrics.add_row("P@5", f"[green]{p5_val:.3f}[/]")
    metrics.add_row("P@10", f"[green]{p10_val:.3f}[/]")
    metrics.add_row("R@5", f"[green]{r5_val:.3f}[/]")
    metrics.add_row("R@10", f"[green]{r10_val:.3f}[/]")
    metrics.add_row("NDCG", f"[green]{ndcg_val:.3f}[/]")
    console.print(Panel(metrics, title="[bold]Metrics", border_style="cyan", width=50))
    console.print()

    plot_dir = ROOT / ".cache"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "eval_metrics.png"

    k_values = [1, 2, 5, 10, 20, 50]
    p_at_k = [RetrievalPrecision(top_k=k)(scores, targets, indexes=indexes).item() for k in k_values]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#f8f8f8")

    axes[0].bar(range(len(k_values)), p_at_k, color="#4a90d9", tick_label=[f"P@{k}" for k in k_values])
    axes[0].set_ylabel("Score", fontsize=10, color="#333333")
    axes[0].set_title("Precision@k", fontsize=11, fontweight="bold", color="#222222")
    axes[0].set_facecolor("#f8f8f8")
    axes[0].grid(True, alpha=0.3, linestyle="--", axis="y")
    for spine in axes[0].spines.values():
        spine.set_color("#cccccc")

    bar_metrics = {"MAP": map_val, "MRR": mrr_val, "P@1": p1_val, "P@5": p5_val, "NDCG": ndcg_val}
    axes[1].bar(bar_metrics.keys(), bar_metrics.values(), color="#5b9bd5")
    axes[1].set_ylabel("Score", fontsize=10, color="#333333")
    axes[1].set_title("Summary Metrics", fontsize=11, fontweight="bold", color="#222222")
    axes[1].set_facecolor("#f8f8f8")
    axes[1].grid(True, alpha=0.3, linestyle="--", axis="y")
    axes[1].set_ylim(0, 1.05)
    for spine in axes[1].spines.values():
        spine.set_color("#cccccc")

    with torch.no_grad():
        all_ranks = []
        for i in range(n_queries):
            sorted_idx = scores[i].argsort(descending=True)
            gt = targets[i]
            correct_idx = gt.nonzero(as_tuple=True)[0]
            for idx in sorted_idx:
                if idx in correct_idx:
                    all_ranks.append((scores[i][idx] - scores[i].min()) / (scores[i].max() - scores[i].min() + 1e-8))
                    break
            else:
                all_ranks.append(0.0)

        all_ranks_t = torch.tensor(all_ranks)
        axes[2].hist(all_ranks_t.numpy(), bins=20, color="#6fa3e0", edgecolor="#f8f8f8")
        axes[2].set_xlabel("Normalized Score of Correct Match", fontsize=10, color="#333333")
        axes[2].set_ylabel("Count", fontsize=10, color="#333333")
        axes[2].set_title("Score Distribution (Correct)", fontsize=11, fontweight="bold", color="#222222")
        axes[2].set_facecolor("#f8f8f8")
        axes[2].grid(True, alpha=0.3, linestyle="--", axis="y")
        for spine in axes[2].spines.values():
            spine.set_color("#cccccc")

    plt.tight_layout()
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    console.print(f"[green]{plot_path}[/]")
    console.print()


if __name__ == "__main__":
    main()
