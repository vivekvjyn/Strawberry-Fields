import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from pathlib import Path

from libs.utils import (
    console,
    get_db,
    setup_schema,
    clear_db,
    insert_song,
    parse_midi_events,
    extract_melody,
    parse_maestro_csv,
    find_midi_files,
    save_artifacts,
)
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.rule import Rule


def main():
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="[bold cyan]Strawberry Fields[/] — hum-to-search pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="Initialize database schema")

    ingest = sub.add_parser("ingest", help="Ingest MAESTRO dataset")
    ingest.add_argument("--dataset-dir", required=True, help="Path to MAESTRO directory")
    ingest.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup()
    elif args.command == "ingest":
        cmd_ingest(args)
    else:
        console.print(Panel(
            "[bold]Usage:[/]\n"
            "  python ingest.py [cyan]setup[/]              Initialize database\n"
            "  python ingest.py [cyan]ingest[/] --dataset-dir PATH",
            title="Strawberry Fields",
            border_style="cyan",
        ))


def cmd_setup():
    console.print()
    console.print(Rule("[bold cyan]Database Setup", style="cyan"))
    console.print()

    count = setup_schema()

    stats = Table(show_header=False, box=None, padding=(0, 2))
    stats.add_row("Schema", "[green]created[/]")
    stats.add_row("Songs", f"[bold]{count}[/]")
    console.print(Panel(stats, title="[bold]Status", border_style="cyan", width=50))
    console.print()


def cmd_ingest(args):
    console.print()
    console.print(Rule("[bold cyan]MAESTRO Ingestion", style="cyan"))
    console.print()

    if not os.path.isdir(args.dataset_dir):
        console.print(f"[red]Error:[/] {args.dataset_dir} is not a directory")
        sys.exit(1)

    metadata = parse_maestro_csv(args.dataset_dir)
    midi_files = find_midi_files(args.dataset_dir)

    if args.limit > 0:
        midi_files = midi_files[: args.limit]

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_row("Dataset", f"[cyan]{args.dataset_dir}[/]")
    info.add_row("Metadata", f"[green]{len(metadata)}[/] entries")
    info.add_row("MIDI files", f"[green]{len(midi_files)}[/]")
    if args.limit:
        info.add_row("Limit", f"[yellow]{args.limit}[/]")
    console.print(Panel(info, title="[bold]Scan Results", border_style="cyan", width=60))
    console.print()

    clear_db()
    console.print("[yellow]Database cleared.[/]")
    console.print()

    conn = get_db()
    cur = conn.cursor()

    processed = 0
    skipped = 0
    results = Table(title="Ingested Songs", show_lines=False, title_style="bold", border_style="dim")
    results.add_column("#", style="dim", width=4)
    results.add_column("Title", style="bold")
    results.add_column("Composer", style="cyan")
    results.add_column("Notes", justify="right", style="green")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing MIDI files", total=len(midi_files))

        for midi_path in midi_files:
            rel = os.path.relpath(midi_path, args.dataset_dir)
            basename = Path(rel).with_suffix("")
            stem = Path(midi_path).stem
            meta = metadata.get(str(basename), metadata.get(stem, {}))

            all_notes = parse_midi_events(midi_path)
            if not all_notes:
                skipped += 1
                progress.update(task, advance=1)
                continue

            melody = extract_melody(all_notes)
            if not melody:
                skipped += 1
                progress.update(task, advance=1)
                continue

            title = meta.get("title", stem.replace("_", " ").title())
            composer = meta.get("composer", "Unknown")
            year = meta.get("year", "")

            insert_song(cur, title, f"MAESTRO {year}", "", composer, "", "", melody)
            save_artifacts(stem, melody)
            processed += 1

            results.add_row(str(processed), title, composer, str(len(melody)))
            progress.update(task, advance=1)

    cur.close()
    conn.close()

    console.print()
    console.print(results)
    console.print()

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Ingested", f"[green]{processed}[/]")
    summary.add_row("Skipped", f"[yellow]{skipped}[/]")
    summary.add_row("Plots", f"[cyan].cache/plots/[/]")
    summary.add_row("Audio", f"[cyan].cache/audio/[/]")
    console.print(Panel(summary, title="[bold]Summary", border_style="green", width=50))
    console.print()


if __name__ == "__main__":
    main()
