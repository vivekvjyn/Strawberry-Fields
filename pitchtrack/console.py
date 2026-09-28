"""The pipeline's voice: one Rich console and the tables it speaks in.

Every command writes through this module, so their output looks alike: the same
colouring of what is wrong and what is not, the same progress bars, the same summary
tables — and a test can capture any of it by passing a console of its own.

Rich only decorates the message. Nothing here decides anything: a command reports what
it found, then returns its own exit status.
"""

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeRemainingColumn)
from rich.table import Table

from .corpus import ERROR, song_folders, verify_corpus

__all__ = ["console", "problems_table", "summary_table", "progress", "check_corpus",
           "show"]


console = Console()


def problems_table(problems, title="problems"):
    """Lay the problems found in a corpus out as a table, worst first.

    :param problems: The problems, in the order they were found.
    :type problems: list[pitchtrack.corpus.Problem]
    :param title: Heading over the table.
    :type title: str
    :return: The table, ready to print.
    :rtype: rich.table.Table
    """
    table = Table(title=title, header_style="bold")
    table.add_column("song", style="cyan", no_wrap=True)
    table.add_column("field", style="magenta", no_wrap=True)
    table.add_column("severity", no_wrap=True)
    table.add_column("detail")
    for problem in sorted(problems, key=lambda found: found.severity != ERROR):
        severity = ("[red]error[/red]" if problem.severity == ERROR
                    else "[yellow]warning[/yellow]")
        table.add_row(problem.song or ".", problem.field, severity, problem.detail)
    return table


def summary_table(rows, title="summary"):
    """Lay measurements out as a table of measure and value.

    :param rows: ``(measure, value)`` pairs, in the order they should appear.
    :type rows: list[tuple[str, str]]
    :param title: Heading over the table.
    :type title: str
    :return: The table, ready to print.
    :rtype: rich.table.Table
    """
    table = Table(title=title, header_style="bold")
    table.add_column("measure", style="dim", no_wrap=True)
    table.add_column("value", justify="right")
    for measure, value in rows:
        table.add_row(measure, value)
    return table


def progress(transient=True, target=console):
    """A progress bar for a sweep over a known number of recordings.

    :param transient: Clear the bar when the work ends, leaving only what follows it.
    :type transient: bool
    :param target: Console to draw the bar on.
    :type target: rich.console.Console
    :return: A bar to run the sweep inside, with a task to add to it.
    :rtype: rich.progress.Progress
    """
    return Progress(SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(compact=True),
                    console=target,
                    transient=transient)


def check_corpus(data_folder, target=console):
    """Verify a corpus folder, drawing the sweep and reporting what it found.

    The check runs over every song folder of ``data folder`` under a progress bar — a
    corrupt copy does not wait for its turn in the build to be found — then a table
    lists the problems, errors first. Errors are announced in red, warnings in yellow,
    a sound corpus in green.

    :param data_folder: Directory holding one subdirectory per song.
    :type data_folder: str or pathlib.Path
    :param target: Console to report on.
    :type target: rich.console.Console
    :return: ``(songs, problems)``, as :func:`pitchtrack.corpus.verify_corpus` gives.
    :rtype: tuple[int, list[pitchtrack.corpus.Problem]]
    """
    target.print(f"[bold]checking[/bold] {data_folder}")
    with progress(target=target) as bar:
        task = bar.add_task("verifying songs", total=len(song_folders(data_folder)))
        songs, problems = verify_corpus(data_folder,
                                        on_song=lambda folder: bar.advance(task))

    if problems:
        show(problems_table(problems, title=f"{len(problems)} problems in "
                                            f"{songs} song folders"),
             target=target)
    errors = [problem for problem in problems if problem.severity == ERROR]
    if errors:
        target.print(f"[red]failed[/red] {len(errors)} errors — this corpus cannot "
                     f"be built from")
    elif problems:
        target.print(f"[yellow]passed[/yellow] {len(problems)} warnings")
    else:
        target.print(f"[green]passed[/green] {songs} song folders, no problems")
    return songs, problems


def show(table, target=console):
    """Print a table on the given console.

    :param table: The table to print.
    :type table: rich.table.Table
    :param target: Console to print on.
    :type target: rich.console.Console
    :return: None
    :rtype: None
    """
    target.print(table)
