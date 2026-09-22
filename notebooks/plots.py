import matplotlib.pyplot as plt
import numpy as np


def set_style(palette):
    """Apply a clean, light, academic-paper matplotlib style.

    Serif type, a light grid with the top/right spines removed, and sizing suited
    to print figures. Call this once near the top of a notebook, before creating
    any figures.

    :param palette: Categorical colors used as the default color cycle.
    :type palette: collections.abc.Sequence[str]
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=list(palette)),
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "image.cmap": "magma",
    })


def plot_f0(ax, times, f0, title=None, color=None, highlight=None, highlight_color="#D55E00", **kw):
    """Plot a pitch track (f0 over time) on a log-frequency axis.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Frequency of each frame in Hz.
    :type f0: numpy.ndarray
    :param title: Axes title.
    :type title: str or None
    :param color: Marker color; ``None`` uses the axes' current color cycle.
    :type color: str or None
    :param highlight: ``(start, end)`` time span to shade, e.g. the region a query
        matched within a reference. ``None`` draws no shading.
    :type highlight: tuple[float, float] or None
    :param highlight_color: Color of the shaded ``highlight`` span.
    :type highlight_color: str
    :param kw: Extra keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
    """
    if highlight is not None:
        ax.axvspan(*highlight, color=highlight_color, alpha=0.15, zorder=0, label="Matched region")
        ax.legend(loc="upper right")
    ax.plot(times, f0, "-", color=color, **kw)
    ax.set(xlabel="Time (s)", ylabel="$f_0$ (Hz)", title=title)
    ax.set_yscale("log")


def plot_contour(ax, contour, eval_cfg, title=None, color=None, highlight=None, highlight_color="#D55E00", **kw):
    """Plot a key-normalised cents contour, as built by :func:`utils.to_contour`.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param contour: Pitch contour in cents, ``nan`` where unvoiced.
    :type contour: numpy.ndarray
    :param eval_cfg: Settings with the key ``hop_seconds``.
    :type eval_cfg: dict
    :param title: Axes title.
    :type title: str or None
    :param color: Line color; ``None`` uses the axes' current color cycle.
    :type color: str or None
    :param highlight: ``(start, end)`` time span to shade.
    :type highlight: tuple[float, float] or None
    :param highlight_color: Color of the shaded ``highlight`` span.
    :type highlight_color: str
    :param kw: Extra keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
    """
    times = np.arange(len(contour)) * eval_cfg["hop_seconds"]
    if highlight is not None:
        ax.axvspan(*highlight, color=highlight_color, alpha=0.15, zorder=0, label="Matched region")
        ax.legend(loc="upper right")
    ax.plot(times, contour, "-", color=color, **kw)
    ax.set(xlabel="Time (s)", ylabel="Cents", title=title)


def plot_salience_image(ax, image, eval_cfg, title=None):
    """Plot a pitch-salience image with time and cents axes.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param image: Image as returned by :func:`utils.to_salience_image`.
    :type image: numpy.ndarray
    :param eval_cfg: Settings with the keys ``hop_seconds``, ``bin_cents`` and ``range_cents``.
    :type eval_cfg: dict
    :param title: Axes title.
    :type title: str or None
    """
    hop, rc = eval_cfg["hop_seconds"], eval_cfg["range_cents"]
    im = ax.imshow(image, origin="lower", aspect="auto",
                    extent=[0, image.shape[1] * hop, -rc, rc])
    ax.set(xlabel="Time (s)", ylabel="Cents", title=title)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, label="Salience", pad=0.02)
    return im


def plot_voiced_fraction_hist(ax, fractions, title=None, color=None):
    """Plot a histogram of the fraction of voiced frames per track.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param fractions: Voiced-frame fraction (0 to 1) of each track.
    :type fractions: collections.abc.Sequence[float]
    :param title: Axes title.
    :type title: str or None
    :param color: Bar color; ``None`` uses the axes' current color cycle.
    :type color: str or None
    """
    ax.hist(fractions, bins=np.linspace(0, 1, 21), color=color, edgecolor="white", linewidth=0.6)
    ax.set(xlabel="Voiced-frame fraction", ylabel="Tracks", title=title, xlim=(0, 1))


def plot_top_k_bar(ax, table, ks=("top-1", "top-3", "top-5", "top-10")):
    """Plot top-k accuracy as grouped bars, one group per representation.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param table: Metrics table indexed by representation name, as built by
        :func:`utils.summarise`.
    :type table: pandas.DataFrame
    :param ks: Columns of ``table`` to plot.
    :type ks: collections.abc.Sequence[str]
    """
    table[list(ks)].plot.bar(ax=ax, rot=25, width=0.75)
    ax.set(title="Top-$k$ accuracy", ylabel="Fraction of queries", ylim=(0, 1))
    ax.legend(title=None)


def plot_cmc_curve(ax, ranks_by_name, n_candidates):
    """Plot Cumulative Match Characteristic (CMC) curves, one line per representation.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param ranks_by_name: Representation name to its array of true-song ranks.
    :type ranks_by_name: dict[str, numpy.ndarray]
    :param n_candidates: Total number of candidates a query is ranked against.
    :type n_candidates: int
    """
    import utils

    for name, ranks in ranks_by_name.items():
        ks, acc = utils.cmc_curve(ranks, n_candidates)
        ax.plot(ks, acc, label=name)
    ax.axhline(1 / n_candidates, color="gray", ls="--", lw=1, label="Chance at $k=1$")
    ax.set(title="CMC curve", xlabel="$k$ (candidates returned)",
           ylabel="Accuracy", ylim=(0, 1.02))
    ax.legend()


def plot_mrr_bar(ax, table):
    """Plot mean reciprocal rank as a bar per representation.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param table: Metrics table indexed by representation name, with an ``MRR`` column.
    :type table: pandas.DataFrame
    """
    table[["MRR"]].plot.bar(ax=ax, rot=25, legend=False)
    ax.set(title="Mean reciprocal rank", ylabel="MRR", ylim=(0, 1))


def plot_rank_histogram(ax, ranks_by_name, n_candidates):
    """Plot histograms of the true song's rank, one outline per representation.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param ranks_by_name: Representation name to its array of true-song ranks.
    :type ranks_by_name: dict[str, numpy.ndarray]
    :param n_candidates: Total number of candidates a query is ranked against.
    :type n_candidates: int
    """
    bins = np.arange(1, n_candidates + 2)
    for name, ranks in ranks_by_name.items():
        ax.hist(ranks, bins=bins, histtype="step", label=name, lw=1.8)
    ax.set(title="Rank of the true song", xlabel="Rank", ylabel="Queries")
    ax.legend()


def plot_timing_bar(ax, table):
    """Plot per-query matching time as a bar per representation.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param table: Metrics table indexed by representation name, with a
        ``seconds / query`` column.
    :type table: pandas.DataFrame
    """
    table[["seconds / query"]].plot.bar(ax=ax, rot=25, legend=False)
    ax.set(title="Matching time per query", ylabel="Seconds")


def plot_cost_distributions(axes, costs_by_name, true_cols, genuine_color, impostor_color):
    """Plot genuine-vs-impostor DTW cost distributions, one axes per representation.

    :param axes: One axes per entry in ``costs_by_name``.
    :type axes: collections.abc.Sequence[matplotlib.axes.Axes]
    :param costs_by_name: Representation name to its full query-by-candidate cost matrix.
    :type costs_by_name: dict[str, numpy.ndarray]
    :param true_cols: Column index of the true song for each query.
    :type true_cols: numpy.ndarray
    :param genuine_color: Color for the true-song cost distribution.
    :type genuine_color: str
    :param impostor_color: Color for the other-songs cost distribution.
    :type impostor_color: str
    :return: Representation name to its separation AUC (probability the true song's
        cost is lower than an impostor's).
    :rtype: dict[str, float]
    """
    from scipy.stats import mannwhitneyu

    auc = {}
    for ax, (name, costs) in zip(np.atleast_1d(axes), costs_by_name.items()):
        genuine = costs[np.arange(len(costs)), true_cols]
        mask = np.ones_like(costs, dtype=bool)
        mask[np.arange(len(costs)), true_cols] = False
        impostor = costs[mask]

        ax.hist(impostor, bins=40, alpha=0.6, density=True, color=impostor_color, label="Other songs")
        ax.hist(genuine, bins=40, alpha=0.8, density=True, color=genuine_color, label="True song")
        ax.set(title=name, xlabel="Normalised DTW cost", ylabel="Density")
        ax.legend()

        u = mannwhitneyu(genuine, impostor, alternative="less").statistic
        auc[name] = 1 - u / (len(genuine) * len(impostor))
    return auc


def plot_paired_ranks(ax_scatter, ax_bar, ranks_a, ranks_b, label_a, label_b, n_candidates,
                      point_color, better_color, worse_color, tie_color):
    """Compare the true song's rank under two representations, query by query.

    :param ax_scatter: Axes for the rank-vs-rank scatter plot.
    :type ax_scatter: matplotlib.axes.Axes
    :param ax_bar: Axes for the win/tie/loss bar chart.
    :type ax_bar: matplotlib.axes.Axes
    :param ranks_a: True-song ranks under the first representation.
    :type ranks_a: numpy.ndarray
    :param ranks_b: True-song ranks under the second representation.
    :type ranks_b: numpy.ndarray
    :param label_a: Name of the first representation.
    :type label_a: str
    :param label_b: Name of the second representation.
    :type label_b: str
    :param n_candidates: Total number of candidates a query is ranked against.
    :type n_candidates: int
    :param point_color: Color of the scatter points.
    :type point_color: str
    :param better_color: Bar color for queries where ``label_b`` ranks higher.
    :type better_color: str
    :param worse_color: Bar color for queries where ``label_a`` ranks higher.
    :type worse_color: str
    :param tie_color: Bar color for queries where both rank the true song the same.
    :type tie_color: str
    """
    lim = (0.8, n_candidates + 1)
    ax_scatter.scatter(ranks_a, ranks_b, alpha=0.6, s=22, color=point_color, edgecolor="white", linewidth=0.4)
    ax_scatter.plot(lim, lim, "--", color="gray", lw=1)
    ax_scatter.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
                   xlabel=f"Rank, {label_a}", ylabel=f"Rank, {label_b}",
                   title="Rank of the true song per query")

    better = int((ranks_b < ranks_a).sum())
    worse = int((ranks_b > ranks_a).sum())
    same = int((ranks_b == ranks_a).sum())
    ax_bar.bar([f"{label_b}\nbetter", "Tie", f"{label_a}\nbetter"], [better, same, worse],
               color=[better_color, tie_color, worse_color])
    ax_bar.set(title="Which representation ranks the true song higher", ylabel="Queries")
