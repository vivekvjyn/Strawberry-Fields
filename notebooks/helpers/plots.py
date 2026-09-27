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


def display_wide(fig):
    """Show a figure at its full pixel width inside a horizontally scrollable box.

    Notebook outputs shrink wide images to fit the column, which throws away the
    detail of a plot covering a whole recording. Embedding the PNG inline in an
    ``overflow-x: auto`` box keeps every pixel and lets the reader pan sideways.

    :param fig: Figure to display. It is closed afterwards, so the notebook's
        end-of-cell flush does not show it a second time.
    :type fig: matplotlib.figure.Figure
    """
    import base64
    import io

    from IPython.display import HTML, display

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    plt.close(fig)
    width = int(round(fig.get_size_inches()[0] * fig.dpi))
    display(HTML(
        '<div style="overflow-x:auto">'
        f'<img alt="figure" src="data:image/png;base64,'
        f'{base64.b64encode(buf.getvalue()).decode()}" '
        f'style="width:{width}px; max-width:none; display:block">'
        "</div>"
    ))


def plot_f0(ax, times, f0, title=None, color=None, highlight=None, highlight_color="#D55E00",
            xlim=None, **kw):
    """Plot a pitch track (f0 over time) on a log-frequency axis labelled with note names.

    The y axis is cropped to the pitch range of the frames in view (plus a semitone
    either side), so a track spanning several octaves doesn't leave the plot mostly empty.

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
    :param xlim: ``(start, end)`` time span to show; ``None`` shows the whole track.
    :type xlim: tuple[float, float] or None
    :param kw: Extra keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
    """
    if highlight is not None:
        ax.axvspan(*highlight, color=highlight_color, alpha=0.15, zorder=0, label="Matched region")
        ax.legend(loc="upper right")
    ax.plot(times, f0, "-", color=color, **kw)
    ax.set(xlabel="Time (s)", ylabel="Note", title=title)
    ax.set_yscale("log")

    times, f0 = np.asarray(times, dtype=np.float64), np.asarray(f0, dtype=np.float64)
    if xlim is not None:
        ax.set_xlim(*xlim)
        f0 = f0[(times >= xlim[0]) & (times <= xlim[1])]
    _label_notes(ax, f0)


def _label_notes(ax, f0):
    """Crop a log-frequency axis to the given frequencies and label every semitone with its piano key name.

    The range is the 1st-99th percentile of the frequencies, so a few stray
    pitch-tracking errors (typically octave jumps) are cropped rather than stretching
    the axis. Sharps are left unlabelled when the range exceeds two octaves so the
    labels stay legible.

    :param ax: Axes whose y axis is frequency in Hz on a log scale.
    :type ax: matplotlib.axes.Axes
    :param f0: The frequencies in view; sets the axis range and the semitones to label.
    :type f0: numpy.ndarray
    """
    from matplotlib.ticker import NullFormatter, NullLocator

    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size == 0:
        return
    lo, hi = np.percentile(69 + 12 * np.log2(voiced / 440.0), [1, 99])
    ax.set_ylim(440.0 * 2 ** ((lo - 1 - 69) / 12), 440.0 * 2 ** ((hi + 1 - 69) / 12))
    notes = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    ax.set_yticks(440.0 * 2 ** ((notes - 69) / 12))
    ax.set_yticklabels(_note_labels(ax, notes, lambda n: f"{_NOTE_NAMES[n % 12]}{n // 12 - 1}"), fontsize=7)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_labels(ax, semitones, name):
    """Name each semitone, thinning the labels to what fits the axes' height.

    Tries every semitone, then the naturals, then every third, sixth and finally
    twelfth semitone, keeping the first set that fits.

    :param ax: Axes the labels will go on; its height decides how many fit.
    :type ax: matplotlib.axes.Axes
    :param semitones: The semitones being labelled, as integers.
    :type semitones: numpy.ndarray
    :param name: Names one semitone.
    :type name: collections.abc.Callable[[int], str]
    :return: A label per semitone, empty where it's been thinned out.
    :rtype: list[str]
    """
    height_points = ax.get_window_extent().height * 72 / ax.figure.dpi
    fits = max(2, int(height_points // 11))
    naturals = (0, 2, 4, 5, 7, 9, 11)
    for keep in (lambda n: True, lambda n: n % 12 in naturals, lambda n: n % 3 == 0,
                 lambda n: n % 6 == 0, lambda n: n % 12 == 0):
        if sum(1 for n in semitones if keep(n)) <= fits:
            break
    return [name(n) if keep(n) else "" for n in semitones]


def plot_pitch_class_profile(ax, profile, eval_cfg, title=None):
    """Plot a pitch-class salience profile, as built by :func:`utils.to_pitch_class_profile`.

    The y axis covers one octave (0-1200 cents), labelled with note names relative
    to class 0 as C; unlike an absolute-cents salience image it never needs cropping,
    since every profile has the same fixed range regardless of the recording's pitch.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param profile: Profile as returned by :func:`utils.to_pitch_class_profile`.
    :type profile: numpy.ndarray
    :param eval_cfg: Settings with the keys ``hop_seconds`` and ``n_classes``.
    :type eval_cfg: dict
    :param title: Axes title.
    :type title: str or None
    """
    hop, n_classes = eval_cfg["hop_seconds"], eval_cfg["n_classes"]
    bin_cents = 1200.0 / n_classes
    im = ax.imshow(profile, origin="lower", aspect="auto",
                    extent=[0, profile.shape[1] * hop, -bin_cents / 2, 1200.0 + bin_cents / 2])
    ax.set(xlabel="Time (s)", ylabel="Pitch class (0 = C)", title=title)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, label="Salience", pad=0.02)

    classes = np.arange(n_classes)
    ax.set_yticks(classes * bin_cents)
    ax.set_yticklabels(_note_labels(ax, classes, lambda n: _NOTE_NAMES[n % 12]), fontsize=7)
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
    from scipy.stats import gaussian_kde

    fractions = np.asarray(fractions, dtype=np.float64)
    bins = np.linspace(0, 1, 21)
    ax.hist(fractions, bins=bins, color=color, alpha=0.6, edgecolor="white", linewidth=0.6)
    if len(fractions) > 1 and np.ptp(fractions) > 0:
        grid = np.linspace(0, 1, 300)
        # scale the density to the count histogram: counts = density * n * bin width
        ax.plot(grid, gaussian_kde(fractions)(grid) * len(fractions) * (bins[1] - bins[0]), color=color, lw=1.8)
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
    from . import utils

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

    Each distribution is drawn as a density histogram with a Gaussian kernel density
    estimate overlaid.

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
    from scipy.stats import gaussian_kde, mannwhitneyu

    auc = {}
    for ax, (name, costs) in zip(np.atleast_1d(axes), costs_by_name.items()):
        genuine = costs[np.arange(len(costs)), true_cols]
        mask = np.ones_like(costs, dtype=bool)
        mask[np.arange(len(costs)), true_cols] = False
        impostor = costs[mask]
        genuine, impostor = genuine[np.isfinite(genuine)], impostor[np.isfinite(impostor)]

        ax.hist(impostor, bins=40, alpha=0.4, density=True, color=impostor_color, label="Other songs")
        ax.hist(genuine, bins=40, alpha=0.6, density=True, color=genuine_color, label="True song")
        grid = np.linspace(min(genuine.min(), impostor.min()), max(genuine.max(), impostor.max()), 300)
        for sample, color in ((impostor, impostor_color), (genuine, genuine_color)):
            if len(sample) > 1 and np.ptp(sample) > 0:
                ax.plot(grid, gaussian_kde(sample)(grid), color=color, lw=1.8)
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


def plot_ssm(ax, matrix, times, *, title=None, window=None, vmax=None):
    """Plot a self-similarity matrix with its axes in seconds.

    Two regions of the audio that sound alike give a bright off-diagonal block, so
    the repeats in a performance are the off-diagonal structures rather than the
    (removed) diagonal. The mask means the matrix is sparse and the row/column order
    skips silences, so the axis is labelled with the time of each surviving frame.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param matrix: Similarity matrix, over the frames the mask left in.
    :type matrix: numpy.ndarray
    :param times: Time in seconds of each matrix axis element, in matrix order.
    :type times: numpy.ndarray
    :param title: Axes title.
    :type title: str or None
    :param window: ``(start_s, end_s)`` to crop the matrix to, for a legible view of
        one region of a long performance; ``None`` shows the whole matrix.
    :type window: tuple[float, float] or None
    :param vmax: Upper end of the colour scale.
    :type vmax: float or None
    :return: The image, for adding a colorbar.
    :rtype: matplotlib.image.AxesImage
    """
    times = np.asarray(times, dtype=np.float64)
    lo, hi = (0, matrix.shape[0]) if window is None else (
        int(np.searchsorted(times, window[0])), int(np.searchsorted(times, window[1])))
    lo, hi = max(lo, 0), min(hi, matrix.shape[0])
    if hi - lo < 2:  # a window past the end of the matrix just shows all of it
        lo, hi = 0, matrix.shape[0]

    last = min(hi, len(times) - 1)
    im = ax.imshow(matrix[lo:hi, lo:hi], origin="lower", aspect="auto",
                   interpolation="nearest", vmax=vmax,
                   extent=[times[lo], times[last], times[lo], times[last]])
    ax.set(xlabel="Time (s)", ylabel="Time (s)", title=title)
    ax.grid(False)
    return im


def span_windows(spans, pad_seconds):
    """The window of the recording each panel shows, equal in length for all of them.

    Every panel gets the same number of seconds, starting just before its own
    occurrence, so the contours still lie on top of one another while the axis reads
    the recording's own time. The same windows are what the audio of each panel is cut
    from, so plot and sound cover the same seconds.

    :param spans: ``(start_s, end_s)`` of each occurrence.
    :type spans: collections.abc.Sequence[tuple[float, float]]
    :param pad_seconds: Context drawn either side of each occurrence, in seconds.
    :type pad_seconds: float
    :return: ``(left_s, right_s)`` for each span, in the same order.
    :rtype: list[tuple[float, float]]
    """
    spans = list(spans)
    if not spans:
        return []
    width = max(end - start for start, end in spans) + 2 * pad_seconds
    return [(start - pad_seconds, start - pad_seconds + width) for start, _ in spans]


def _draw_span(ax, times, f0, span, window, color, label=None, prefix=""):
    """Draw one phrase occurrence as a coloured contour over grey context.

    The occurrence itself is coloured, everything else in ``window`` grey, and the
    window is the same length in every panel of a figure, so two panels of the same
    phrase are still drawn on the same scale and start in the same place — only the
    axis moves, and it reads the recording's own seconds.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``nan`` or ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param span: ``(start_s, end_s)`` of the occurrence.
    :type span: tuple[float, float]
    :param window: ``(left_s, right_s)`` of the panel, from :func:`span_windows`.
    :type window: tuple[float, float]
    :param color: Line color; ``None`` uses the axes' current color cycle.
    :type color: str or None
    :param label: Legend label for the occurrence line.
    :type label: str or None
    :param prefix: Words in front of the span in the panel title, e.g. ``"kept "``.
    :type prefix: str
    """
    start, end = span
    left, right = window
    shown = np.flatnonzero((times >= left) & (times <= right))
    if not len(shown):
        ax.set_xlim(left, right)
        return
    voiced = np.where(f0[shown] > 0, f0[shown], np.nan)
    ax.plot(times[shown], voiced, color="0.7", lw=1.0, zorder=1)
    inside = (times[shown] >= start) & (times[shown] <= end)
    ax.plot(times[shown][inside], voiced[inside], color=color, lw=1.6, zorder=2,
            label=label)
    ax.set_xlim(left, right)
    ax.set_title(f"{prefix}{start:.0f}-{end:.0f} s", fontsize=9)
    ax.set_ylabel("Note")


def _crop_to_pitch(axes, times, f0, spans, tonic=None, pad_seconds=1.5):
    """Crop every axes' y range to the pitch visible in ``spans``, then label the svaras.

    Cropping first is what lets the ticks be placed knowing which ones are on screen; an
    axis running to the highest note in the context would bunch every label into one
    corner.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    seen = np.concatenate([f0[(times >= s - pad_seconds) & (times <= e + pad_seconds)]
                           for s, e in spans]) if len(spans) else np.zeros(0)
    seen = seen[np.isfinite(seen) & (seen > 0)]
    for ax in np.atleast_1d(axes).ravel():
        if seen.size:
            lo, hi = np.percentile(seen, [1, 99])
            ax.set_yscale("log")
            ax.set_ylim(lo * 2 ** (-4 / 12), hi * 2 ** (4 / 12))
        if tonic is not None:
            _label_cents(ax, tonic)


def plot_occurrences(axes, times, f0, spans, *, tonic=None, pad_seconds=1.5, title=None,
                     color=None, label=None):
    """Draw every occurrence of one repeated phrase, one per panel, on equal windows.

    Putting the occurrences in separate panels, each showing the same number of seconds
    and starting the same distance before its own occurrence, is what makes a repeat
    visible: if the contours lie on top of each other, the phrase really is the same
    phrase and not two similar ones. The axis reads the recording's own time, so each
    panel's numbers are where those seconds are in the track. The grey lines are the
    surrounding melodic context.

    :param axes: One axes per occurrence.
    :type axes: collections.abc.Sequence[matplotlib.axes.Axes]
    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``nan`` or ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param spans: ``(start_s, end_s)`` of each occurrence, in time order.
    :type spans: collections.abc.Sequence[tuple[float, float]]
    :param tonic: Frequency of sā in Hz; converts the axis to cents relative to it.
    :type tonic: float or None
    :param pad_seconds: Context drawn either side of each occurrence, in seconds.
    :type pad_seconds: float
    :param title: Figure title.
    :type title: str or None
    :param color: Line color; ``None`` uses the axes' current color cycle.
    :type color: str or None
    :param label: Legend label for the occurrence line.
    :type label: str or None
    """
    panels = np.atleast_1d(axes)
    windows = span_windows(spans, pad_seconds)
    for rank, (ax, span, window) in enumerate(zip(panels, spans, windows)):
        _draw_span(ax, times, f0, span, window, color, label if rank == 0 else None)
        if rank == len(spans) - 1:
            ax.set_xlabel("Time (s)")
    if title:
        panels[0].figure.suptitle(title, y=1.02)

    _crop_to_pitch(panels, times, f0, spans, tonic=tonic, pad_seconds=pad_seconds)


def plot_duplicates(axes, times, f0, pairs, *, tonic=None, pad_seconds=1.5, title=None,
                    kept_color=None, dropped_color=None):
    """Draw each repeat beside the statement of the same phrase that is being kept.

    A repeat is only worth cutting if it *is* the phrase said again, and the two are hard
    to judge one after another across a long recording. Side by side — the kept statement
    on the left, the repeat on the right, both on an equal window reading the recording's
    own time — the contours can be compared directly, and a false positive shows up
    immediately as two panels whose lines do not follow each other.

    :param axes: ``(n, 2)`` axes; row ``i`` shows pair ``i``.
    :type axes: numpy.ndarray
    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``nan`` or ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param pairs: ``[((kept_start, kept_end), (removed_start, removed_end)), ...]``.
    :type pairs: collections.abc.Sequence[tuple[tuple[float, float], tuple[float, float]]]
    :param tonic: Frequency of sā in Hz; converts the axis to svara names relative to it.
    :type tonic: float or None
    :param pad_seconds: Context drawn either side of each occurrence, in seconds.
    :type pad_seconds: float
    :param title: Figure title.
    :type title: str or None
    :param kept_color: Color of the statement kept.
    :type kept_color: str or None
    :param dropped_color: Color of the statement being removed.
    :type dropped_color: str or None
    """
    pairs = list(pairs)
    grid = np.asarray(axes, dtype=object).reshape(len(pairs), 2)
    spans = [span for pair in pairs for span in pair]
    windows = span_windows(spans, pad_seconds)
    for row, ((kept, removed), axes_row) in enumerate(zip(pairs, grid)):
        _draw_span(axes_row[0], times, f0, kept, windows[2 * row], kept_color,
                   prefix="kept ")
        _draw_span(axes_row[1], times, f0, removed, windows[2 * row + 1], dropped_color,
                   prefix="removed ")
        if row == len(pairs) - 1:
            for ax in axes_row:
                ax.set_xlabel("Time (s)")
    if title:
        grid[0, 0].figure.suptitle(title, y=1.02)

    _crop_to_pitch(grid, times, f0, spans, tonic=tonic, pad_seconds=pad_seconds)


def _label_cents(ax, tonic, span=1200, max_labels=4):
    """Label a Hz log axis with the svaras of the raga.

    Only the svaras inside the current y range are labelled, and if that still leaves
    more than ``max_labels`` of them they are thinned, because these panels are only an
    inch or so tall and seven labels on that height overlap into a smudge.

    :param ax: Axes whose y axis is frequency in Hz on a log scale.
    :type ax: matplotlib.axes.Axes
    :param tonic: Frequency of Sa in Hz.
    :type tonic: float
    :param span: Cents the ticks should span, from Sa up to the Sa an octave above.
    :type span: float
    :param max_labels: Most labels to draw before thinning.
    :type max_labels: int
    """
    from matplotlib.ticker import NullFormatter, NullLocator

    names = ["Sa", "Ri", "Ga", "Ma", "Pa", "Dha", "Ni"]
    cents = [0, 200, 400, 500, 700, 900, 1100]
    # A svara position is a distance in cents, so it has to be turned into a frequency
    # before it can go on a Hz axis. Taking log2 of the cents directly would be a unit
    # error, and the zero entry would make the tick -inf and the axis vanish. Pitch is
    # heard logarithmically, so the axis is switched to log to match the ticks.
    pairs = [(tonic * 2.0 ** (c / 1200.0), n) for n, c in zip(names, cents) if c <= span]

    ax.set_yscale("log")
    bottom, top = ax.get_ylim()
    visible = [(f, n) for f, n in pairs if bottom <= f <= top]
    if not visible:  # the octave is off the top of the view; fall back to the whole set
        visible = pairs
    stride = max(1, -(-len(visible) // max_labels))
    visible = visible[::stride]

    ax.set_yticks([f for f, _ in visible])
    ax.set_yticklabels([n for _, n in visible], fontsize=8)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_deduplicated_track(ax, times, f0, duplicates, *, kept=None, title=None,
                            keep_color=None, kept_color="#009E73",
                            dropped_color="#999999", xlim=None):
    """Plot a pitch track with the kept statements of each repeated phrase highlighted
    in green and the discarded repeats greyed out.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``nan`` or ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param duplicates: Boolean per frame; ``True`` where the frame is a discarded repeat.
    :type duplicates: numpy.ndarray
    :param kept: ``(start_s, end_s)`` spans of the kept statements, drawn in
        ``kept_color``.
    :type kept: collections.abc.Sequence[tuple[float, float]] or None
    :param title: Axes title.
    :type title: str or None
    :param keep_color: Color of the retained contour; ``None`` uses the axes' cycle.
    :type keep_color: str or None
    :param dropped_color: Color of the discarded repeats.
    :type dropped_color: str
    :param xlim: ``(start, end)`` time span to show; ``None`` shows the whole track.
    :type xlim: tuple[float, float] or None
    """
    times, f0 = np.asarray(times), np.asarray(f0)
    duplicates = np.asarray(duplicates, dtype=bool)
    pitch = np.where(f0 > 0, f0, np.nan)
    if duplicates.any():
        ax.plot(times, np.where(duplicates, pitch, np.nan), "-", color=dropped_color, lw=1.2,
                zorder=1, label="Removed repeat")
    labelled = False
    for start, end in kept or []:
        sel = (times >= start) & (times < end)
        ax.plot(times[sel], pitch[sel], "-", color=kept_color, lw=2.6, zorder=3,
                label="Kept statement of a repeated phrase" if not labelled else None)
        labelled = True
    ax.plot(times, np.where(duplicates, np.nan, pitch), "-", color=keep_color, zorder=2,
            label="Kept")
    ax.set(xlabel="Time (s)", ylabel="Note", title=title)
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    _label_notes(ax, pitch)


def plot_discard_overview(ax, summary, *, top=20, title=None, color=None):
    """Plot the fraction of each song discarded as a repeat, worst first.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param summary: Corpus summary from :func:`phrases.summary_table`.
    :type summary: pandas.DataFrame
    :param top: How many songs to show.
    :type top: int
    :param title: Axes title.
    :type title: str or None
    :param color: Bar color; ``None`` uses the axes' current color cycle.
    :type color: str or None
    """
    top_rows = summary.head(top).iloc[::-1]
    labels = [f"{r.title} — {r.raga}" for r in top_rows.itertuples()]
    ax.barh(np.arange(len(top_rows)), top_rows.discarded_fraction, color=color)
    ax.set(yticks=np.arange(len(top_rows)), yticklabels=labels,
           xlabel="Fraction of the song discarded",
           title=title or f"Most repeated {top} songs")
    ax.tick_params(axis="y", labelsize=8)

