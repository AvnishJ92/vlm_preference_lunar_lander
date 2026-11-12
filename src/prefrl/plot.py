"""Figures and a summary table from finished (or still running) runs.

    python -m prefrl.plot <run dir>/outputs        # reads <condition>/seed_<n>/, writes plots/

Curves are the mean over seeds with a 95% confidence band. Every condition keeps one fixed colour in every figure.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Label and colour per condition. The four hues, in this order, pass a colour-vision-deficiency check; the two light
# ones (aqua, yellow) are below 3:1 contrast on the surface, hence the direct labels and the summary table.
CONDITIONS = {
    "pure_rl": ("RL on true reward", "#2a78d6"),
    "oracle": ("Oracle preferences", "#eb6834"),
    "vlm": ("VLM preferences", "#1baf7a"),
    "mixed": ("Oracle + VLM preferences", "#eda100"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 160,
    "savefig.bbox": "tight", "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 0,
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})  # fmt: skip
MILLIONS = FuncFormatter(lambda v, _: f"{v / 1e6:g}M")
# Two-sided 95% Student-t critical value by number of seeds (scipy is not a dependency); 1.96 beyond the table.
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
CI_NOTE = "Line: mean over seeds. Band: 95% confidence interval of that mean (Student t)."
ZOOM = (-600, 330)  # return range shown in zoomed panels: a run that flies away scores thousands below zero


def ci95(std, n):
    """Half-width of the 95% confidence interval of a mean of n seeds with sample std `std`; 0 for a single seed."""
    n = np.asarray(n, float)
    t = np.vectorize(lambda k: T975.get(int(k), 1.96))(n)
    return np.nan_to_num(t * np.asarray(std, float) / np.sqrt(n))


def load(root, filename):
    """{condition: {seed: DataFrame}} for one per-run csv, in CONDITIONS order; runs without the file are skipped."""
    found = {}
    for path in sorted(root.glob(f"*/seed_*/{filename}")):
        condition, seed = path.parts[-3], path.parts[-2].removeprefix("seed_")
        if seed.isdigit():  # not a set-aside "seed_0.incomplete-..." directory
            found.setdefault(condition, {})[int(seed)] = pd.read_csv(path)
    return {c: found[c] for c in CONDITIONS if c in found}


def curves(ax, runs, x, y, title, ylabel=None, label_ends=False, ylim=None):
    """One line per condition: mean over seeds with its 95% confidence band. Returns the legend handles."""
    ends = []
    for condition, by_seed in runs.items():
        name, color = CONDITIONS[condition]
        stat = pd.concat(by_seed.values()).groupby(x)[y].agg(["mean", "std", "count"]).dropna(subset="mean")
        if stat.empty:
            continue
        spread = ci95(stat["std"], stat["count"])
        ax.fill_between(stat.index, stat["mean"] - spread, stat["mean"] + spread, color=color, alpha=0.15, lw=0)
        ax.plot(stat.index, stat["mean"], color=color, lw=2, label=f"{name} ({len(by_seed)} seeds)")
        ends.append((stat["mean"].iloc[-1], name))
    ax.set_title(title)
    ax.set_xlabel("environment timesteps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(MILLIONS)
    if ylim:
        ax.set_ylim(*ylim)
    if label_ends:  # name each line at its right end, pushed apart so the names never overlap
        low, high = ax.get_ylim()
        y_text = -np.inf
        for y_end, name in sorted(ends):
            y_text = max(min(max(y_end, low), high), y_text + 0.055 * (high - low))
            ax.annotate(name + (" (below the axis)" if y_end < low else ""), (1.0, y_text),
                        xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, annotation_clip=False)  # fmt: skip
    return ax.get_legend_handles_labels()


def reference(ax, y, text, outside=False):
    """A labelled horizontal reference line; `outside` puts the label in the right margin, clear of the marks."""
    ax.axhline(y, color=MUTED, lw=0.8)
    where = dict(xy=(1.0, y), xytext=(6, 0), va="center") if outside else dict(xy=(0.0, y), xytext=(4, 3))
    ax.annotate(text, xycoords=("axes fraction", "data"), textcoords="offset points", fontsize=8.5, color=MUTED,
                annotation_clip=False, **where)  # fmt: skip


def save(fig, out, name, handles=None):
    if handles and handles[0]:
        fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles[0]))
        fig.text(0.5, -0.09, CI_NOTE, ha="center", fontsize=8.5, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)


def returns_over_training(runs, x, y, title, ylabel, out, name):
    """The same curves twice: full range, and zoomed so that runs which fly away do not flatten the others."""
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"wspace": 0.2})
    handles = curves(full, runs, x, y, f"{title}: full range", ylabel)
    curves(zoom, runs, x, y, f"{title}: zoomed to {ZOOM[0]} .. {ZOOM[1]}", label_ends=True, ylim=ZOOM)
    for ax in (full, zoom):
        reference(ax, 200, "solved (200)")
    save(fig, out, name, handles)


def learning_curves(evals, progress, out):
    returns_over_training(evals, "t", "mean", "Evaluation return", "mean return over 20 fixed evaluation episodes\n"
                          "(deterministic policy, true reward)", out, "learning_curves.png")  # fmt: skip
    returns_over_training(progress, "time/total_timesteps", "rollout/ep_rew_mean", "Training return",
                          "mean return of the last 100 training episodes\n(stochastic policy, true reward)", out,
                          "training_return.png")  # fmt: skip

    fig, axes = plt.subplots(1, len(evals), figsize=(3.4 * len(evals), 3.6), sharey=True, squeeze=False)
    for ax, (condition, by_seed) in zip(axes[0], evals.items()):
        name, color = CONDITIONS[condition]
        for df in by_seed.values():
            ax.plot(df["t"], df["mean"], color=color, lw=1, alpha=0.45)
        mean = pd.concat(by_seed.values()).groupby("t")["mean"].mean()
        ax.plot(mean.index, mean, color=color, lw=2)
        ax.set_title(name)
        ax.set_xlabel("environment timesteps")
        ax.xaxis.set_major_formatter(MILLIONS)
        ax.set_ylim(*ZOOM)
        reference(ax, 200, "solved")
    axes[0, 0].set_ylabel("evaluation return (thin: one seed, thick: mean)")
    fig.text(0.5, -0.04, f"Axis cut at {ZOOM[0]}: lower returns are off the bottom (full range in learning_curves.png).",
             ha="center", fontsize=8.5, color=MUTED)  # fmt: skip
    save(fig, out, "learning_curves_per_seed.png")


def final_performance(summaries, out):
    """Final 100-episode evaluation per condition: return (left) and share of episodes that landed (right).

    Dot = one seed, bar = mean over seeds, whisker and brackets = 95% confidence interval of the mean.
    """
    fig, (ret, land) = plt.subplots(1, 2, figsize=(3.2 + 2.9 * len(summaries), 4.4), gridspec_kw={"wspace": 0.3})
    floor = ZOOM[0]
    for i, (condition, by_seed) in enumerate(summaries.items()):
        name, color = CONDITIONS[condition]
        dots = dict(s=42, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        xs = i + np.linspace(-0.08, 0.08, len(by_seed))
        for ax, values, fmt in ((ret, [s["final_eval"]["mean"] for s in by_seed.values()], "{:.0f}"),
                                (land, [100 * s["final_eval"]["landed"] for s in by_seed.values()], "{:.0f}%")):  # fmt: skip
            values = np.array(values)
            mean = values.mean()
            half = float(ci95(values.std(ddof=1), len(values))) if len(values) > 1 else 0.0
            low = values < floor if ax is ret else np.zeros(len(values), bool)
            ax.scatter(xs[~low], values[~low], **dots)
            ax.scatter(xs[low], np.full(low.sum(), floor + 30), marker="v", **dots)  # a seed below the axis
            top = max(values[~low].max(initial=0), 0)
            # a share cannot leave 0..100%, so its interval is cut there
            lo, hi = (mean - half, mean + half) if ax is ret else (max(mean - half, 0), min(mean + half, 100))
            if not (ax is ret and mean < floor):  # a bar cut off by the axis would misstate the mean
                ax.bar(i, mean, width=0.3, color=color, alpha=0.35)
                ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=INK, lw=1.2, capsize=4, zorder=2)
                top = max(top, min(hi, ZOOM[1] if ax is ret else 100))
            text = f"{fmt.format(mean)}  [{fmt.format(lo)}, {fmt.format(hi)}]"
            ax.annotate(text, (i, top), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    labels = [CONDITIONS[c][0].replace(" preferences", "\npreferences") for c in summaries]
    for ax in (ret, land):
        ax.set_xticks(range(len(summaries)), labels)
        ax.grid(axis="x", visible=False)
    ret.set_title("Final return")
    ret.set_ylabel("mean return over 100 episodes")
    ret.set_ylim(floor, ZOOM[1] + 90)
    reference(ret, 200, "solved (200)", outside=True)
    land.set_title("Successful landings")
    land.set_ylabel("share of the 100 final episodes")
    land.set_ylim(0, 118)
    land.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    fig.text(0.5, -0.06, "Dot: one seed (triangle: below the axis). Bar: mean over seeds. Whisker and [ ]: 95% confidence "
             "interval of the mean (Student t).", ha="center", fontsize=8.5, color=MUTED)  # fmt: skip
    save(fig, out, "final_performance.png")


def reward_learning(rounds, out):
    panels = [
        ("pred_oracle_acc", "Predictor vs oracle on unseen clip pairs", "share of pairs ranked like the oracle", 0.5),
        ("rhat_true_corr", "Predicted vs true reward", "Pearson correlation per step", None),
        ("labels", "Preference labels collected", "labels in the database", None),
        ("val_loss", "Predictor validation loss", "cross-entropy (chance = 0.69)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    for ax, (column, title, ylabel, chance) in zip(axes.flat, panels):
        handles = curves(ax, rounds, "t", column, title, ylabel)
        if chance is not None:
            reference(ax, chance, "chance")
    save(fig, out, "reward_predictor.png", handles)

    with_vlm = {}
    for condition, by_seed in rounds.items():
        used = {s: df.assign(labelled=df.vlm_labelled / df.vlm_asked, tied=df.vlm_ties / df.vlm_asked)
                for s, df in by_seed.items() if df.vlm_asked.iloc[-1] > 0}  # fmt: skip
        if used:
            with_vlm[condition] = used
    if not with_vlm:
        return
    panels = [
        ("vlm_oracle_agree", "VLM agrees with the oracle", "share of its clear-cut labels, cumulative", 0.5),
        ("labelled", "VLM queries that produced a label", "share of VLM queries, cumulative", None),
        ("tied", "VLM queries answered 'tie'", "share of VLM queries, cumulative", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (column, title, ylabel, chance) in zip(axes, panels):
        handles = curves(ax, with_vlm, "t", column, title, ylabel)
        ax.set_ylim(0, 1)
        if chance is not None:
            reference(ax, chance, "chance")
    save(fig, out, "vlm_labels.png", handles)


def summary_table(summaries, out):
    rows = []
    for condition, by_seed in summaries.items():
        runs = list(by_seed.values())
        finals = np.array([s["final_eval"]["mean"] for s in runs])
        half = float(ci95(finals.std(ddof=1), len(finals))) if len(finals) > 1 else 0.0
        vlm = [s["vlm"] for s in runs if s.get("vlm", {}).get("hard")]
        rows.append({
            "condition": CONDITIONS[condition][0], "seeds": len(runs),
            "final_return_mean": finals.mean().round(1),
            "ci95_low": (finals.mean() - half).round(1), "ci95_high": (finals.mean() + half).round(1),
            "std_over_seeds": finals.std(ddof=1).round(1) if len(finals) > 1 else 0.0,
            "worst_seed": finals.min().round(1), "best_seed": finals.max().round(1),
            "landed_share": np.mean([s["final_eval"]["landed"] for s in runs]).round(3),
            "queries_asked": np.mean([s.get("asked", 0) for s in runs]),
            "labels_obtained": np.mean([s.get("labels", 0) for s in runs]).round(1),
            "vlm_oracle_agreement": np.mean([v["agree"] / v["hard"] for v in vlm]).round(3) if vlm else None,
        })  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    print(table.to_string(index=False))


def main(root):
    root, out = Path(root), Path(root) / "plots"
    evals = {c: {s: df[df.episodes == df.episodes.iloc[0]] for s, df in by_seed.items()}  # drop the final-eval row
             for c, by_seed in load(root, "eval.csv").items()}  # fmt: skip
    if not evals:
        sys.exit(f"no runs found under {root}")
    learning_curves(evals, load(root, "progress.csv"), out)
    rounds = load(root, "rounds.csv")
    if rounds:
        reward_learning(rounds, out)
    summaries = {}
    for path in sorted(root.glob("*/seed_*/summary.json")):
        if path.parts[-3] in CONDITIONS and path.parts[-2].removeprefix("seed_").isdigit():
            summaries.setdefault(path.parts[-3], {})[path.parts[-2]] = json.loads(path.read_text())
    summaries = {c: summaries[c] for c in CONDITIONS if c in summaries}
    if summaries:
        final_performance(summaries, out)
        summary_table(summaries, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
