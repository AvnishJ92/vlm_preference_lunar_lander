"""Figures and a summary table from finished (or still running) runs.

    python -m prefrl.plot <run dir>/outputs        # reads <condition>/seed_<n>/, writes plots/

Curves are the mean over seeds with a +-1 std band. Every condition keeps one fixed colour in every figure.
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


def load(root, filename):
    """{condition: {seed: DataFrame}} for one per-run csv, in CONDITIONS order; runs without the file are skipped."""
    found = {}
    for path in sorted(root.glob(f"*/seed_*/{filename}")):
        condition, seed = path.parts[-3], path.parts[-2].removeprefix("seed_")
        if seed.isdigit():  # not a set-aside "seed_0.incomplete-..." directory
            found.setdefault(condition, {})[int(seed)] = pd.read_csv(path)
    return {c: found[c] for c in CONDITIONS if c in found}


def curves(ax, runs, x, y, title, ylabel=None, label_ends=False):
    """One mean +- std line per condition. Returns the legend handles."""
    ends = []
    for condition, by_seed in runs.items():
        name, color = CONDITIONS[condition]
        stat = pd.concat(by_seed.values()).groupby(x)[y].agg(["mean", "std"]).dropna(subset="mean")
        if stat.empty:
            continue
        spread = stat["std"].fillna(0)
        ax.fill_between(stat.index, stat["mean"] - spread, stat["mean"] + spread, color=color, alpha=0.15, lw=0)
        ax.plot(stat.index, stat["mean"], color=color, lw=2, label=f"{name} ({len(by_seed)} seeds)")
        ends.append((stat["mean"].iloc[-1], name))
    ax.set_title(title)
    ax.set_xlabel("environment timesteps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(MILLIONS)
    if label_ends:  # name each line at its right end, pushed apart so the names never overlap
        low, high = ax.get_ylim()
        y_text = -np.inf
        for y_end, name in sorted(ends):
            y_text = max(y_end, y_text + 0.055 * (high - low))
            ax.annotate(name, (1.0, y_text), xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, annotation_clip=False)  # fmt: skip
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
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)


def learning_curves(evals, progress, out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    handles = curves(ax, evals, "t", "mean", "Evaluation return (deterministic policy, true reward)",
                     "mean return over 20 fixed evaluation episodes", label_ends=True)  # fmt: skip
    reference(ax, 200, "solved (200)")
    save(fig, out, "learning_curves.png", handles)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    handles = curves(ax, progress, "time/total_timesteps", "rollout/ep_rew_mean",
                     "Training return (stochastic policy, true reward)",
                     "mean return of the last 100 training episodes", label_ends=True)  # fmt: skip
    reference(ax, 200, "solved (200)")
    save(fig, out, "training_return.png", handles)

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
        reference(ax, 200, "solved")
    axes[0, 0].set_ylabel("evaluation return (thin: one seed, thick: mean)")
    save(fig, out, "learning_curves_per_seed.png")


def final_performance(summaries, out):
    """One bar per condition (mean over seeds of the final 100-episode evaluation), one dot per seed."""
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(summaries), 4.2))
    for i, (condition, by_seed) in enumerate(summaries.items()):
        name, color = CONDITIONS[condition]
        finals = np.array([s["final_eval"]["mean"] for s in by_seed.values()])
        ax.bar(i, finals.mean(), width=0.3, color=color, alpha=0.35)
        ax.scatter(i + np.linspace(-0.08, 0.08, len(finals)), finals, s=42, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.annotate(f"{finals.mean():.0f} ± {finals.std():.0f}", (i, max(finals.max(), 0)), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)  # fmt: skip
    ax.set_xticks(range(len(summaries)), [CONDITIONS[c][0].replace(" preferences", "\npreferences") for c in summaries])
    ax.set_title("Final return (100 evaluation episodes per seed)")
    ax.set_ylabel("mean return; dot = one seed, bar = mean over seeds")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)  # room for the value labels under the title
    reference(ax, 200, "solved (200)", outside=True)
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
        vlm = [s["vlm"] for s in runs if s.get("vlm", {}).get("hard")]
        rows.append({
            "condition": CONDITIONS[condition][0], "seeds": len(runs),
            "final_return_mean": finals.mean().round(1), "final_return_std_over_seeds": finals.std().round(1),
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
