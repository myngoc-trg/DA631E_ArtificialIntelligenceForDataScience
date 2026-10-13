"""Visualize the existing Stage 1, 2, and 3 walk-forward search results.

Returns four figures plus the complete, independently recomputed summary.
Selection uses mean validation Sharpe, then worst Sharpe, smaller across-fold
SD, and mean IC, matching modeling.summarize_search.

The inputs must use identical features, eligibility rules, purging, time folds,
portfolio construction, and target definitions. These cannot all be verified
from saved metrics. MSE plots assume predictions and Target share return units;
for rank-score models, set raw-return MSE/Train_MSE to NaN beforehand.

Only NumPy, pandas, and Matplotlib are required.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


STAGE_COLORS = {
    "Stage 1": "#4C78A8",
    "Stage 2": "#8064A2",
    "Stage 3": "#59A14F",
    "Baseline": "#8B8B8B",
}
SELECTED_COLOR = "#D55E00"
TRAIN_COLOR = "#7D8793"
VALID_COLOR = "#0072B2"
HGB = "hist_gradient_boosting"
REFERENCE = "hgb_reference"


def _json_params(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Each params_json value must represent a dictionary.")
    return result


def _prepare(stages):
    frames, parameters = [], {}
    required = {
        "candidate_name", "fold_name", "SpreadSharpeRatio", "MeanDailyIC"
    }
    metric_columns = [
        "SpreadSharpeRatio", "MeanDailyIC", "MSE",
        "Train_SpreadSharpeRatio", "Train_MeanDailyIC", "Train_MSE"
    ]
    for number, result in enumerate(stages, start=1):
        if result is None:
            continue
        frame = (
            result.copy() if isinstance(result, pd.DataFrame)
            else result.fold_results.copy()
        )
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Stage {number} is missing {sorted(missing)}. "
                "Pass per-fold results, not the aggregated summary."
            )
        if frame.empty:
            raise ValueError(f"Stage {number} has no fold results.")
        frame["stage"] = f"Stage {number}"
        if "is_baseline" not in frame:
            frame["is_baseline"] = frame["candidate_name"].str.startswith("baseline_")
        if frame["is_baseline"].isna().any():
            raise ValueError("is_baseline contains missing values.")
        if not frame["is_baseline"].isin([True, False, 0, 1]).all():
            raise ValueError("is_baseline must contain booleans.")
        frame["is_baseline"] = frame["is_baseline"].astype(bool)
        frame.loc[frame["is_baseline"], "stage"] = "Baseline"

        metadata = getattr(result, "candidates", {})
        model_names = {}
        for name, rows in frame.groupby("candidate_name", sort=False):
            candidate = metadata.get(name)
            if candidate is not None:
                params = dict(candidate.params)
                model_names[name] = candidate.model_name
            else:
                params = {}
            if "params_json" in rows:
                row_params = [_json_params(value) for value in rows["params_json"]]
                canonical = {
                    json.dumps(p, sort_keys=True, default=str) for p in row_params
                }
                if len(canonical) != 1:
                    raise ValueError(f"{name}: hyperparameters differ across folds")
                recorded = row_params[0]
                if candidate is not None and recorded != params:
                    raise ValueError(f"{name}: candidate metadata disagrees with params_json")
                params = recorded
            if name in parameters and parameters[name] != params:
                raise ValueError(f"{name}: hyperparameters changed between stages.")
            parameters[name] = params
            if name not in model_names:
                model_names[name] = (
                    HGB if str(name).startswith("hgb_")
                    else "ridge" if str(name).startswith("ridge_")
                    else str(name)
                )
        if "model_name" not in frame:
            frame["model_name"] = frame["candidate_name"].map(model_names)
        for column in metric_columns:
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Stage {number}: a required value is missing.")
        if not np.isfinite(
            frame[["SpreadSharpeRatio", "MeanDailyIC"]].to_numpy()
        ).all():
            raise ValueError(f"Stage {number}: validation Sharpe or IC is not finite.")
        frames.append(frame)

    if not frames:
        raise ValueError("Provide at least one completed stage.")
    folds = pd.concat(frames, ignore_index=True)

    duplicates = folds.duplicated(["candidate_name", "fold_name"], keep=False)
    for (name, fold), rows in folds.loc[duplicates].groupby(
        ["candidate_name", "fold_name"], sort=False
    ):
        values = rows[metric_columns].to_numpy(dtype=float)
        if not np.allclose(values, values[0], rtol=1e-9, atol=1e-12, equal_nan=True):
            raise ValueError(f"{name}/{fold}: conflicting repeated results.")
        if rows["model_name"].nunique() != 1:
            raise ValueError(f"{name}/{fold}: conflicting model names.")
    folds = folds.drop_duplicates(["candidate_name", "fold_name"], keep="first")
    fold_sets = folds.groupby("candidate_name")["fold_name"].agg(
        lambda values: tuple(sorted(values))
    )
    if fold_sets.nunique() != 1:
        raise ValueError("Every candidate must have the same complete set of folds.")

    summary = folds.groupby("candidate_name", sort=False).agg(
        model_name=("model_name", "first"),
        stage=("stage", "first"),
        is_baseline=("is_baseline", "first"),
        folds=("fold_name", "nunique"),
        mean_sharpe=("SpreadSharpeRatio", "mean"),
        worst_sharpe=("SpreadSharpeRatio", "min"),
        std_sharpe=("SpreadSharpeRatio", "std"),
        mean_ic=("MeanDailyIC", "mean"),
        mean_mse=("MSE", "mean"),
        mean_train_sharpe=("Train_SpreadSharpeRatio", "mean"),
        mean_train_ic=("Train_MeanDailyIC", "mean"),
        mean_train_mse=("Train_MSE", "mean"),
    )
    summary["params"] = pd.Series(parameters, dtype=object)
    summary = summary.sort_values(
        ["mean_sharpe", "worst_sharpe", "std_sharpe", "mean_ic"],
        ascending=[False, False, True, False], kind="mergesort",
    )
    return folds.reset_index(drop=True), summary


def _color(name, row, selected):
    if name == selected:
        return SELECTED_COLOR
    if name == REFERENCE or row["is_baseline"]:
        return STAGE_COLORS["Baseline"]
    return STAGE_COLORS[row["stage"]]


def _shortlist(summary, selected, count):
    if count is None:
        return list(summary.index)
    if count < 1:
        raise ValueError("top_n must be positive or None.")
    names = set(summary.loc[~summary["is_baseline"]].head(count).index)
    names.add(selected)
    if REFERENCE in summary.index:
        names.add(REFERENCE)
    ridge = summary.loc[summary["model_name"].eq("ridge")]
    if not ridge.empty:
        names.add(ridge.index[0])
    names.update(summary.loc[summary["is_baseline"]].index)
    return [name for name in summary.index if name in names]


def _empty(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, wrap=True, color="#666666")
    ax.set_axis_off()


def _overview(folds, summary, selected, top_n):
    names = _shortlist(summary, selected, top_n)
    scores = summary.loc[names]
    height = max(6.0, 0.32 * len(names) + 2.6)
    fig, axes = plt.subplots(
        1, 2, figsize=(17, height), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    ax = axes[0]
    y = np.arange(len(names))
    colors = [_color(name, scores.loc[name], selected) for name in names]
    ax.barh(y, scores["mean_sharpe"], color=colors, height=0.65, alpha=0.9)
    ax.scatter(scores["worst_sharpe"], y, s=30, color="#202020", zorder=3)
    ax.set_yticks(y, labels=names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    for tick, name in zip(ax.get_yticklabels(), names):
        if name == selected:
            tick.set_fontweight("bold")
            tick.set_color(SELECTED_COLOR)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_title("Selection criterion: mean validation Sharpe", loc="left")
    ax.set_xlabel("Spread Sharpe ratio")
    ax.grid(axis="x", alpha=0.2)
    low = min(0.0, float(scores[["mean_sharpe", "worst_sharpe"]].min().min()))
    high = max(0.0, float(scores[["mean_sharpe", "worst_sharpe"]].max().max()))
    span = max(high - low, 0.05)
    ax.set_xlim(low - 0.06 * span, high + 0.22 * span)
    for i, value in enumerate(scores["mean_sharpe"]):
        ax.text(value + 0.025 * span, i, f"{value:.4f}",
                va="center", fontsize=9)
    handles = [
        Patch(facecolor=SELECTED_COLOR, label="Selected"),
        Line2D([], [], marker="o", color="#202020", linestyle="",
               label="Worst fold"),
    ]
    handles += [
        Patch(facecolor=STAGE_COLORS[stage], label=stage)
        for stage in ("Stage 1", "Stage 2", "Stage 3")
        if stage in scores["stage"].values
    ]
    handles.append(Patch(facecolor=STAGE_COLORS["Baseline"],
                         label="Reference / momentum"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    table = folds.pivot(
        index="candidate_name", columns="fold_name", values="SpreadSharpeRatio"
    ).loc[names]
    table = table.reindex(sorted(table.columns), axis=1)
    matrix = table.to_numpy(dtype=float)
    vmax = max(float(np.abs(matrix).max()), 1e-6)
    ax = axes[1]
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
    )
    ax.set_xticks(np.arange(len(table.columns)),
                  labels=[str(x).replace("validate_", "") for x in table.columns])
    ax.set_yticks([])
    ax.set_title("Robustness: validation Sharpe in each time fold", loc="left")
    ax.set_xlabel("Validation fold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 * vmax else "#202020",
                    fontsize=9)
    row = names.index(selected)
    ax.add_patch(Rectangle(
        (-0.5, row - 0.5), matrix.shape[1], 1,
        fill=False, edgecolor=SELECTED_COLOR, lw=2,
    ))
    fig.colorbar(im, ax=ax, label="Spread Sharpe ratio", shrink=0.8)
    fig.suptitle(
        f"Walk-forward comparison — highlighted candidate: {selected}\n"
        f"{len(table.columns)} validation folds; black dots are worst folds, "
        "not confidence intervals.",
        fontsize=13,
    )
    return fig


def _heatmap(ax, table, title, value_format):
    data = table.to_numpy(dtype=float)
    if not np.isfinite(data).any():
        _empty(ax, "No finite results for this grid.")
        return
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(table.columns)), labels=table.columns)
    ax.set_yticks(np.arange(len(table.index)), labels=table.index)
    ax.set_xlabel("Maximum leaves per tree")
    ax.set_ylabel("Minimum samples per leaf")
    ax.set_title(title, loc="left")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                rgba = im.cmap(im.norm(data[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, format(data[i, j], value_format),
                        ha="center", va="center",
                        color="white" if luminance < 0.5 else "#202020")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)


def _stage1(summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    rows = summary.loc[
        summary["stage"].eq("Stage 1")
        & summary["model_name"].eq(HGB)
        & (summary.index != REFERENCE)
    ].copy()
    if rows.empty:
        _empty(axes[0], "No Stage 1 HGB shape trials.")
        _empty(axes[1], "No Stage 1 HGB shape trials.")
    else:
        rows["leaves"] = rows["params"].map(lambda p: p["max_leaf_nodes"])
        rows["min_leaf"] = rows["params"].map(lambda p: p["min_samples_leaf"])
        fixed = rows["params"].map(lambda p: json.dumps(
            {k: v for k, v in p.items()
             if k not in ("max_leaf_nodes", "min_samples_leaf")},
            sort_keys=True, default=str,
        ))
        if fixed.nunique() != 1 or rows.duplicated(["leaves", "min_leaf"]).any():
            message = ("Shape trials vary other hyperparameters or repeat cells.\n"
                       "A single controlled grid cannot be drawn.")
            _empty(axes[0], message)
            _empty(axes[1], message)
        else:
            for ax, metric, title, fmt in [
                (axes[0], "mean_sharpe", "Mean validation Sharpe", ".4f"),
                (axes[1], "mean_ic", "Mean daily validation IC", ".5f"),
            ]:
                table = rows.pivot(
                    index="min_leaf", columns="leaves", values=metric
                ).sort_index().sort_index(axis=1)
                _heatmap(ax, table, title, fmt)

    ridge = summary.loc[
        summary["stage"].eq("Stage 1") & summary["model_name"].eq("ridge")
    ].copy()
    if ridge.empty:
        _empty(axes[2], "No Stage 1 Ridge trials.")
    else:
        ridge["alpha"] = ridge["params"].map(lambda p: p["alpha"])
        ridge = ridge.sort_values("alpha")
        if ridge["alpha"].le(0).any():
            axes[2].set_xscale("symlog", linthresh=0.1)
        else:
            axes[2].set_xscale("log")
        axes[2].plot(ridge["alpha"], ridge["mean_sharpe"],
                     marker="o", color=STAGE_COLORS["Stage 1"])
        axes[2].set_xticks(ridge["alpha"],
                           labels=[f"{x:g}" for x in ridge["alpha"]])
        axes[2].set_ylim(
            min(0.0, float(ridge["mean_sharpe"].min()) * 1.1),
            max(0.1, float(ridge["mean_sharpe"].max()) * 1.2),
        )
        axes[2].set_xlabel("Ridge alpha")
        axes[2].set_ylabel("Mean validation Sharpe")
        axes[2].set_title("Ridge sensitivity to regularization", loc="left")
        axes[2].grid(alpha=0.2)
    fig.suptitle(
        "Stage 1: choose tree shape and assess Ridge tuning\n"
        "HGB heatmap cells use the same fixed settings; the reference is excluded.",
        fontsize=13,
    )
    return fig


def _controlled_curves(summary, selected):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    hgb = summary.loc[summary["model_name"].eq(HGB)]
    if hgb.empty:
        for ax in axes:
            _empty(ax, "No HGB trials.")
        return fig
    anchor = selected if selected in hgb.index else hgb.index[0]
    anchor_params = hgb.loc[anchor, "params"]
    if not anchor_params:
        for ax in axes:
            _empty(ax, "Hyperparameters are unavailable.")
        return fig
    def structure(params):
        return json.dumps(
            {k: v for k, v in params.items()
             if k not in ("max_iter", "l2_regularization")},
            sort_keys=True, default=str,
        )
    rows = hgb.loc[hgb["params"].map(structure).eq(structure(anchor_params))].copy()
    rows["iterations"] = rows["params"].map(lambda p: int(p["max_iter"]))
    rows["l2"] = rows["params"].map(lambda p: float(p["l2_regularization"]))
    # Identical configurations under different names may have been rerun.
    for _, duplicates in rows.groupby(["iterations", "l2"]):
        values = duplicates[["mean_sharpe", "mean_ic", "mean_mse"]].to_numpy()
        if not np.allclose(values, values[0], equal_nan=True, rtol=1e-9, atol=1e-12):
            raise ValueError("Repeated curve settings have different metrics; resolve the repeats.")
    rows = rows.drop_duplicates(["iterations", "l2"])
    l2_values = sorted(rows["l2"].unique())
    palette = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    markers = {"Stage 1": "^", "Stage 2": "o", "Stage 3": "s"}
    metrics = [
        ("mean_sharpe", "Mean validation Sharpe", "Higher is better"),
        ("mean_ic", "Mean daily validation IC", "Higher is better"),
        ("mean_mse", "Mean validation MSE (return²)", "Lower is better"),
    ]
    for index, l2 in enumerate(l2_values):
        group = rows.loc[rows["l2"].eq(l2)].sort_values("iterations")
        color = palette[index % len(palette)]
        for ax, (metric, label, direction) in zip(axes, metrics):
            ax.plot(group["iterations"], group[metric], color=color, lw=1.8,
                    label=f"L2 = {l2:g}")
            for stage, marker in markers.items():
                points = group.loc[group["stage"].eq(stage)]
                ax.scatter(points["iterations"], points[metric],
                           marker=marker, color=color, s=38, zorder=3)
    for ax, (metric, label, direction) in zip(axes, metrics):
        ax.set_xlabel("Boosting iterations")
        ax.set_ylabel(label)
        ax.set_title(direction, loc="left")
        ax.set_xticks(sorted(rows["iterations"].unique()))
        ax.grid(alpha=0.2)
        if selected in rows.index:
            value = rows.loc[selected, metric]
            if np.isfinite(value):
                ax.scatter(rows.loc[selected, "iterations"], value,
                           marker="*", s=180, color=SELECTED_COLOR,
                           edgecolor="#333333", linewidth=0.6, zorder=5)
    axes[0].legend(title="Regularization", fontsize=8)
    stage_handles = [
        Line2D([], [], marker=marker, color="#555555", linestyle="", label=stage)
        for stage, marker in markers.items() if stage in rows["stage"].values
    ]
    if selected in rows.index:
        stage_handles.append(Line2D(
            [], [], marker="*", color=SELECTED_COLOR, markersize=11,
            linestyle="", label="Selected",
        ))
    axes[1].legend(handles=stage_handles, fontsize=8)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axes[2].yaxis.set_major_formatter(formatter)
    fig.suptitle(
        "Stages 2–3: controlled iteration and L2 curves\n"
        f"Fixed tree shape: leaves={anchor_params.get('max_leaf_nodes')}, "
        f"min leaf={anchor_params.get('min_samples_leaf')}, "
        f"depth={anchor_params.get('max_depth')}, "
        f"learning rate={anchor_params.get('learning_rate')}. "
        "A matching Stage 1 point fills any skipped duplicate.",
        fontsize=12,
    )
    return fig, rows


def _training_gaps(summary, selected, top_n):
    names = _shortlist(summary, selected, top_n)
    rows = summary.loc[names]
    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(5, 0.4 * len(names) + 2.5)),
        constrained_layout=True,
    )
    panels = [
        ("mean_train_sharpe", "mean_sharpe", "Spread Sharpe ratio"),
        ("mean_train_ic", "mean_ic", "Mean daily Spearman IC"),
    ]
    for ax, (train_key, valid_key, label) in zip(axes, panels):
        if not rows[train_key].notna().any():
            _empty(ax, f"{train_key} is unavailable.")
            continue
        for i, name in enumerate(names):
            train_value = rows.loc[name, train_key]
            valid_value = rows.loc[name, valid_key]
            color = SELECTED_COLOR if name == selected else VALID_COLOR
            if np.isfinite(train_value):
                ax.plot([valid_value, train_value], [i, i],
                        color="#C4C8CC", lw=2, zorder=1)
                ax.scatter(train_value, i, marker="o", facecolors="none",
                           edgecolors=TRAIN_COLOR, s=50, zorder=2)
            ax.scatter(valid_value, i, color=color, s=45, zorder=3)
        ax.set_yticks(np.arange(len(names)), labels=names)
        ax.set_ylim(len(names) - 0.5, -0.5)
        for tick, name in zip(ax.get_yticklabels(), names):
            if name == selected:
                tick.set_fontweight("bold")
                tick.set_color(SELECTED_COLOR)
        ax.set_xlabel(label)
        ax.set_title("Training versus validation", loc="left")
        ax.axvline(0, color="#999999", lw=0.8)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(handles=[
            Line2D([], [], marker="o", markerfacecolor="none",
                   color=TRAIN_COLOR, linestyle="", label="Training"),
            Line2D([], [], marker="o", color=VALID_COLOR,
                   linestyle="", label="Validation"),
            Line2D([], [], marker="o", color=SELECTED_COLOR,
                   linestyle="", label="Selected validation"),
        ], fontsize=8)
    fig.suptitle(
        "Training optimism: shortlisted models and benchmarks\n"
        "Earlier training periods and later validation periods differ. "
        "The gap is a diagnostic, not a standalone overfitting test.",
        fontsize=12,
    )
    return fig


def plot_tuning_stages(
    stage1,
    stage2=None,
    stage3=None,
    *,
    selected_name: str | None = None,
    top_n: int | None = None,
    diagnostic_top_n: int = 6,
    save_dir: str | Path | None = None,
    show: bool = True,
    dpi: int = 180,
) -> dict[str, Any]:
    """Create four figures from completed search results without fitting models.

    selected_name defaults to the winner under the existing Sharpe rule.
    A different supplied name is highlighted but is not relabelled as the winner.
    top_n=None shows every candidate in the overview. An integer shows that many
    leading supervised candidates plus the highlighted candidate, reference,
    best Ridge, and any momentum baseline. The returned summary always includes
    every candidate.

    If save_dir is supplied, save each figure as PNG and PDF and the summary as
    CSV. Training metrics are optional; missing panels explain what is absent.
    Hyperparameters come from each result's candidates dictionary or params_json.
    """
    folds, summary = _prepare((stage1, stage2, stage3))
    supervised = summary.loc[~summary["is_baseline"]]
    if supervised.empty:
        raise ValueError("No supervised candidates were supplied.")
    winner = supervised.index[0]
    selected = winner if selected_name is None else selected_name
    if selected not in supervised.index:
        raise ValueError("selected_name must identify a supervised candidate in the results.")
    if diagnostic_top_n < 1:
        raise ValueError("diagnostic_top_n must be positive.")
    if dpi < 60:
        raise ValueError("Use dpi >= 60.")
    missing_params = [
        name for name, row in summary.iterrows()
        if row["model_name"] in (HGB, "ridge") and not row["params"]
    ]
    if missing_params:
        raise ValueError(
            f"Hyperparameters unavailable for {missing_params[:5]}. "
            "Pass TuningResults objects or per-fold DataFrames with params_json."
        )

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        figures = {
            "01_selection_and_years": _overview(folds, summary, selected, top_n),
            "02_stage1_tree_shape": _stage1(summary),
        }
        curves = _controlled_curves(summary, selected)
        if isinstance(curves, tuple):
            figures["03_stage2_stage3_curves"], curve_data = curves
        else:
            figures["03_stage2_stage3_curves"], curve_data = curves, pd.DataFrame()
        figures["04_training_validation_gaps"] = _training_gaps(
            summary, selected, diagnostic_top_n
        )
        saved = []
        if save_dir is not None:
            directory = Path(save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                for extension in ("png", "pdf"):
                    path = directory / f"{name}.{extension}"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    saved.append(path)
            export = summary.drop(columns="params").copy()
            export["params_json"] = summary["params"].map(
                lambda p: json.dumps(p, sort_keys=True, default=str)
            )
            path = directory / "tuning_selection_summary.csv"
            export.to_csv(path)
            saved.append(path)
        if show:
            plt.show()
    return {
        "summary": summary,
        "fold_results": folds,
        "curve_data": curve_data,
        "figures": figures,
        "saved_paths": saved,
        "winner_name": winner,
        "selected_name": selected,
        "selected_is_winner": selected == winner,
    }
