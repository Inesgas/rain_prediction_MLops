from __future__ import annotations

from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_WINNER_COLOR = "#245866"
ALTERNATIVE_COLOR = "#b9c9c1"
ACCENT_COLOR = "#b86a3b"
TEAL_ACCENT = "#6f9c96"
TEXT_COLOR = "#173848"
REFERENCE_COLOR = "#7a7a7a"
WIDE_CHART_WIDTH = 880
ROBUSTNESS_CHART_WIDTH = 920
LOCATION_CHART_WIDTH = 940


def _finish_chart(chart: alt.Chart):
    return (
        chart.configure_axis(
            labelFontSize=12,
            titleFontSize=13,
            labelColor=TEXT_COLOR,
            titleColor=TEXT_COLOR,
            gridColor="#d8e2df",
            tickColor="#b9c9c1",
        )
        .configure_legend(
            labelFontSize=12,
            titleFontSize=12,
            labelColor=TEXT_COLOR,
            titleColor=TEXT_COLOR,
            orient="bottom",
        )
        .configure_title(fontSize=15, color=TEXT_COLOR)
        .configure_view(strokeWidth=0)
    )


def calibration_chart(curve_df: pd.DataFrame, view_label: str) -> pd.DataFrame:
    result = curve_df[["mean_predicted_probability", "observed_frequency"]].copy()
    result["view"] = view_label
    return result


def build_calibration_curve_chart(raw_curve: pd.DataFrame, calibrated_curve: pd.DataFrame):
    chart_data = pd.concat(
        [
            calibration_chart(raw_curve, "Final Hybrid Baseline (Raw Winner)"),
            calibration_chart(calibrated_curve, "Climate-Regime Calibrated View"),
            pd.DataFrame(
                {
                    "mean_predicted_probability": [0.0, 1.0],
                    "observed_frequency": [0.0, 1.0],
                    "view": ["Ideal reference", "Ideal reference"],
                }
            ),
        ],
        ignore_index=True,
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3)
        .encode(
            x=alt.X("mean_predicted_probability:Q", title="Mean predicted probability"),
            y=alt.Y("observed_frequency:Q", title="Observed rain frequency"),
            color=alt.Color(
                "view:N",
                title="Calibration view",
                scale=alt.Scale(
                    domain=[
                        "Final Hybrid Baseline (Raw Winner)",
                        "Climate-Regime Calibrated View",
                        "Ideal reference",
                    ],
                    range=[FINAL_WINNER_COLOR, ACCENT_COLOR, REFERENCE_COLOR],
                ),
            ),
            tooltip=["view", "mean_predicted_probability", "observed_frequency"],
        )
        .properties(width=WIDE_CHART_WIDTH, height=420)
    )
    return _finish_chart(chart)


def build_model_ranking_chart(comparison_df: pd.DataFrame):
    chart_df = comparison_df.copy()
    chart_df["group"] = chart_df["rank"].eq(1).map({True: "Final winner", False: "Shortlisted alternative"})
    sort_order = chart_df.sort_values("test_f1", ascending=True)["display_label"].tolist()
    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            y=alt.Y("display_label:N", sort=sort_order, title=None),
            x=alt.X("test_f1:Q", title="Holdout F1"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["Final winner", "Shortlisted alternative"],
                    range=[FINAL_WINNER_COLOR, ALTERNATIVE_COLOR],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Model view"),
                alt.Tooltip("test_f1:Q", title="Holdout F1", format=".4f"),
                alt.Tooltip("test_roc_auc:Q", title="ROC-AUC", format=".4f"),
                alt.Tooltip("test_precision:Q", title="Rain precision", format=".4f"),
                alt.Tooltip("test_recall:Q", title="Rain recall", format=".4f"),
            ],
        )
        .properties(width=WIDE_CHART_WIDTH, height=320)
    )
    labels = (
        alt.Chart(chart_df)
        .mark_text(align="left", baseline="middle", dx=6, color=TEXT_COLOR)
        .encode(
            y=alt.Y("display_label:N", sort=sort_order, title=None),
            x=alt.X("test_f1:Q"),
            text=alt.Text("test_f1:Q", format=".4f"),
        )
    )
    return _finish_chart(bars + labels)


def build_robustness_time_chart(rolling_df: pd.DataFrame):
    chart_df = rolling_df[["split_name", "test_f1", "test_roc_auc"]].melt(
        id_vars="split_name",
        value_vars=["test_f1", "test_roc_auc"],
        var_name="metric",
        value_name="score",
    )
    chart_df["metric"] = chart_df["metric"].map(
        {"test_f1": "Holdout F1", "test_roc_auc": "Holdout ROC-AUC"}
    )
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3)
        .encode(
            x=alt.X("split_name:N", title="Chronological window"),
            y=alt.Y("score:Q", title="Score"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["Holdout F1", "Holdout ROC-AUC"], range=[FINAL_WINNER_COLOR, ACCENT_COLOR]),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("split_name:N", title="Window"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(width=ROBUSTNESS_CHART_WIDTH, height=360)
    )
    return _finish_chart(chart)


def build_segment_bar_chart(segment_df: pd.DataFrame, title: str):
    chart_df = segment_df.copy()
    if "eligible_for_summary" in chart_df.columns:
        chart_df = chart_df.loc[chart_df["eligible_for_summary"]]
    chart_df = chart_df.sort_values("f1", ascending=True)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            y=alt.Y("segment_value:N", sort=chart_df["segment_value"].tolist(), title=None),
            x=alt.X("f1:Q", title=title),
            color=alt.Color(
                "f1:Q",
                scale=alt.Scale(scheme="tealblues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("segment_value:N", title="Segment"),
                alt.Tooltip("support:Q", title="Support"),
                alt.Tooltip("roc_auc:Q", title="ROC-AUC", format=".4f"),
                alt.Tooltip("f1:Q", title="F1", format=".4f"),
                alt.Tooltip("precision:Q", title="Precision", format=".4f"),
                alt.Tooltip("recall:Q", title="Recall", format=".4f"),
            ],
        )
        .properties(width=WIDE_CHART_WIDTH, height=max(260, 52 * len(chart_df)))
    )
    return _finish_chart(chart)


def build_location_robustness_chart(location_df: pd.DataFrame):
    chart_df = location_df.copy()
    if "eligible_for_summary" in chart_df.columns:
        chart_df = chart_df.loc[chart_df["eligible_for_summary"]]
    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=130, opacity=0.78, stroke="white", strokeWidth=0.8)
        .encode(
            x=alt.X("support:Q", title="Location support"),
            y=alt.Y("f1:Q", title="Location-level F1"),
            color=alt.Color("climate_regime:N", title="Climate regime"),
            tooltip=[
                alt.Tooltip("segment_value:N", title="Location"),
                alt.Tooltip("climate_regime:N", title="Climate regime"),
                alt.Tooltip("support:Q", title="Support"),
                alt.Tooltip("roc_auc:Q", title="ROC-AUC", format=".4f"),
                alt.Tooltip("f1:Q", title="F1", format=".4f"),
                alt.Tooltip("precision:Q", title="Precision", format=".4f"),
                alt.Tooltip("recall:Q", title="Recall", format=".4f"),
            ],
        )
        .properties(width=LOCATION_CHART_WIDTH, height=420)
    )
    return _finish_chart(chart)


def _save_fig(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_ranking_figure(comparison_df: pd.DataFrame, output_path: Path) -> None:
    chart_df = comparison_df.copy().sort_values("test_f1", ascending=True)
    groups = chart_df["rank"].eq(1)
    colors = [FINAL_WINNER_COLOR if is_winner else ALTERNATIVE_COLOR for is_winner in groups]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.barh(chart_df["display_label"], chart_df["test_f1"], color=colors)
    for bar, value in zip(bars, chart_df["test_f1"]):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", ha="left", color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel("Holdout F1")
    ax.set_ylabel("")
    ax.set_title("Best model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    _save_fig(fig, output_path)


def save_calibration_curve_figure(raw_curve: pd.DataFrame, calibrated_curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(
        raw_curve["mean_predicted_probability"],
        raw_curve["observed_frequency"],
        marker="o",
        color=FINAL_WINNER_COLOR,
        linewidth=2.3,
        label="Final Hybrid Baseline (Raw Winner)",
    )
    ax.plot(
        calibrated_curve["mean_predicted_probability"],
        calibrated_curve["observed_frequency"],
        marker="o",
        color=ACCENT_COLOR,
        linewidth=2.3,
        label="Climate-Regime Calibrated View",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOR, linewidth=1.5, label="Ideal reference")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rain frequency")
    ax.set_title("Calibration view used in the app")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")
    _save_fig(fig, output_path)


def save_robustness_time_figure(rolling_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(rolling_df))
    ax.plot(x, rolling_df["test_f1"], marker="o", linewidth=2.4, color=FINAL_WINNER_COLOR, label="Holdout F1")
    ax.plot(x, rolling_df["test_roc_auc"], marker="o", linewidth=2.4, color=ACCENT_COLOR, label="Holdout ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(rolling_df["split_name"], rotation=0)
    ax.set_xlabel("Chronological window")
    ax.set_ylabel("Score")
    ax.set_title("Temporal robustness used in the app")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")
    _save_fig(fig, output_path)


def _tealblues_colors(values: pd.Series) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("PuBuGn")
    if len(values) == 0:
        return []
    ranks = pd.Series(values).rank(method="dense", pct=True).fillna(0.5)
    return [cmap(0.35 + 0.55 * rank) for rank in ranks]


def save_segment_bar_figure(segment_df: pd.DataFrame, title: str, output_path: Path) -> None:
    chart_df = segment_df.copy()
    if "eligible_for_summary" in chart_df.columns:
        chart_df = chart_df.loc[chart_df["eligible_for_summary"]]
    chart_df = chart_df.sort_values("f1", ascending=True)
    fig_height = max(4.0, 0.45 * len(chart_df))
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    colors = _tealblues_colors(chart_df["f1"])
    ax.barh(chart_df["segment_value"], chart_df["f1"], color=colors)
    ax.set_xlabel(title)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, output_path)


def save_location_robustness_figure(location_df: pd.DataFrame, output_path: Path) -> None:
    chart_df = location_df.copy()
    if "eligible_for_summary" in chart_df.columns:
        chart_df = chart_df.loc[chart_df["eligible_for_summary"]]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    regimes = list(chart_df["climate_regime"].dropna().astype(str).unique())
    palette = plt.get_cmap("tab10")
    for idx, regime in enumerate(regimes):
        part = chart_df.loc[chart_df["climate_regime"].astype(str) == regime]
        ax.scatter(
            part["support"],
            part["f1"],
            s=54,
            alpha=0.78,
            color=palette(idx % 10),
            edgecolor="white",
            linewidth=0.6,
            label=regime,
        )
    ax.set_xlabel("Location support")
    ax.set_ylabel("Location-level F1")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best", fontsize=8)
    _save_fig(fig, output_path)
