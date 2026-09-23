from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# Prefer plotly if available, with graceful matplotlib fallback
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def render_with_plotly(benchmark_data: dict, charts_dir: Path) -> None:
    models_summary = benchmark_data["model_performance_summary"]
    
    model_labels = ["M0 (TF-IDF)", "M1 (Enr. TF-IDF)", "M2 (Semantic)", "M3 (KG)", "M4-v2 (Dual)", "M5 (Balanced)"]
    model_keys = list(models_summary.keys())

    ndcg_vals = [models_summary[k]["macro_metrics"]["mean_ndcg_at_5"] for k in model_keys]
    ndcg_ci_low = [models_summary[k]["confidence_intervals_95"]["ndcg_at_5"]["ci_lower"] for k in model_keys]
    ndcg_ci_high = [models_summary[k]["confidence_intervals_95"]["ndcg_at_5"]["ci_upper"] for k in model_keys]
    ndcg_err_y = [high - val for high, val in zip(ndcg_ci_high, ndcg_vals)]
    ndcg_err_y_minus = [val - low for low, val in zip(ndcg_ci_low, ndcg_vals)]

    p1_vals = [models_summary[k]["macro_metrics"]["mean_precision_at_1_strict"] for k in model_keys]
    p5_vals = [models_summary[k]["macro_metrics"]["mean_precision_at_5_strict"] for k in model_keys]
    mrr_vals = [models_summary[k]["macro_metrics"]["mean_mrr_strict"] for k in model_keys]
    map_vals = [models_summary[k]["macro_metrics"]["mean_map_at_5_strict"] for k in model_keys]

    # Chart 1: Grouped IR Metrics Progression
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="NDCG@5", x=model_labels, y=ndcg_vals, marker_color="#1E3A8A"))
    fig1.add_trace(go.Bar(name="Precision@1 (Strict)", x=model_labels, y=p1_vals, marker_color="#3B82F6"))
    fig1.add_trace(go.Bar(name="Precision@5 (Strict)", x=model_labels, y=p5_vals, marker_color="#60A5FA"))
    fig1.add_trace(go.Bar(name="MRR (Strict)", x=model_labels, y=mrr_vals, marker_color="#10B981"))
    fig1.add_trace(go.Bar(name="MAP@5 (Strict)", x=model_labels, y=map_vals, marker_color="#F59E0B"))

    fig1.update_layout(
        title="<b>M6 Information Retrieval Benchmark: Ablation Progression (M0 → M5)</b>",
        title_font=dict(size=18, color="#0F172A", family="Inter, Arial, sans-serif"),
        barmode="group",
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="<b>Model Architecture Tier</b>", gridcolor="#E2E8F0"),
        yaxis=dict(title="<b>Metric Score [0.0 - 1.0]</b>", range=[0.0, 1.0], gridcolor="#E2E8F0"),
        font=dict(family="Inter, Arial, sans-serif", size=13),
        margin=dict(l=60, r=40, t=90, b=60),
    )

    chart1_path = charts_dir / "ir_metrics_progression_m0_to_m5.png"
    fig1.write_image(str(chart1_path), scale=2, width=1000, height=600)
    print(f"Rendered: {chart1_path}")

    # Chart 2: NDCG@5 with 95% Bootstrap Confidence Intervals
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=model_labels,
        y=ndcg_vals,
        mode="lines+markers",
        name="NDCG@5 Mean",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=10, color="#1E3A8A"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=ndcg_err_y,
            arrayminus=ndcg_err_y_minus,
            visible=True,
            thickness=2,
            width=8,
            color="#DC2626",
        )
    ))

    fig2.update_layout(
        title="<b>NDCG@5 Ranking Quality with 95% Non-Parametric Bootstrap CI (B=2000)</b>",
        title_font=dict(size=18, color="#0F172A", family="Inter, Arial, sans-serif"),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(title="<b>Model Architecture</b>", gridcolor="#E2E8F0"),
        yaxis=dict(title="<b>NDCG@5 (95% CI)</b>", range=[0.0, 1.0], gridcolor="#E2E8F0"),
        font=dict(family="Inter, Arial, sans-serif", size=13),
        margin=dict(l=60, r=40, t=90, b=60),
    )

    chart2_path = charts_dir / "ndcg_confidence_intervals.png"
    fig2.write_image(str(chart2_path), scale=2, width=900, height=550)
    print(f"Rendered: {chart2_path}")

    # Chart 3: Radar Chart of Tradeoffs
    categories = ["NDCG@5", "P@1 Strict", "P@5 Strict", "MRR Strict", "MAP@5 Strict"]
    fig3 = go.Figure()

    colors = ["#94A3B8", "#64748B", "#2563EB", "#8B5CF6", "#EC4899", "#10B981"]
    for idx, (m_id, label) in enumerate(zip(model_keys, model_labels)):
        m = models_summary[m_id]["macro_metrics"]
        r_vals = [
            m["mean_ndcg_at_5"],
            m["mean_precision_at_1_strict"],
            m["mean_precision_at_5_strict"],
            m["mean_mrr_strict"],
            m["mean_map_at_5_strict"],
        ]
        # Close the loop
        r_vals.append(r_vals[0])
        fig3.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=categories + [categories[0]],
            fill="toself" if m_id in ("M2_Semantic", "M5_LoadBalanced") else "none",
            name=label,
            line=dict(color=colors[idx], width=2),
            opacity=0.4 if m_id in ("M2_Semantic", "M5_LoadBalanced") else 0.8,
        ))

    fig3.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], gridcolor="#E2E8F0"),
            bgcolor="#F8FAFC"
        ),
        title="<b>Multi-Dimensional IR Performance Radar (M0 through M5)</b>",
        title_font=dict(size=18, color="#0F172A", family="Inter, Arial, sans-serif"),
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=90, b=80),
    )

    chart3_path = charts_dir / "radar_ablation_comparison.png"
    fig3.write_image(str(chart3_path), scale=2, width=800, height=700)
    print(f"Rendered: {chart3_path}")


def render_with_matplotlib(benchmark_data: dict, charts_dir: Path) -> None:
    models_summary = benchmark_data["model_performance_summary"]
    model_labels = ["M0\n(TF-IDF)", "M1\n(Enr. TF-IDF)", "M2\n(Semantic)", "M3\n(KG)", "M4-v2\n(Dual)", "M5\n(Balanced)"]
    model_keys = list(models_summary.keys())

    ndcg_vals = [models_summary[k]["macro_metrics"]["mean_ndcg_at_5"] for k in model_keys]
    ndcg_ci_low = [models_summary[k]["confidence_intervals_95"]["ndcg_at_5"]["ci_lower"] for k in model_keys]
    ndcg_ci_high = [models_summary[k]["confidence_intervals_95"]["ndcg_at_5"]["ci_upper"] for k in model_keys]
    p1_vals = [models_summary[k]["macro_metrics"]["mean_precision_at_1_strict"] for k in model_keys]
    p5_vals = [models_summary[k]["macro_metrics"]["mean_precision_at_5_strict"] for k in model_keys]
    mrr_vals = [models_summary[k]["macro_metrics"]["mean_mrr_strict"] for k in model_keys]
    map_vals = [models_summary[k]["macro_metrics"]["mean_map_at_5_strict"] for k in model_keys]

    x = np.arange(len(model_labels))
    width = 0.16

    # Chart 1: Grouped IR Metrics Progression
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    ax.bar(x - 2*width, ndcg_vals, width, label="NDCG@5", color="#1E3A8A")
    ax.bar(x - width, p1_vals, width, label="Precision@1 (Strict)", color="#3B82F6")
    ax.bar(x, p5_vals, width, label="Precision@5 (Strict)", color="#60A5FA")
    ax.bar(x + width, mrr_vals, width, label="MRR (Strict)", color="#10B981")
    ax.bar(x + 2*width, map_vals, width, label="MAP@5 (Strict)", color="#F59E0B")

    ax.set_title("M6 Information Retrieval Benchmark: Ablation Progression (M0 → M5)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Model Architecture Tier", fontweight="bold", labelpad=10)
    ax.set_ylabel("Metric Score [0.0 - 1.0]", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    chart1_path = charts_dir / "ir_metrics_progression_m0_to_m5.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"Rendered (Matplotlib): {chart1_path}")

    # Chart 2: NDCG@5 with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    err_low = np.array(ndcg_vals) - np.array(ndcg_ci_low)
    err_high = np.array(ndcg_ci_high) - np.array(ndcg_vals)
    yerr = [err_low, err_high]

    ax.errorbar(x, ndcg_vals, yerr=yerr, fmt="-o", color="#2563EB", ecolor="#EF4444", elinewidth=2, capsize=5, capthick=1.5, label="NDCG@5 (95% Bootstrap CI)")
    for i, v in enumerate(ndcg_vals):
        ax.annotate(f"{v:.3f}", (x[i], v + 0.03), ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Empirical Ranking Quality: NDCG@5 with 95% Non-Parametric Bootstrap CIs (B=2000)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Model Tier", fontweight="bold")
    ax.set_ylabel("Mean NDCG@5 Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, max(ndcg_ci_high) + 0.1 if max(ndcg_ci_high) > 0 else 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    chart2_path = charts_dir / "ndcg_confidence_intervals.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"Rendered (Matplotlib): {chart2_path}")

    # Chart 3: Radar Chart
    categories = ["NDCG@5", "P@1", "P@5", "MRR", "MAP@5"]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=200)
    colors = ["#94A3B8", "#64748B", "#2563EB", "#8B5CF6", "#EC4899", "#10B981"]

    for idx, (m_id, label) in enumerate(zip(model_keys, model_labels)):
        m = models_summary[m_id]["macro_metrics"]
        values = [
            m["mean_ndcg_at_5"],
            m["mean_precision_at_1_strict"],
            m["mean_precision_at_5_strict"],
            m["mean_mrr_strict"],
            m["mean_map_at_5_strict"],
        ]
        values += values[:1]
        clean_label = label.replace("\n", " ")
        ax.plot(angles, values, color=colors[idx], linewidth=1.5, label=clean_label)
        if m_id in ("M2_Semantic", "M5_LoadBalanced"):
            ax.fill(angles, values, color=colors[idx], alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Dimensional IR Performance Radar (M0 → M5)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    chart3_path = charts_dir / "radar_ablation_comparison.png"
    plt.savefig(chart3_path)
    plt.close()
    print(f"Rendered (Matplotlib): {chart3_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    benchmark_json_path = (
        project_root / "data" / "outputs" / "m6" / "consolidated_ir_benchmark.json"
    )
    charts_dir = project_root / "data" / "outputs" / "m6" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_json_path.exists():
        print("Running M6 benchmark to generate prerequisite JSON...")
        import sys
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))
        from m6.pipeline import run_m6_pipeline
        run_m6_pipeline(project_root=project_root)

    benchmark_data = json.loads(benchmark_json_path.read_text(encoding="utf-8"))

    if PLOTLY_AVAILABLE:
        print("Rendering publication charts with Plotly & Kaleido...")
        try:
            render_with_plotly(benchmark_data, charts_dir)
        except Exception as e:
            print(f"Plotly/Kaleido export error: {e}. Falling back to Matplotlib...")
            if MATPLOTLIB_AVAILABLE:
                render_with_matplotlib(benchmark_data, charts_dir)
    elif MATPLOTLIB_AVAILABLE:
        print("Rendering charts with Matplotlib...")
        render_with_matplotlib(benchmark_data, charts_dir)
    else:
        print("Neither Plotly nor Matplotlib available for chart export.")

    print(f"All M6 evaluation charts rendered into: {charts_dir}")


if __name__ == "__main__":
    main()
