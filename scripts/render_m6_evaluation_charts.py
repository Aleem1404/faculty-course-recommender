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
        render_with_plotly(benchmark_data, charts_dir)
    else:
        print("Plotly not available. Skipping chart export.")

    print(f"All M6 evaluation charts rendered into: {charts_dir}")


if __name__ == "__main__":
    main()
