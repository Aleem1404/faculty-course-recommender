from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio


def load_chart_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_metadata(
    image_path: Path,
    caption: str,
    description: str,
) -> None:
    metadata_path = Path(
        str(image_path) + ".meta.json"
    )

    metadata = {
        "caption": caption,
        "description": description,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def save_figure(
    fig,
    output_path: Path,
    caption: str,
    description: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_image(
        str(output_path)
    )

    write_metadata(
        image_path=output_path,
        caption=caption,
        description=description,
    )


def chart_frame(
    chart_data: pd.DataFrame,
    group_name: str,
) -> pd.DataFrame:
    return chart_data.loc[
        chart_data["chart_group"]
        == group_name
    ].copy()


def make_top_rank_change_chart(
    chart_data: pd.DataFrame,
    output_dir: Path,
) -> Path:
    frame = chart_frame(
        chart_data,
        "top_rank_change",
    )

    fig = px.bar(
        frame,
        x="label",
        y="value",
        text="value",
        title=(
            "Top Recommendation Changes"
            "<br><span style='font-size: 18px; "
            "font-weight: normal;'>"
            "M3-KG reranking outcome across "
            "1,575 modules"
            "</span>"
        ),
    )

    fig.update_traces(
        cliponaxis=False,
        textposition="outside",
    )
    fig.update_xaxes(
        title_text="Outcome"
    )
    fig.update_yaxes(
        title_text="Modules"
    )

    output_path = (
        output_dir
        / "top_rank_changes.png"
    )

    save_figure(
        fig=fig,
        output_path=output_path,
        caption="Top recommendation changes",
        description=(
            "Bar chart showing how many "
            "modules had changed versus "
            "unchanged top-ranked "
            "recommendations after "
            "M3-KG reranking."
        ),
    )

    return output_path


def make_shared_topic_chart(
    chart_data: pd.DataFrame,
    output_dir: Path,
) -> Path:
    frame = chart_frame(
        chart_data,
        "shared_topics",
    )

    fig = px.bar(
        frame,
        x="label",
        y="value",
        text="value",
        title=(
            "Shared Topic Coverage"
            "<br><span style='font-size: 18px; "
            "font-weight: normal;'>"
            "M3-KG recommendation evidence "
            "distribution"
            "</span>"
        ),
    )

    fig.update_traces(
        cliponaxis=False,
        textposition="outside",
    )
    fig.update_xaxes(
        title_text="Evidence type"
    )
    fig.update_yaxes(
        title_text="Recs"
    )

    output_path = (
        output_dir
        / "shared_topic_coverage.png"
    )

    save_figure(
        fig=fig,
        output_path=output_path,
        caption="Shared topic coverage",
        description=(
            "Bar chart showing how many "
            "M3-KG recommendations had "
            "shared-topic evidence versus "
            "no shared-topic evidence."
        ),
    )

    return output_path


def make_case_type_chart(
    chart_data: pd.DataFrame,
    output_dir: Path,
) -> Path:
    frame = chart_frame(
        chart_data,
        "case_types",
    )

    fig = px.bar(
        frame,
        x="label",
        y="value",
        text="value",
        title=(
            "Expert Review Case Types"
            "<br><span style='font-size: 18px; "
            "font-weight: normal;'>"
            "Composition of the 20-case "
            "expert judgement sample"
            "</span>"
        ),
    )

    fig.update_traces(
        cliponaxis=False,
        textposition="outside",
    )
    fig.update_xaxes(
        title_text="Case type"
    )
    fig.update_yaxes(
        title_text="Cases"
    )

    output_path = (
        output_dir
        / "expert_review_case_types.png"
    )

    save_figure(
        fig=fig,
        output_path=output_path,
        caption="Expert review case types",
        description=(
            "Bar chart showing the number "
            "of changed-top-rank, strong "
            "graph-evidence, and stable-"
            "but-explained cases selected "
            "for expert review."
        ),
    )

    return output_path


def make_score_component_chart(
    chart_data: pd.DataFrame,
    output_dir: Path,
) -> Path:
    frame = chart_frame(
        chart_data,
        "score_means",
    )

    fig = px.bar(
        frame,
        x="label",
        y="value",
        text="value",
        title=(
            "Mean Ranking Components"
            "<br><span style='font-size: 18px; "
            "font-weight: normal;'>"
            "Average top-rank scores used "
            "for evaluation"
            "</span>"
        ),
    )

    fig.update_traces(
        cliponaxis=False,
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    fig.update_xaxes(
        title_text="Component"
    )
    fig.update_yaxes(
        title_text="Mean score"
    )

    output_path = (
        output_dir
        / "ranking_score_components.png"
    )

    save_figure(
        fig=fig,
        output_path=output_path,
        caption="Mean ranking components",
        description=(
            "Bar chart comparing the mean "
            "top M2-H semantic score, mean "
            "top M3-KG score, and mean top "
            "graph-support score."
        ),
    )

    return output_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    chart_data_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "charts"
        / "evaluation_chart_data.csv"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "charts"
    )

    chart_data = load_chart_data(
        chart_data_path
    )

    generated_paths = [
        make_top_rank_change_chart(
            chart_data=chart_data,
            output_dir=output_dir,
        ),
        make_shared_topic_chart(
            chart_data=chart_data,
            output_dir=output_dir,
        ),
        make_case_type_chart(
            chart_data=chart_data,
            output_dir=output_dir,
        ),
        make_score_component_chart(
            chart_data=chart_data,
            output_dir=output_dir,
        ),
    ]

    summary = {
        "chart_data_path": str(
            chart_data_path
        ),
        "generated_chart_files": [
            str(path)
            for path in generated_paths
        ],
        "generated_chart_count": len(
            generated_paths
        ),
    }

    summary_path = (
        output_dir
        / "rendered_chart_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()