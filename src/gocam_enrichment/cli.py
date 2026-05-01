from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from gocam_enrichment.diffusion import diffuse, extract_subgraphs, map_inputs
from gocam_enrichment.gocam_graph import build_graph, filter_graph_by_taxon, graph_summary, load_graph, save_graph
from gocam_enrichment.model_io import fetch_models, load_input_metadata, load_models, parse_inputs, write_models_yaml
from gocam_enrichment.output_adapters import write_outputs

app = typer.Typer(help="GO-CAM grounded network retrieval from mixed object lists.")
console = Console()


@app.command()
def fetch(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output multi-document YAML file.")],
    model_ids: Annotated[
        Optional[list[str]],
        typer.Option("--model-id", help="GO-CAM model ID to fetch. Repeatable."),
    ] = None,
    taxon: Annotated[
        Optional[str],
        typer.Option("--taxon", help="Keep only models with this exact taxon CURIE."),
    ] = None,
    max_models: Annotated[
        Optional[int],
        typer.Option("--max-models", help="Maximum number of API models to fetch before filtering."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all-models", help="Allow fetching the full public GO-CAM index."),
    ] = False,
) -> None:
    """Fetch GO-CAM models from the GO API and write YAML."""

    _guard_api_fetch(model_ids=model_ids, taxon=taxon, max_models=max_models, all_models=all_models)
    models = fetch_models(model_ids=model_ids, taxon=taxon, max_models=max_models)
    write_models_yaml(models, out)
    console.print(f"Wrote {len(models)} model(s) to {out}")


@app.command()
def build(
    models: Annotated[
        list[Path],
        typer.Option("--models", "-m", help="YAML/JSON model file. Repeatable."),
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output graph JSON file.")],
    taxon: Annotated[
        Optional[str],
        typer.Option("--taxon", help="Keep only models with this exact taxon CURIE."),
    ] = None,
) -> None:
    """Build a reusable heterogeneous graph from GO-CAM YAML/JSON."""

    loaded = load_models(models, taxon=taxon)
    graph = build_graph(loaded)
    save_graph(graph, out)
    summary = graph_summary(graph)
    console.print(
        f"Wrote graph to {out}: {summary['models']} model(s), "
        f"{summary['activities']} activities, {summary['nodes']} nodes, {summary['edges']} edges"
    )


@app.command()
def run(
    inputs: Annotated[Path, typer.Argument(help="TSV/CSV/YAML/plain input objects.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output result directory.")],
    graph_path: Annotated[
        Optional[Path],
        typer.Option("--graph", help="Prebuilt graph JSON from `gocam-enrich build`."),
    ] = None,
    models: Annotated[
        Optional[list[Path]],
        typer.Option("--models", "-m", help="YAML/JSON model file. Repeatable."),
    ] = None,
    source: Annotated[
        str,
        typer.Option("--source", help="Model source when --graph is absent: local or api."),
    ] = "local",
    model_ids: Annotated[
        Optional[list[str]],
        typer.Option("--model-id", help="GO-CAM model ID to fetch when --source api. Repeatable."),
    ] = None,
    taxon: Annotated[
        Optional[str],
        typer.Option("--taxon", help="Keep only models with this exact taxon CURIE."),
    ] = None,
    max_models: Annotated[
        Optional[int],
        typer.Option("--max-models", help="Maximum API models to fetch before filtering."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all-models", help="Allow fetching the full public GO-CAM index."),
    ] = False,
    top: Annotated[int, typer.Option("--top", help="Maximum number of subgraphs to return.")] = 20,
    radius: Annotated[int, typer.Option("--radius", help="Causal activity radius around each seed.")] = 1,
    max_bridge_degree: Annotated[
        int,
        typer.Option(
            "--max-bridge-degree",
            help="Skip shared object expansion through terms attached to more than this many activities.",
        ),
    ] = 50,
    max_subgraph_activities: Annotated[
        int,
        typer.Option("--max-subgraph-activities", help="Maximum activities retained per returned subgraph."),
    ] = 150,
    restart: Annotated[float, typer.Option("--restart", help="Random-walk restart probability.")] = 0.35,
    steps: Annotated[int, typer.Option("--steps", help="Diffusion iterations.")] = 40,
    save_built_graph: Annotated[
        Optional[Path],
        typer.Option("--save-graph", help="Save graph built during this run."),
    ] = None,
) -> None:
    """Run diffusion retrieval and write ranked grounded subgraphs."""

    query_inputs = parse_inputs(inputs)
    if not query_inputs:
        raise typer.BadParameter(f"No input objects found in {inputs}")

    graph = _load_or_build_graph(
        graph_path=graph_path,
        model_paths=models,
        source=source,
        model_ids=model_ids,
        taxon=taxon,
        max_models=max_models,
        all_models=all_models,
    )
    if save_built_graph is not None:
        save_graph(graph, save_built_graph)

    mapped = map_inputs(graph, query_inputs)
    mapped_count = sum(1 for item in mapped if item.node is not None)
    if mapped_count == 0:
        raise typer.BadParameter("None of the input objects mapped to graph nodes.")

    scores, signed_scores = diffuse(graph, mapped, restart=restart, steps=steps)
    subgraphs = extract_subgraphs(
        graph,
        mapped,
        scores,
        signed_scores,
        max_subgraphs=top,
        radius=radius,
        max_bridge_degree=max_bridge_degree,
        max_subgraph_activities=max_subgraph_activities,
    )
    write_outputs(subgraphs, mapped, out, input_path=inputs, input_metadata=load_input_metadata(inputs))
    console.print(
        f"Mapped {mapped_count}/{len(mapped)} input object(s); wrote "
        f"{len(subgraphs)} subgraph(s) to {out}"
    )


def _load_or_build_graph(
    *,
    graph_path: Path | None,
    model_paths: list[Path] | None,
    source: str,
    model_ids: list[str] | None,
    taxon: str | None,
    max_models: int | None,
    all_models: bool,
):
    if graph_path is not None:
        return filter_graph_by_taxon(load_graph(graph_path), taxon)

    if source == "api":
        _guard_api_fetch(model_ids=model_ids, taxon=taxon, max_models=max_models, all_models=all_models)
        loaded = fetch_models(model_ids=model_ids, taxon=taxon, max_models=max_models)
    elif source == "local":
        if not model_paths:
            raise typer.BadParameter("Provide --graph, --models, or use --source api.")
        loaded = load_models(model_paths, taxon=taxon)
    else:
        raise typer.BadParameter("--source must be either `local` or `api`.")

    if not loaded:
        raise typer.BadParameter("No GO-CAM models loaded for the requested source/filter.")
    return build_graph(loaded)


def _guard_api_fetch(
    *,
    model_ids: list[str] | None,
    taxon: str | None,
    max_models: int | None,
    all_models: bool,
) -> None:
    if model_ids or taxon or max_models is not None or all_models:
        return
    raise typer.BadParameter(
        "API fetch needs --model-id, --taxon, --max-models, or explicit --all-models."
    )


if __name__ == "__main__":
    app()
