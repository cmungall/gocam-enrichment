from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

import networkx as nx

from gocam_enrichment.diffusion import MappedInput
from gocam_enrichment.gocam_graph import original_id
from gocam_enrichment.results import (
    ActivityGrounding,
    EdgeResult,
    InputResult,
    NodeResult,
    PathResult,
    ResultBundle,
    SubgraphResult,
    TermSummary,
)


class OutputAdapter(Protocol):
    """A writer for one or more concrete output artifacts."""

    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        """Write artifacts to an output directory."""


def build_result_bundle(
    subgraphs: list[dict[str, Any]],
    mapped_inputs: list[MappedInput],
    *,
    input_path: Path | None = None,
    input_metadata: dict[str, object] | None = None,
) -> ResultBundle:
    """Convert extraction dictionaries to the public output data model."""

    subgraph_results = [
        _subgraph_result(index, subgraph)
        for index, subgraph in enumerate(subgraphs, start=1)
    ]
    unmapped = [_input_result(mapped) for mapped in mapped_inputs if mapped.node is None]
    return ResultBundle(
        input_count=len(mapped_inputs),
        mapped_input_count=sum(1 for item in mapped_inputs if item.node is not None),
        subgraphs=subgraph_results,
        unmapped_inputs=unmapped,
        input_path=str(input_path) if input_path is not None else None,
        input_metadata=dict(input_metadata or {}),
    )


def write_outputs(
    subgraphs: list[dict[str, Any]],
    mapped_inputs: list[MappedInput],
    output_dir: Path,
    adapters: list[OutputAdapter] | None = None,
    *,
    input_path: Path | None = None,
    input_metadata: dict[str, object] | None = None,
) -> ResultBundle:
    """Write result artifacts through one or more adapters."""

    bundle = build_result_bundle(
        subgraphs,
        mapped_inputs,
        input_path=input_path,
        input_metadata=input_metadata,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for adapter in adapters or default_adapters():
        adapter.write(bundle, output_dir)
    return bundle


def default_adapters() -> list[OutputAdapter]:
    return [
        RankedSubgraphsTsvAdapter(),
        GroundingTsvAdapter(),
        UnmappedInputsTsvAdapter(),
        DotSubgraphAdapter(),
        GraphvizImageAdapter(),
        MarkdownReportAdapter(),
        JsonSubgraphAdapter(),
        GraphmlSubgraphAdapter(),
    ]


class RankedSubgraphsTsvAdapter:
    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        path = output_dir / "ranked_subgraphs.tsv"
        with path.open("w") as handle:
            handle.write(
                "rank\tscore\traw_score\tsigned_score\tactivities\tinput_hits\tmodels\ttop_terms\ttop_term_labels\tdescription_stub\n"
            )
            for subgraph in bundle.subgraphs:
                top_terms = ";".join(term.id for term in subgraph.top_terms[:5])
                top_labels = ";".join(term.label or term.id for term in subgraph.top_terms[:5])
                description = ", ".join(term.label or term.id for term in subgraph.top_terms[:3])
                handle.write(
                    "\t".join(
                        [
                            str(subgraph.rank),
                            f"{subgraph.score:.8g}",
                            f"{subgraph.raw_score:.8g}",
                            f"{subgraph.signed_score:.8g}",
                            str(subgraph.activity_count),
                            str(subgraph.input_hit_count),
                            str(len(subgraph.models)),
                            top_terms,
                            top_labels,
                            description,
                        ]
                    )
                    + "\n"
                )


class GroundingTsvAdapter:
    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        path = output_dir / "grounding.tsv"
        with path.open("w") as handle:
            handle.write(
                "\t".join(
                    [
                        "subgraph",
                        "activity_id",
                        "activity_label",
                        "model_id",
                        "model_title",
                        "score",
                        "signed_score",
                        "enabled_by",
                        "enabled_by_label",
                        "molecular_function",
                        "molecular_function_label",
                        "biological_process",
                        "biological_process_label",
                        "cellular_component",
                        "cellular_component_label",
                    ]
                )
                + "\n"
            )
            for subgraph in bundle.subgraphs:
                for activity in subgraph.activities:
                    handle.write(
                        "\t".join(
                            [
                                str(subgraph.rank),
                                activity.id,
                                activity.label or "",
                                activity.model_id or "",
                                _tsv(activity.model_title),
                                f"{activity.score:.8g}",
                                f"{activity.signed_score:.8g}",
                                activity.enabled_by or "",
                                activity.enabled_by_label or "",
                                activity.molecular_function or "",
                                activity.molecular_function_label or "",
                                activity.biological_process or "",
                                activity.biological_process_label or "",
                                activity.cellular_component or "",
                                activity.cellular_component_label or "",
                            ]
                        )
                        + "\n"
                    )


class UnmappedInputsTsvAdapter:
    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        path = output_dir / "unmapped_inputs.tsv"
        with path.open("w") as handle:
            handle.write("id\ttype\tmodality\tvalue\tlabel\n")
            for input_obj in bundle.unmapped_inputs:
                handle.write(
                    "\t".join(
                        [
                            input_obj.id,
                            input_obj.type or "",
                            input_obj.modality or "",
                            str(input_obj.value),
                            input_obj.label or "",
                        ]
                    )
                    + "\n"
                )


class JsonSubgraphAdapter:
    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        for subgraph in bundle.subgraphs:
            path = output_dir / f"subgraph_{subgraph.rank:03d}.json"
            path.write_text(json.dumps(subgraph.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True))


class GraphmlSubgraphAdapter:
    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        for subgraph in bundle.subgraphs:
            path = output_dir / f"subgraph_{subgraph.rank:03d}.graphml"
            _write_graphml(subgraph, path)


class DotSubgraphAdapter:
    def __init__(self, max_activities: int = 30) -> None:
        self.max_activities = max_activities

    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        for subgraph in bundle.subgraphs:
            path = output_dir / f"subgraph_{subgraph.rank:03d}.dot"
            path.write_text(_dot_subgraph(subgraph, max_activities=self.max_activities))


class GraphvizImageAdapter:
    def __init__(self, formats: tuple[str, ...] = ("svg", "png")) -> None:
        self.formats = formats

    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        dot_binary = shutil.which("dot")
        if dot_binary is None:
            (output_dir / "graphviz_rendering.txt").write_text(
                "Graphviz `dot` was not found, so only .dot files were written.\n"
            )
            return

        for subgraph in bundle.subgraphs:
            dot_path = output_dir / f"subgraph_{subgraph.rank:03d}.dot"
            if not dot_path.exists():
                continue
            for image_format in self.formats:
                image_path = output_dir / f"subgraph_{subgraph.rank:03d}.{image_format}"
                subprocess.run(
                    [dot_binary, f"-T{image_format}", str(dot_path), "-o", str(image_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )


class MarkdownReportAdapter:
    def write(self, bundle: ResultBundle, output_dir: Path) -> None:
        path = output_dir / "report.md"
        path.write_text(_markdown_report(bundle, output_dir))


def _markdown_report(bundle: ResultBundle, output_dir: Path | None = None) -> str:
    metadata = bundle.input_metadata
    title = _text(metadata.get("title") or metadata.get("id") or "GO-CAM Enrichment Report")
    lines = [f"# {title}", ""]

    description = _text(metadata.get("description"))
    if description:
        lines.extend([description, ""])

    lines.extend(
        [
            "## Run Summary",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Input source | {_md(bundle.input_path or 'not recorded')} |",
            f"| Input set | {_md(_text(metadata.get('id')) or 'not recorded')} |",
            f"| Category | {_md(_text(metadata.get('category')) or 'not recorded')} |",
            f"| Input coverage | {bundle.mapped_input_count}/{bundle.input_count} mapped |",
            f"| Returned subgraphs | {len(bundle.subgraphs)} |",
            "",
        ]
    )

    context = metadata.get("context")
    if isinstance(context, dict):
        lines.extend(["## Context", "", "| Field | Value |", "| --- | --- |"])
        for key in ("taxon", "species_label", "recommended_graph"):
            value = _text(context.get(key))
            if value:
                lines.append(f"| {_md(_labelize(key))} | {_md(value)} |")
        lines.append("")

    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        lines.extend(["## Provenance", "", "| Field | Value |", "| --- | --- |"])
        for key in ("source_kind", "source_name", "source_path", "selection_notes", "value_notes"):
            value = _text(provenance.get(key))
            if value:
                lines.append(f"| {_md(_labelize(key))} | {_md(value)} |")
        lines.append("")

    expected = metadata.get("expected_answer")
    if isinstance(expected, dict):
        lines.extend(["## Expected Answer", ""])
        intent = _text(expected.get("intent"))
        if intent:
            lines.extend([intent, ""])
        anchors = expected.get("should_surface")
        if isinstance(anchors, list) and anchors:
            lines.extend(["| ID | Label | Role |", "| --- | --- | --- |"])
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md(_text(anchor.get("id"))),
                            _md(_text(anchor.get("label"))),
                            _md(_text(anchor.get("role"))),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        notes = _text(expected.get("evaluation_notes"))
        if notes:
            lines.extend(["Evaluation notes: " + notes, ""])

    lines.extend(["## Ranked Subgraphs", ""])
    if bundle.subgraphs:
        lines.extend(
            [
                "| Rank | Score | Signed Score | Activities | Input Hits | Models | Top Terms |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for subgraph in bundle.subgraphs:
            top_terms = "; ".join(_term_label(term) for term in subgraph.top_terms[:5])
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(subgraph.rank),
                        f"{subgraph.score:.4g}",
                        f"{subgraph.signed_score:.4g}",
                        str(subgraph.activity_count),
                        str(subgraph.input_hit_count),
                        str(len(subgraph.models)),
                        _md(top_terms),
                    ]
                )
                + " |"
            )
        lines.append("")
    else:
        lines.extend(["No subgraphs were returned.", ""])

    if bundle.subgraphs:
        lines.extend(["## Visualizations", ""])
        lines.extend(
            [
                "Static diagrams show the leading activity nodes by diffusion score. "
                "The complete returned subgraphs are in JSON and GraphML.",
                "",
            ]
        )
        for subgraph in bundle.subgraphs:
            links = []
            for suffix, label in (("svg", "SVG"), ("png", "PNG"), ("dot", "DOT")):
                filename = f"subgraph_{subgraph.rank:03d}.{suffix}"
                if output_dir is None or (output_dir / filename).exists():
                    links.append(f"[{label}]({filename})")
            if links:
                lines.append(f"- Subgraph {subgraph.rank}: " + ", ".join(links))
        note_path = output_dir / "graphviz_rendering.txt" if output_dir is not None else None
        if note_path is not None and note_path.exists():
            lines.append("- Graphviz image rendering was skipped; see `graphviz_rendering.txt`.")
        lines.append("")

    if bundle.unmapped_inputs:
        lines.extend(["## Unmapped Inputs", "", "| ID | Type | Modality | Label |", "| --- | --- | --- | --- |"])
        for item in bundle.unmapped_inputs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(item.id),
                        _md(item.type or ""),
                        _md(item.modality or ""),
                        _md(item.label or ""),
                    ]
                )
                + " |"
            )
        lines.append("")

    for subgraph in bundle.subgraphs[:10]:
        lines.extend([f"## Subgraph {subgraph.rank}", ""])
        svg_name = f"subgraph_{subgraph.rank:03d}.svg"
        if output_dir is None or (output_dir / svg_name).exists():
            lines.extend([f"![Subgraph {subgraph.rank}]({svg_name})", ""])
        lines.extend(
            [
                f"- Score: `{subgraph.score:.6g}`",
                f"- Raw score: `{subgraph.raw_score:.6g}`",
                f"- Signed score: `{subgraph.signed_score:.6g}`",
                f"- Activities: `{subgraph.activity_count}`",
                f"- Input hits: `{subgraph.input_hit_count}`",
                f"- Models: `{len(subgraph.models)}`",
                "",
            ]
        )
        if subgraph.top_terms:
            lines.extend(["Top terms:", ""])
            for term in subgraph.top_terms[:8]:
                lines.append(f"- `{term.id}` {_md_inline(term.label or term.id)} (`score={term.score:.4g}`, `count={term.count}`)")
            lines.append("")
        if subgraph.matched_inputs:
            lines.extend(["Matched inputs:", ""])
            for item in subgraph.matched_inputs[:12]:
                label = item.label or item.node_label or item.id
                modality = f", {item.modality}" if item.modality else ""
                lines.append(f"- `{item.id}` {_md_inline(label)} (`value={item.value:g}`{modality})")
            lines.append("")
        if subgraph.activities:
            lines.extend(["Leading activities:", "", "| Activity | Enabled by | Function | Process | Score |", "| --- | --- | --- | --- | ---: |"])
            for activity in subgraph.activities[:10]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md(_activity_label(activity)),
                            _md(_id_label(activity.enabled_by, activity.enabled_by_label)),
                            _md(_id_label(activity.molecular_function, activity.molecular_function_label)),
                            _md(_id_label(activity.biological_process, activity.biological_process_label)),
                            f"{activity.score:.4g}",
                        ]
                    )
                    + " |"
                )
            lines.append("")

    lines.extend(
        [
            "## Machine-Readable Files",
            "",
            "- `ranked_subgraphs.tsv`: ranked subgraph summary.",
            "- `grounding.tsv`: activity-level grounding with labels.",
            "- `subgraph_NNN.dot`: Graphviz source for the static activity-centric diagram.",
            "- `subgraph_NNN.svg`: static diagram rendered by Graphviz when `dot` is installed.",
            "- `subgraph_NNN.png`: static diagram rendered by Graphviz when `dot` is installed.",
            "- `subgraph_NNN.json`: full structured subgraph output.",
            "- `subgraph_NNN.graphml`: graph export for Cytoscape or NetworkX tooling.",
            "- `unmapped_inputs.tsv`: inputs that did not map to the graph.",
            "",
        ]
    )
    return "\n".join(lines)


def _dot_subgraph(subgraph: SubgraphResult, *, max_activities: int) -> str:
    visual_activities = subgraph.activities[:max_activities]
    activity_by_key = {activity.key: activity for activity in visual_activities}
    matched_by_node = {
        item.node: item
        for item in subgraph.matched_inputs
        if item.node is not None
    }
    max_score = max((activity.score for activity in visual_activities), default=0.0)
    matched_activity_keys = _matched_activity_keys(subgraph, set(matched_by_node))

    lines = [
        f"digraph {_dot_quote(f'subgraph_{subgraph.rank:03d}')} {{",
        "  graph [rankdir=LR, bgcolor=\"white\", pad=\"0.25\", nodesep=\"0.45\", ranksep=\"0.75\", splines=true, overlap=false];",
        "  node [fontname=\"Helvetica\", fontsize=10, color=\"#475569\", penwidth=1.4];",
        "  edge [fontname=\"Helvetica\", fontsize=8, color=\"#64748b\", arrowsize=0.7];",
        f"  label={_dot_quote(_dot_graph_label(subgraph))};",
        "  labelloc=\"t\";",
        "  fontsize=16;",
        "",
    ]

    if matched_by_node:
        lines.append("  subgraph cluster_inputs {")
        lines.append("    label=\"Matched inputs\";")
        lines.append("    color=\"#e2e8f0\";")
        lines.append("    style=\"rounded\";")
        for node_key, input_obj in sorted(matched_by_node.items()):
            lines.append(
                "    "
                + _dot_node(
                    node_key,
                    label=_input_dot_label(input_obj),
                    shape=_input_shape(input_obj),
                    style="filled,bold",
                    fillcolor=_modality_color(input_obj.modality),
                    color="#7c3aed",
                    penwidth=2.2,
                )
            )
        lines.append("  }")
        lines.append("")

    top_terms = "\n".join(_wrap(_term_label(term), width=42, max_lines=1) for term in subgraph.top_terms[:8])
    if top_terms:
        lines.append(
            "  "
            + _dot_node(
                f"subgraph:{subgraph.rank}:top_terms",
                label="Top terms\n" + top_terms,
                shape="note",
                style="filled",
                fillcolor="#f8fafc",
                color="#cbd5e1",
            )
        )
        lines.append("")

    if len(visual_activities) < len(subgraph.activities):
        lines.append(
            "  "
            + _dot_node(
                f"subgraph:{subgraph.rank}:omitted",
                label=f"Diagram shows top {len(visual_activities)} of {len(subgraph.activities)} activities\nFull subgraph: JSON/GraphML",
                shape="note",
                style="filled",
                fillcolor="#f8fafc",
                color="#cbd5e1",
            )
        )
        lines.append("")

    for activity in visual_activities:
        hit = activity.key in matched_activity_keys
        lines.append(
            "  "
            + _dot_node(
                activity.key,
                label=_activity_dot_label(activity),
                shape="box",
                style="rounded,filled,bold" if hit else "rounded,filled",
                fillcolor=_activity_color(activity.signed_score),
                color="#7c3aed" if hit else "#334155",
                penwidth=_activity_penwidth(activity.score, max_score, hit=hit),
            )
        )
    lines.append("")

    causal_edges = 0
    for edge in subgraph.edges:
        if edge.source in activity_by_key and edge.target in activity_by_key and (edge.relation or "").startswith("causal"):
            causal_edges += 1
            lines.append(
                "  "
                + _dot_edge(
                    edge.source,
                    edge.target,
                    label=_edge_label(edge),
                    color=_edge_color(edge.sign),
                    penwidth=1.8,
                )
            )
    if causal_edges:
        lines.append("")

    for edge in subgraph.edges:
        if edge.source in matched_by_node and edge.target in activity_by_key:
            lines.append(
                "  "
                + _dot_edge(
                    edge.source,
                    edge.target,
                    label=edge.relation or "",
                    color="#7c3aed",
                    style="dashed",
                )
            )
        elif edge.target in matched_by_node and edge.source in activity_by_key:
            lines.append(
                "  "
                + _dot_edge(
                    edge.source,
                    edge.target,
                    label=edge.relation or "",
                    color="#7c3aed",
                    style="dashed",
                )
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _subgraph_result(rank: int, subgraph: dict[str, Any]) -> SubgraphResult:
    label_by_id = {
        str(node.get("id")): node.get("label")
        for node in subgraph["nodes"]
        if node.get("id") is not None and node.get("label") is not None
    }
    return SubgraphResult(
        rank=rank,
        score=float(subgraph["score"]),
        raw_score=float(subgraph.get("raw_score", subgraph["score"])),
        signed_score=float(subgraph["signed_score"]),
        activity_count=int(subgraph["activity_count"]),
        input_hit_count=int(subgraph["input_hit_count"]),
        models=list(subgraph["models"]),
        top_terms=[TermSummary(**term) for term in subgraph["top_terms"]],
        nodes=[_node_result(node) for node in subgraph["nodes"]],
        edges=[_edge_result(edge) for edge in subgraph["edges"]],
        activities=[_activity_grounding(activity, label_by_id) for activity in subgraph["activities"]],
        matched_inputs=[InputResult(**matched) for matched in subgraph["matched_inputs"]],
        paths=[PathResult(**path) for path in subgraph["paths"]],
    )


def _node_result(node: dict[str, Any]) -> NodeResult:
    core = {"key", "id", "kind", "label", "score", "signed_score"}
    return NodeResult(
        key=str(node["key"]),
        id=str(node.get("id") or original_id(str(node["key"]))),
        kind=node.get("kind"),
        label=node.get("label"),
        score=float(node.get("score") or 0.0),
        signed_score=float(node.get("signed_score") or 0.0),
        attributes={key: value for key, value in node.items() if key not in core},
    )


def _edge_result(edge: dict[str, Any]) -> EdgeResult:
    core = {"source", "target", "key", "relation", "weight", "sign", "model_id", "predicate", "evidence"}
    return EdgeResult(
        source=str(edge["source"]),
        target=str(edge["target"]),
        key=edge.get("key"),
        relation=edge.get("relation"),
        weight=edge.get("weight"),
        sign=edge.get("sign"),
        model_id=edge.get("model_id"),
        predicate=edge.get("predicate"),
        evidence=edge.get("evidence") or [],
        attributes={key: value for key, value in edge.items() if key not in core},
    )


def _activity_grounding(activity: dict[str, Any], label_by_id: dict[str, str]) -> ActivityGrounding:
    core = {
        "key",
        "id",
        "kind",
        "label",
        "score",
        "signed_score",
        "model_id",
        "model_title",
        "taxon",
        "enabled_by",
        "enabled_by_label",
        "molecular_function",
        "molecular_function_label",
        "biological_process",
        "biological_process_label",
        "cellular_component",
        "cellular_component_label",
        "evidence",
    }
    enabled_by = activity.get("enabled_by")
    molecular_function = activity.get("molecular_function")
    biological_process = activity.get("biological_process")
    cellular_component = activity.get("cellular_component")
    return ActivityGrounding(
        key=str(activity["key"]),
        id=str(activity.get("id") or original_id(str(activity["key"]))),
        label=activity.get("label"),
        score=float(activity.get("score") or 0.0),
        signed_score=float(activity.get("signed_score") or 0.0),
        model_id=activity.get("model_id"),
        model_title=activity.get("model_title"),
        taxon=activity.get("taxon"),
        enabled_by=enabled_by,
        enabled_by_label=activity.get("enabled_by_label") or _resolve_label(label_by_id, enabled_by),
        molecular_function=molecular_function,
        molecular_function_label=activity.get("molecular_function_label") or _resolve_label(label_by_id, molecular_function),
        biological_process=biological_process,
        biological_process_label=activity.get("biological_process_label") or _resolve_label(label_by_id, biological_process),
        cellular_component=cellular_component,
        cellular_component_label=activity.get("cellular_component_label") or _resolve_label(label_by_id, cellular_component),
        evidence=activity.get("evidence") or {},
        attributes={key: value for key, value in activity.items() if key not in core},
    )


def _resolve_label(label_by_id: dict[str, str], term_id: str | None) -> str | None:
    if term_id is None:
        return None
    return label_by_id.get(term_id)


def _input_result(mapped: MappedInput) -> InputResult:
    return InputResult(
        id=mapped.input.id,
        type=mapped.input.type,
        modality=mapped.input.modality,
        value=mapped.input.value,
        label=mapped.input.label,
        node=mapped.node,
        match=mapped.match,
    )


def _write_graphml(subgraph: SubgraphResult, path: Path) -> None:
    graph = nx.MultiDiGraph()
    for node in subgraph.nodes:
        attrs = {
            "id": node.id,
            "kind": node.kind or "",
            "label": node.label or "",
            "score": node.score,
            "signed_score": node.signed_score,
            **{key: _graphml_value(value) for key, value in node.attributes.items()},
        }
        graph.add_node(node.key, **attrs)
    for edge in subgraph.edges:
        attrs = {
            "relation": edge.relation or "",
            "weight": edge.weight or 0.0,
            "sign": edge.sign or 1,
            "model_id": edge.model_id or "",
            "predicate": edge.predicate or "",
            "evidence": _graphml_value(edge.evidence),
            **{key: _graphml_value(value) for key, value in edge.attributes.items()},
        }
        graph.add_edge(edge.source, edge.target, **attrs)
    nx.write_graphml(graph, path)


def _matched_activity_keys(subgraph: SubgraphResult, matched_nodes: set[str]) -> set[str]:
    matched = {node for node in matched_nodes if node.startswith("activity:")}
    for edge in subgraph.edges:
        if edge.source in matched_nodes and edge.target.startswith("activity:"):
            matched.add(edge.target)
        if edge.target in matched_nodes and edge.source.startswith("activity:"):
            matched.add(edge.source)
    return matched


def _dot_graph_label(subgraph: SubgraphResult) -> str:
    top_terms = ", ".join(term.label or term.id for term in subgraph.top_terms[:3])
    return (
        f"Subgraph {subgraph.rank} | score {subgraph.score:.4g} | "
        f"{subgraph.activity_count} activities | {subgraph.input_hit_count} input hits"
        + (f" | {top_terms}" if top_terms else "")
    )


def _dot_node(
    node_id: str,
    *,
    label: str,
    shape: str,
    style: str,
    fillcolor: str,
    color: str,
    penwidth: float = 1.4,
) -> str:
    attrs = {
        "label": label,
        "shape": shape,
        "style": style,
        "fillcolor": fillcolor,
        "color": color,
        "penwidth": f"{penwidth:.2f}",
    }
    return f"{_dot_quote(node_id)} [{_dot_attrs(attrs)}];"


def _dot_edge(
    source: str,
    target: str,
    *,
    label: str = "",
    color: str = "#64748b",
    style: str = "solid",
    penwidth: float = 1.4,
) -> str:
    attrs = {
        "label": label,
        "color": color,
        "fontcolor": color,
        "style": style,
        "penwidth": f"{penwidth:.2f}",
    }
    return f"{_dot_quote(source)} -> {_dot_quote(target)} [{_dot_attrs(attrs)}];"


def _dot_attrs(attrs: dict[str, object]) -> str:
    return ", ".join(f"{key}={_dot_quote(value)}" for key, value in attrs.items() if value != "")


def _dot_quote(value: object) -> str:
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _activity_dot_label(activity: ActivityGrounding) -> str:
    parts = [
        _wrap(activity.label or activity.enabled_by_label or activity.id, width=34, max_lines=2),
        _wrap(activity.id, width=38, max_lines=1),
    ]
    enabled_by = _id_label(activity.enabled_by, activity.enabled_by_label)
    molecular_function = _id_label(activity.molecular_function, activity.molecular_function_label)
    biological_process = _id_label(activity.biological_process, activity.biological_process_label)
    cellular_component = _id_label(activity.cellular_component, activity.cellular_component_label)
    if enabled_by:
        parts.append("gene: " + _wrap(enabled_by, width=34, max_lines=2))
    if molecular_function:
        parts.append("MF: " + _wrap(molecular_function, width=34, max_lines=2))
    if biological_process:
        parts.append("BP: " + _wrap(biological_process, width=34, max_lines=2))
    if cellular_component:
        parts.append("CC: " + _wrap(cellular_component, width=34, max_lines=1))
    parts.append(f"score={activity.score:.3g} signed={activity.signed_score:.3g}")
    return "\n".join(part for part in parts if part)


def _input_dot_label(input_obj: InputResult) -> str:
    parts = [
        _wrap(input_obj.label or input_obj.node_label or input_obj.id, width=30, max_lines=2),
        _wrap(input_obj.id, width=34, max_lines=1),
    ]
    if input_obj.modality or input_obj.type:
        parts.append(" / ".join(value for value in (input_obj.type, input_obj.modality) if value))
    parts.append(f"value={input_obj.value:g}")
    return "\n".join(parts)


def _activity_penwidth(score: float, max_score: float, *, hit: bool) -> float:
    if max_score <= 0:
        return 2.4 if hit else 1.4
    return min(6.0, (2.4 if hit else 1.4) + 3.2 * score / max_score)


def _activity_color(signed_score: float) -> str:
    if signed_score > 0:
        return "#fee2e2"
    if signed_score < 0:
        return "#dbeafe"
    return "#f8fafc"


def _edge_color(sign: int | None) -> str:
    if sign is not None and sign < 0:
        return "#dc2626"
    if sign is not None and sign > 0:
        return "#16a34a"
    return "#64748b"


def _edge_label(edge: EdgeResult) -> str:
    return edge.predicate or edge.relation or ""


def _input_shape(input_obj: InputResult) -> str:
    if input_obj.type == "chemical":
        return "ellipse"
    if input_obj.type == "go_term":
        return "note"
    return "oval"


def _modality_color(modality: str | None) -> str:
    colors = {
        "transcriptomics": "#dbeafe",
        "proteomics": "#dcfce7",
        "phosphoproteomics": "#ede9fe",
        "metabolomics": "#ffedd5",
        "semantic": "#fef9c3",
    }
    return colors.get(modality or "", "#f8fafc")


def _wrap(value: object | None, *, width: int, max_lines: int) -> str:
    words = _text(value).split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word)
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def _graphml_value(value: Any) -> str | int | float:
    if isinstance(value, (str, int, float)):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _tsv(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\t", " ").replace("\n", " ").strip()


def _text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)


def _labelize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _md(value: object | None) -> str:
    text = _text(value)
    return text.replace("|", "\\|")


def _md_inline(value: object | None) -> str:
    return _text(value).replace("[", "\\[").replace("]", "\\]")


def _term_label(term: TermSummary) -> str:
    label = term.label or term.id
    if label == term.id:
        return term.id
    return f"{label} ({term.id})"


def _id_label(identifier: str | None, label: str | None) -> str:
    if not identifier and not label:
        return ""
    if identifier and label and identifier != label:
        return f"{label} ({identifier})"
    return label or identifier or ""


def _activity_label(activity: ActivityGrounding) -> str:
    return _id_label(activity.id, activity.label)
