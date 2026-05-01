from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from gocam.datamodel import Activity, Association, Model

ACTIVITY_PREFIX = "activity:"
TERM_PREFIX = "term:"
MODEL_PREFIX = "model:"

POSITIVE_PREDICATES = {
    "RO:0002213",
    "RO:0002304",
    "RO:0002406",
    "RO:0002629",
}
NEGATIVE_PREDICATES = {
    "RO:0002212",
    "RO:0002305",
    "RO:0002408",
    "RO:0002630",
}


def activity_key(activity_id: str) -> str:
    return f"{ACTIVITY_PREFIX}{activity_id}"


def term_key(term_id: str) -> str:
    return f"{TERM_PREFIX}{term_id}"


def model_key(model_id: str) -> str:
    return f"{MODEL_PREFIX}{model_id}"


def original_id(node_key: str) -> str:
    if node_key.startswith(ACTIVITY_PREFIX):
        return node_key[len(ACTIVITY_PREFIX) :]
    if node_key.startswith(TERM_PREFIX):
        return node_key[len(TERM_PREFIX) :]
    if node_key.startswith(MODEL_PREFIX):
        return node_key[len(MODEL_PREFIX) :]
    return node_key


def build_graph(models: Iterable[Model]) -> nx.MultiDiGraph:
    """Build a heterogeneous GO-CAM graph from typed model objects."""

    graph = nx.MultiDiGraph()
    for model in models:
        _add_model(graph, model)
    graph.graph["model_count"] = len(
        {data["model_id"] for _, data in graph.nodes(data=True) if data.get("kind") == "model"}
    )
    graph.graph["activity_count"] = sum(1 for _, data in graph.nodes(data=True) if data.get("kind") == "activity")
    return graph


def _add_model(graph: nx.MultiDiGraph, model: Model) -> None:
    labels = {obj.id: obj.label for obj in model.objects or [] if obj.id and obj.label}
    graph.add_node(
        model_key(model.id),
        kind="model",
        id=model.id,
        model_id=model.id,
        label=model.title,
        taxon=model.taxon,
        status=str(model.status) if model.status is not None else None,
    )

    for activity in model.activities or []:
        _add_activity_node(graph, model, activity, labels)

    activity_ids = {activity.id for activity in model.activities or []}
    for activity in model.activities or []:
        source = activity_key(activity.id)
        for causal in activity.causal_associations or []:
            if not causal.downstream_activity or causal.downstream_activity not in activity_ids:
                continue
            target = activity_key(causal.downstream_activity)
            sign = predicate_sign(causal.predicate)
            _add_edge(
                graph,
                source,
                target,
                "causal",
                weight=2.0,
                sign=sign,
                model_id=model.id,
                predicate=causal.predicate,
                evidence=evidence_payload(causal),
            )
            _add_edge(
                graph,
                target,
                source,
                "causal_reverse",
                weight=0.8,
                sign=sign,
                model_id=model.id,
                predicate=causal.predicate,
                evidence=evidence_payload(causal),
            )


def _add_activity_node(
    graph: nx.MultiDiGraph,
    model: Model,
    activity: Activity,
    labels: dict[str, str],
) -> None:
    enabled_by = getattr(activity.enabled_by, "term", None)
    mf = getattr(activity.molecular_function, "term", None)
    bp = getattr(activity.part_of, "term", None)
    cc = getattr(activity.occurs_in, "term", None)
    graph.add_node(
        activity_key(activity.id),
        kind="activity",
        id=activity.id,
        label=labels.get(enabled_by, enabled_by or activity.id),
        model_id=model.id,
        model_title=model.title,
        taxon=model.taxon,
        enabled_by=enabled_by,
        enabled_by_label=_label(labels, enabled_by),
        molecular_function=mf,
        molecular_function_label=_label(labels, mf),
        biological_process=bp,
        biological_process_label=_label(labels, bp),
        cellular_component=cc,
        cellular_component_label=_label(labels, cc),
        evidence={
            "enabled_by": evidence_payload(activity.enabled_by),
            "molecular_function": evidence_payload(activity.molecular_function),
            "part_of": evidence_payload(activity.part_of),
            "occurs_in": evidence_payload(activity.occurs_in),
        },
    )

    enabled_by_kind = "protein_complex" if _is_complex_enabler(activity.enabled_by) else "gene_product"
    _connect_association(graph, model, activity, labels, activity.enabled_by, "enabled_by", enabled_by_kind, 4.0)
    members = getattr(activity.enabled_by, "members", None)
    for member in members or []:
        _connect_term(
            graph,
            model,
            activity,
            labels,
            member,
            "complex_member",
            "gene_product",
            weight=4.0,
            evidence=evidence_payload(activity.enabled_by),
        )
    _connect_association(
        graph,
        model,
        activity,
        labels,
        activity.molecular_function,
        "molecular_function",
        "molecular_function",
        1.5,
    )
    _connect_association(graph, model, activity, labels, activity.part_of, "part_of", "biological_process", 1.5)
    _connect_association(graph, model, activity, labels, activity.occurs_in, "occurs_in", "cellular_component", 1.2)
    for assoc in activity.has_input or []:
        _connect_association(graph, model, activity, labels, assoc, "has_input", "chemical", 3.0)
    if activity.has_primary_input:
        _connect_association(graph, model, activity, labels, activity.has_primary_input, "has_primary_input", "chemical", 3.0)
    for assoc in activity.has_output or []:
        _connect_association(graph, model, activity, labels, assoc, "has_output", "chemical", 3.0)
    if activity.has_primary_output:
        _connect_association(graph, model, activity, labels, activity.has_primary_output, "has_primary_output", "chemical", 3.0)


def _connect_association(
    graph: nx.MultiDiGraph,
    model: Model,
    activity: Activity,
    labels: dict[str, str],
    association: Association | None,
    relation: str,
    term_kind: str,
    weight: float,
) -> None:
    term_id = getattr(association, "term", None)
    if not term_id:
        return
    _connect_term(
        graph,
        model,
        activity,
        labels,
        term_id,
        relation,
        term_kind,
        weight=weight,
        evidence=evidence_payload(association),
    )


def _label(labels: dict[str, str], term_id: str | None) -> str | None:
    if term_id is None:
        return None
    return labels.get(term_id, term_id)


def _is_complex_enabler(association: Association | None) -> bool:
    if association is None:
        return False
    assoc_type = getattr(association, "type", "")
    return assoc_type == "EnabledByProteinComplexAssociation" or bool(getattr(association, "members", None))


def _connect_term(
    graph: nx.MultiDiGraph,
    model: Model,
    activity: Activity,
    labels: dict[str, str],
    term_id: str,
    relation: str,
    term_kind: str,
    *,
    weight: float,
    evidence: list[dict[str, Any]],
) -> None:
    term = term_key(term_id)
    activity_node = activity_key(activity.id)
    graph.add_node(
        term,
        kind=term_kind,
        id=term_id,
        label=labels.get(term_id, term_id),
        taxon=model.taxon,
    )
    _add_edge(
        graph,
        term,
        activity_node,
        relation,
        weight=weight,
        sign=1,
        model_id=model.id,
        evidence=evidence,
    )
    _add_edge(
        graph,
        activity_node,
        term,
        f"{relation}_reverse",
        weight=max(weight * 0.25, 0.2),
        sign=1,
        model_id=model.id,
        evidence=evidence,
    )


def _add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relation: str,
    *,
    weight: float,
    sign: int,
    model_id: str,
    predicate: str | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> None:
    graph.add_edge(
        source,
        target,
        relation=relation,
        weight=weight,
        sign=sign,
        model_id=model_id,
        predicate=predicate,
        evidence=evidence or [],
    )


def predicate_sign(predicate: str | None) -> int:
    if predicate is None:
        return 1
    normalized = predicate.lower()
    if predicate in NEGATIVE_PREDICATES or "negative" in normalized or "inhibit" in normalized:
        return -1
    if predicate in POSITIVE_PREDICATES or "positive" in normalized or "activate" in normalized:
        return 1
    return 1


def evidence_payload(association: Association | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if association is None or association.evidence is None:
        return payload
    for evidence in association.evidence:
        payload.append(evidence.model_dump(exclude_none=True))
    return payload


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "graph": dict(graph.graph),
        "nodes": [{"key": node, **attrs} for node, attrs in graph.nodes(data=True)],
        "edges": [
            {"source": source, "target": target, "key": key, **attrs}
            for source, target, key, attrs in graph.edges(keys=True, data=True)
        ],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_graph(path: Path) -> nx.MultiDiGraph:
    data = json.loads(path.read_text())
    graph = nx.MultiDiGraph()
    graph.graph.update(data.get("graph", {}))
    for node in data.get("nodes", []):
        key = node.pop("key")
        graph.add_node(key, **node)
    for edge in data.get("edges", []):
        source = edge.pop("source")
        target = edge.pop("target")
        key = edge.pop("key", None)
        graph.add_edge(source, target, key=key, **edge)
    return graph


def filter_graph_by_taxon(graph: nx.MultiDiGraph, taxon: str | None) -> nx.MultiDiGraph:
    """Return a taxon-scoped graph while keeping terms attached to kept activities."""

    if taxon is None:
        return graph

    kept_activities = {
        node
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "activity" and data.get("taxon") == taxon
    }
    kept_models = {
        node
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "model" and data.get("taxon") == taxon
    }
    kept_terms: set[str] = set()
    for activity in kept_activities:
        kept_terms.update(source for source, _target in graph.in_edges(activity))
        kept_terms.update(target for _source, target in graph.out_edges(activity))

    kept_nodes = kept_activities | kept_models | {node for node in kept_terms if graph.has_node(node)}
    subgraph = graph.subgraph(kept_nodes).copy()
    subgraph.graph.update(graph.graph)
    subgraph.graph["taxon_filter"] = taxon
    return subgraph


def graph_summary(graph: nx.MultiDiGraph) -> dict[str, int]:
    kinds = Counter(data.get("kind", "unknown") for _, data in graph.nodes(data=True))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "models": kinds.get("model", 0),
        "activities": kinds.get("activity", 0),
        "terms": graph.number_of_nodes() - kinds.get("activity", 0) - kinds.get("model", 0),
    }


def aggregate_parallel_edges(graph: nx.MultiDiGraph) -> dict[str, dict[str, tuple[float, int]]]:
    """Return adjacency with parallel edge weights summed for diffusion."""

    adjacency: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 1.0]))
    for source, target, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight") or 0.0)
        if weight <= 0:
            continue
        sign = int(attrs.get("sign") or 1)
        adjacency[source][target][0] += weight
        adjacency[source][target][1] = sign
    return {
        source: {target: (values[0], int(values[1])) for target, values in targets.items()}
        for source, targets in adjacency.items()
    }
