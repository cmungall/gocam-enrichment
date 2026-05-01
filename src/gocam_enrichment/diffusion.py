from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Any

import networkx as nx

from gocam_enrichment.gocam_graph import (
    ACTIVITY_PREFIX,
    aggregate_parallel_edges,
    activity_key,
    original_id,
    term_key,
)
from gocam_enrichment.model_io import InputObject


@dataclass(frozen=True)
class MappedInput:
    input: InputObject
    node: str | None


def map_inputs(graph: nx.MultiDiGraph, inputs: list[InputObject]) -> list[MappedInput]:
    mapped: list[MappedInput] = []
    for input_obj in inputs:
        term = term_key(input_obj.id)
        activity = activity_key(input_obj.id)
        if graph.has_node(term):
            mapped.append(MappedInput(input_obj, term))
        elif graph.has_node(activity):
            mapped.append(MappedInput(input_obj, activity))
        else:
            mapped.append(MappedInput(input_obj, None))
    return mapped


def diffuse(
    graph: nx.MultiDiGraph,
    mapped_inputs: list[MappedInput],
    *,
    restart: float = 0.35,
    steps: int = 40,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run simple random-walk-with-restart diffusion over weighted graph edges."""

    seeds = _seed_vector(mapped_inputs, signed=False)
    signed_seeds = _seed_vector(mapped_inputs, signed=True)
    adjacency = aggregate_parallel_edges(graph)
    scores = dict(seeds)
    signed_scores = dict(signed_seeds)
    out_weight = {
        node: sum(weight for weight, _sign in targets.values())
        for node, targets in adjacency.items()
    }

    for _ in range(steps):
        scores = _step(scores, seeds, adjacency, out_weight, restart, signed=False)
        signed_scores = _step(signed_scores, signed_seeds, adjacency, out_weight, restart, signed=True)
    return scores, signed_scores


def _seed_vector(mapped_inputs: list[MappedInput], *, signed: bool) -> dict[str, float]:
    seeds: dict[str, float] = defaultdict(float)
    for mapped in mapped_inputs:
        if mapped.node is None:
            continue
        value = mapped.input.value if signed else mapped.input.abs_weight
        seeds[mapped.node] += value
    normalizer = sum(abs(value) for value in seeds.values())
    if normalizer == 0:
        return dict(seeds)
    return {node: value / normalizer for node, value in seeds.items()}


def _step(
    scores: dict[str, float],
    seeds: dict[str, float],
    adjacency: dict[str, dict[str, tuple[float, int]]],
    out_weight: dict[str, float],
    restart: float,
    *,
    signed: bool,
) -> dict[str, float]:
    next_scores: dict[str, float] = defaultdict(float)
    for node, value in seeds.items():
        next_scores[node] += restart * value
    for source, value in scores.items():
        denom = out_weight.get(source, 0.0)
        if denom <= 0:
            continue
        for target, (weight, sign) in adjacency.get(source, {}).items():
            propagated = value * weight / denom
            if signed:
                propagated *= sign
            next_scores[target] += (1.0 - restart) * propagated
    return dict(next_scores)


def extract_subgraphs(
    graph: nx.MultiDiGraph,
    mapped_inputs: list[MappedInput],
    scores: dict[str, float],
    signed_scores: dict[str, float],
    *,
    max_subgraphs: int = 20,
    radius: int = 1,
    max_bridge_degree: int = 50,
    max_subgraph_activities: int = 150,
) -> list[dict[str, Any]]:
    """Extract local high-scoring activity neighborhoods."""

    ranked_activities = [
        node
        for node, data in sorted(
            graph.nodes(data=True),
            key=lambda item: scores.get(item[0], 0.0),
            reverse=True,
        )
        if data.get("kind") == "activity" and scores.get(node, 0.0) > 0
    ]

    clusters: list[set[str]] = []
    bridge_degrees = _bridge_degrees(graph)
    for seed in ranked_activities:
        neighborhood = _activity_neighborhood(
            graph,
            seed,
            radius=radius,
            bridge_degrees=bridge_degrees,
            max_bridge_degree=max_bridge_degree,
        )
        if not neighborhood:
            continue
        merged = False
        for cluster in clusters:
            if cluster & neighborhood:
                candidate = cluster | neighborhood
                cluster.clear()
                cluster.update(_top_activity_subset(candidate, scores, max_subgraph_activities))
                merged = True
                break
        if not merged:
            clusters.append(_top_activity_subset(neighborhood, scores, max_subgraph_activities))
        if len(clusters) >= max_subgraphs * 3:
            break

    subgraphs = []
    for cluster in clusters:
        subgraphs.append(_materialize_subgraph(graph, cluster, mapped_inputs, scores, signed_scores))

    subgraphs.sort(key=lambda item: item["score"], reverse=True)
    return subgraphs[:max_subgraphs]


def _activity_neighborhood(
    graph: nx.MultiDiGraph,
    seed: str,
    *,
    radius: int,
    bridge_degrees: dict[str, int],
    max_bridge_degree: int,
) -> set[str]:
    selected = {seed}
    frontier = {seed}
    for _ in range(radius):
        next_frontier: set[str] = set()
        for activity in frontier:
            for neighbor in _causal_activity_neighbors(graph, activity):
                if neighbor not in selected:
                    selected.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    bridged = set(selected)
    for activity in selected:
        for term in _object_terms_attached_to_activity(
            graph,
            activity,
            bridge_degrees=bridge_degrees,
            max_bridge_degree=max_bridge_degree,
        ):
            bridged.update(_activities_attached_to_term(graph, term))
    return bridged


def _causal_activity_neighbors(graph: nx.MultiDiGraph, activity: str) -> set[str]:
    neighbors: set[str] = set()
    for source, target, attrs in graph.out_edges(activity, data=True):
        if attrs.get("relation", "").startswith("causal") and target.startswith(ACTIVITY_PREFIX):
            neighbors.add(target)
    for source, target, attrs in graph.in_edges(activity, data=True):
        if attrs.get("relation", "").startswith("causal") and source.startswith(ACTIVITY_PREFIX):
            neighbors.add(source)
    return neighbors


def _object_terms_attached_to_activity(
    graph: nx.MultiDiGraph,
    activity: str,
    *,
    bridge_degrees: dict[str, int],
    max_bridge_degree: int,
) -> set[str]:
    terms: set[str] = set()
    bridge_kinds = {"gene_product", "protein_complex", "chemical"}
    for source, _target, attrs in graph.in_edges(activity, data=True):
        if bridge_degrees.get(source, 0) > max_bridge_degree:
            continue
        if graph.nodes[source].get("kind") in bridge_kinds and attrs.get("relation") in {
            "enabled_by",
            "complex_member",
            "has_input",
            "has_primary_input",
            "has_output",
            "has_primary_output",
        }:
            terms.add(source)
    return terms


def _bridge_degrees(graph: nx.MultiDiGraph) -> dict[str, int]:
    bridge_relations = {
        "enabled_by",
        "complex_member",
        "has_input",
        "has_primary_input",
        "has_output",
        "has_primary_output",
    }
    degrees: dict[str, set[str]] = defaultdict(set)
    for source, target, attrs in graph.edges(data=True):
        if target.startswith(ACTIVITY_PREFIX) and attrs.get("relation") in bridge_relations:
            degrees[source].add(target)
    return {node: len(activities) for node, activities in degrees.items()}


def _top_activity_subset(
    activities: set[str],
    scores: dict[str, float],
    max_activities: int,
) -> set[str]:
    if len(activities) <= max_activities:
        return set(activities)
    return set(sorted(activities, key=lambda node: scores.get(node, 0.0), reverse=True)[:max_activities])


def _activities_attached_to_term(graph: nx.MultiDiGraph, term: str) -> set[str]:
    activities: set[str] = set()
    for _source, target, attrs in graph.out_edges(term, data=True):
        if target.startswith(ACTIVITY_PREFIX) and attrs.get("relation") in {
            "enabled_by",
            "complex_member",
            "has_input",
            "has_primary_input",
            "has_output",
            "has_primary_output",
        }:
            activities.add(target)
    return activities


def _materialize_subgraph(
    graph: nx.MultiDiGraph,
    activity_nodes: set[str],
    mapped_inputs: list[MappedInput],
    scores: dict[str, float],
    signed_scores: dict[str, float],
) -> dict[str, Any]:
    attached_terms = _attached_terms_for_activities(graph, activity_nodes)
    node_set = set(activity_nodes) | attached_terms
    edges = [
        _edge_payload(source, target, key, attrs)
        for source, target, key, attrs in graph.edges(keys=True, data=True)
        if source in node_set and target in node_set
    ]
    nodes = [_node_payload(node, graph.nodes[node], scores, signed_scores) for node in sorted(node_set)]
    activities = [
        _node_payload(node, graph.nodes[node], scores, signed_scores)
        for node in sorted(activity_nodes, key=lambda item: scores.get(item, 0.0), reverse=True)
    ]
    matched_inputs = [
        {
            "id": mapped.input.id,
            "type": mapped.input.type,
            "modality": mapped.input.modality,
            "value": mapped.input.value,
            "label": mapped.input.label,
            "node": mapped.node,
            "node_id": graph.nodes[mapped.node].get("id") if mapped.node is not None else None,
            "node_kind": graph.nodes[mapped.node].get("kind") if mapped.node is not None else None,
            "node_label": graph.nodes[mapped.node].get("label") if mapped.node is not None else None,
        }
        for mapped in mapped_inputs
        if mapped.node in node_set
    ]
    models = sorted({graph.nodes[node].get("model_id") for node in activity_nodes if graph.nodes[node].get("model_id")})
    top_terms = _top_terms(graph, activity_nodes, scores)
    raw_score = sum(scores.get(node, 0.0) for node in activity_nodes)
    score = raw_score / sqrt(max(len(activity_nodes), 1))
    return {
        "score": score,
        "raw_score": raw_score,
        "signed_score": sum(signed_scores.get(node, 0.0) for node in activity_nodes),
        "activity_count": len(activity_nodes),
        "input_hit_count": len(matched_inputs),
        "models": models,
        "top_terms": top_terms,
        "nodes": nodes,
        "edges": edges,
        "activities": activities,
        "matched_inputs": matched_inputs,
        "paths": _input_paths(graph, node_set, matched_inputs, activity_nodes),
    }


def _attached_terms_for_activities(graph: nx.MultiDiGraph, activity_nodes: set[str]) -> set[str]:
    terms: set[str] = set()
    for activity in activity_nodes:
        for source, _target in graph.in_edges(activity):
            if not source.startswith(ACTIVITY_PREFIX):
                terms.add(source)
        for _source, target in graph.out_edges(activity):
            if not target.startswith(ACTIVITY_PREFIX):
                terms.add(target)
    return terms


def _node_payload(
    node: str,
    attrs: dict[str, Any],
    scores: dict[str, float],
    signed_scores: dict[str, float],
) -> dict[str, Any]:
    return {
        "key": node,
        "id": attrs.get("id", original_id(node)),
        "kind": attrs.get("kind"),
        "label": attrs.get("label"),
        "score": scores.get(node, 0.0),
        "signed_score": signed_scores.get(node, 0.0),
        **{key: value for key, value in attrs.items() if key not in {"id", "kind", "label"}},
    }


def _edge_payload(source: str, target: str, key: int, attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "key": key,
        **attrs,
    }


def _top_terms(
    graph: nx.MultiDiGraph,
    activity_nodes: set[str],
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for activity in activity_nodes:
        data = graph.nodes[activity]
        activity_score = scores.get(activity, 0.0)
        for slot in ("biological_process", "molecular_function", "cellular_component", "enabled_by"):
            term_id = data.get(slot)
            if not term_id:
                continue
            term = term_key(term_id)
            term_data = graph.nodes.get(term, {})
            entry = counts.setdefault(
                term,
                {
                    "id": term_id,
                    "kind": term_data.get("kind", slot),
                    "label": term_data.get("label", term_id),
                    "count": 0,
                    "score": 0.0,
                },
            )
            entry["count"] += 1
            entry["score"] += activity_score
    return sorted(counts.values(), key=lambda item: item["score"], reverse=True)[:12]


def _input_paths(
    graph: nx.MultiDiGraph,
    node_set: set[str],
    matched_inputs: list[dict[str, Any]],
    activity_nodes: set[str],
) -> list[dict[str, Any]]:
    simple = nx.Graph()
    simple.add_nodes_from(node_set)
    for source, target, attrs in graph.edges(data=True):
        if source in node_set and target in node_set and float(attrs.get("weight") or 0.0) > 0:
            simple.add_edge(source, target)
    paths: list[dict[str, Any]] = []
    ranked_activities = list(activity_nodes)[:5]
    for matched in matched_inputs:
        node = matched.get("node")
        if not node or node not in simple:
            continue
        for activity in ranked_activities:
            if activity not in simple:
                continue
            try:
                path = nx.shortest_path(simple, node, activity)
            except nx.NetworkXNoPath:
                continue
            if len(path) <= 5:
                paths.append(
                    {
                        "input_id": matched["id"],
                        "activity_id": original_id(activity),
                        "path": path,
                        "path_labels": [graph.nodes[node].get("label") for node in path],
                    }
                )
                break
    return paths
