from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InputResult(BaseModel):
    id: str
    type: str | None = None
    modality: str | None = None
    value: float = 1.0
    label: str | None = None
    node: str | None = None
    match: str | None = None
    node_id: str | None = None
    node_kind: str | None = None
    node_label: str | None = None


class TermSummary(BaseModel):
    id: str
    label: str | None = None
    kind: str | None = None
    count: int = 0
    score: float = 0.0


class NodeResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    key: str
    id: str
    kind: str | None = None
    label: str | None = None
    score: float = 0.0
    signed_score: float = 0.0
    attributes: dict[str, Any] = {}


class EdgeResult(BaseModel):
    source: str
    target: str
    key: int | str | None = None
    relation: str | None = None
    weight: float | None = None
    sign: int | None = None
    model_id: str | None = None
    predicate: str | None = None
    evidence: list[dict[str, Any]] = []
    attributes: dict[str, Any] = {}


class ActivityGrounding(BaseModel):
    key: str
    id: str
    label: str | None = None
    score: float = 0.0
    signed_score: float = 0.0
    model_id: str | None = None
    model_title: str | None = None
    taxon: str | None = None
    enabled_by: str | None = None
    enabled_by_label: str | None = None
    molecular_function: str | None = None
    molecular_function_label: str | None = None
    biological_process: str | None = None
    biological_process_label: str | None = None
    cellular_component: str | None = None
    cellular_component_label: str | None = None
    evidence: dict[str, list[dict[str, Any]]] = {}
    attributes: dict[str, Any] = {}


class PathResult(BaseModel):
    input_id: str
    activity_id: str
    path: list[str]
    path_labels: list[str | None] = []


class SubgraphResult(BaseModel):
    rank: int
    score: float
    raw_score: float
    signed_score: float
    activity_count: int
    input_hit_count: int
    models: list[str]
    top_terms: list[TermSummary]
    nodes: list[NodeResult]
    edges: list[EdgeResult]
    activities: list[ActivityGrounding]
    matched_inputs: list[InputResult]
    paths: list[PathResult]


class ResultBundle(BaseModel):
    input_count: int
    mapped_input_count: int
    subgraphs: list[SubgraphResult]
    unmapped_inputs: list[InputResult]
    input_path: str | None = None
    input_metadata: dict[str, Any] = Field(default_factory=dict)
