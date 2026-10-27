from __future__ import annotations

import networkx as nx

from gocam_enrichment.chemicals import ChemicalIndex
from gocam_enrichment.diffusion import diffuse, extract_subgraphs, map_inputs
from gocam_enrichment.gocam_graph import _add_edge, activity_key, term_key
from gocam_enrichment.model_io import InputObject

CITRATE = "CHEBI:16947"  # citrate(3-), the form GO-CAM uses
CITRIC_ACID = "CHEBI:30769"
ATP = "CHEBI:30616"


def _graph() -> nx.MultiDiGraph:
    """a1 consumes citrate; a1 and a2 both use ATP and are otherwise unconnected."""

    graph = nx.MultiDiGraph()
    for activity in ("a1", "a2"):
        graph.add_node(activity_key(activity), kind="activity", id=activity, model_id="m")
    for chebi, label in ((CITRATE, "citrate(3-)"), (ATP, "ATP")):
        graph.add_node(term_key(chebi), kind="chemical", id=chebi, label=label)

    def link(chemical: str, activity: str) -> None:
        _add_edge(graph, term_key(chemical), activity_key(activity), "has_input", weight=3.0, sign=1, model_id="m")
        _add_edge(graph, activity_key(activity), term_key(chemical), "has_input_reverse", weight=0.75, sign=1, model_id="m")

    link(CITRATE, "a1")
    link(ATP, "a1")
    link(ATP, "a2")
    return graph


def _index() -> ChemicalIndex:
    return ChemicalIndex(
        to_canonical={CITRATE: CITRATE, CITRIC_ACID: CITRATE, ATP: ATP, "CHEBI:15422": ATP},
        xrefs={"hmdb:HMDB0000094": [CITRIC_ACID]},
    )


def test_map_inputs_normalizes_chemical_ids():
    graph = _graph()
    inputs = [
        InputObject(id="HMDB00094", type="chemical"),
        InputObject(id="chebi:30769", type="chemical"),
        InputObject(id=CITRATE, type="chemical"),
        InputObject(id="HMDB9999999", type="chemical"),
    ]

    without = map_inputs(graph, inputs)
    assert [m.node for m in without] == [None, None, term_key(CITRATE), None]

    mapped = map_inputs(graph, inputs, _index())
    assert [m.node for m in mapped] == [term_key(CITRATE)] * 3 + [None]
    assert [m.match for m in mapped] == ["chemical_normalized", "chemical_normalized", "exact", None]


def test_currency_metabolites_do_not_bridge_activities():
    graph = _graph()
    mapped = map_inputs(graph, [InputObject(id=CITRATE, type="chemical")])

    open_scores, _ = diffuse(graph, mapped)
    blocked_scores, _ = diffuse(graph, mapped, is_currency=_index().is_currency)

    assert open_scores[activity_key("a2")] > 0
    assert blocked_scores.get(activity_key("a2"), 0.0) == 0.0
    assert blocked_scores.get(term_key(ATP), 0.0) == 0.0
    assert blocked_scores[activity_key("a1")] > 0


def test_currency_seed_still_emits_signal():
    graph = _graph()
    mapped = map_inputs(graph, [InputObject(id="CHEBI:15422", type="chemical")], _index())
    assert mapped[0].node == term_key(ATP)

    scores, _ = diffuse(graph, mapped, is_currency=_index().is_currency)
    assert scores[activity_key("a1")] > 0
    assert scores[activity_key("a2")] > 0


def test_hub_damping_reduces_flow_into_shared_chemicals():
    graph = _graph()
    mapped = map_inputs(graph, [InputObject(id=CITRATE, type="chemical")])

    plain, _ = diffuse(graph, mapped)
    damped, _ = diffuse(graph, mapped, chemical_hub_alpha=1.0)
    assert damped[term_key(ATP)] < plain[term_key(ATP)]


def test_equivalent_inputs_seed_the_same_node():
    graph = _graph()
    graph.add_node(term_key(CITRIC_ACID), kind="chemical", id=CITRIC_ACID, label="citric acid")
    _add_edge(graph, term_key(CITRIC_ACID), activity_key("a2"), "has_input", weight=3.0, sign=1, model_id="m")

    mapped = map_inputs(
        graph,
        [InputObject(id=CITRIC_ACID), InputObject(id="HMDB0000094"), InputObject(id=CITRATE)],
        _index(),
    )
    assert {m.node for m in mapped} == {term_key(CITRATE)}
    assert [m.match for m in mapped] == ["chemical_normalized", "chemical_normalized", "exact"]


def test_multi_target_xref_uses_candidate_present_in_graph():
    index = _index()
    index.to_canonical["CHEBI:4167"] = "CHEBI:4167"  # absent from the graph, lower number
    index.xrefs["kegg.compound:C00158"] = ["CHEBI:4167", CITRIC_ACID]

    mapped = map_inputs(_graph(), [InputObject(id="C00158")], index)
    assert mapped[0].node == term_key(CITRATE)


def test_extraction_does_not_bridge_through_currency():
    graph = _graph()
    mapped = map_inputs(graph, [InputObject(id=CITRATE)], _index())
    scores, signed = diffuse(graph, mapped, is_currency=_index().is_currency)

    open_subgraphs = extract_subgraphs(graph, mapped, scores, signed)
    blocked_subgraphs = extract_subgraphs(graph, mapped, scores, signed, is_currency=_index().is_currency)

    def activities(subgraphs):
        return {a["id"] for s in subgraphs for a in s["activities"]}

    assert "a2" in activities(open_subgraphs)
    assert activities(blocked_subgraphs) == {"a1"}
