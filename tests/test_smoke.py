from __future__ import annotations

import json
import shutil

from typer.testing import CliRunner

from gocam_enrichment.cli import app
from gocam_enrichment.model_io import parse_inputs

runner = CliRunner()


MINI_MODEL = """---
id: gomodel:test
title: Test signaling model
taxon: NCBITaxon:9606
status: production
activities:
- id: gomodel:test/a1
  enabled_by:
    type: EnabledByGeneProductAssociation
    term: UniProtKB:P1
  molecular_function:
    type: MolecularFunctionAssociation
    term: GO:0004672
  occurs_in:
    type: CellularAnatomicalEntityAssociation
    term: GO:0005829
  part_of:
    type: BiologicalProcessAssociation
    term: GO:0007165
  has_input:
  - type: MoleculeAssociation
    term: CHEBI:1
  causal_associations:
  - type: CausalAssociation
    predicate: RO:0002629
    downstream_activity: gomodel:test/a2
- id: gomodel:test/a2
  enabled_by:
    type: EnabledByGeneProductAssociation
    term: UniProtKB:P2
  molecular_function:
    type: MolecularFunctionAssociation
    term: GO:0000166
  occurs_in:
    type: CellularAnatomicalEntityAssociation
    term: GO:0005829
  part_of:
    type: BiologicalProcessAssociation
    term: GO:0007165
objects:
- id: UniProtKB:P1
  label: protein one
  type: gocam:Object
- id: UniProtKB:P2
  label: protein two
  type: gocam:Object
- id: CHEBI:1
  label: molecule one
  type: gocam:Object
- id: GO:0007165
  label: signal transduction
  type: gocam:Object
"""


def test_build_and_run(tmp_path):
    models = tmp_path / "models.yaml"
    models.write_text(MINI_MODEL)
    graph = tmp_path / "graph.json"
    result = runner.invoke(app, ["build", "--models", str(models), "--out", str(graph)])
    assert result.exit_code == 0, result.output
    assert graph.exists()

    inputs = tmp_path / "inputs.tsv"
    inputs.write_text("id\ttype\tmodality\tvalue\nUniProtKB:P1\tprotein\tproteomics\t2.0\nCHEBI:1\tchemical\tmetabolomics\t-1.0\n")
    metadata = tmp_path / "input_set.yaml"
    metadata.write_text(
        "id: test_input_set\n"
        "title: Test input set\n"
        "category: mixed_omics\n"
        "inputs_path: inputs.tsv\n"
        "provenance:\n"
        "  source_name: smoke test\n"
        "expected_answer:\n"
        "  intent: Recover signal transduction.\n"
        "  should_surface:\n"
        "    - id: GO:0007165\n"
        "      label: signal transduction\n"
        "      role: process anchor\n"
    )

    out = tmp_path / "results"
    result = runner.invoke(app, ["run", str(metadata), "--graph", str(graph), "--out", str(out), "--top", "3"])
    assert result.exit_code == 0, result.output
    ranked = out / "ranked_subgraphs.tsv"
    subgraph = out / "subgraph_001.json"
    dot = out / "subgraph_001.dot"
    report = out / "report.md"
    assert ranked.exists()
    assert subgraph.exists()
    assert dot.exists()
    assert report.exists()
    if shutil.which("dot") is not None:
        assert (out / "subgraph_001.svg").exists()
        assert (out / "subgraph_001.png").exists()
    data = json.loads(subgraph.read_text())
    assert data["activity_count"] >= 1
    assert data["input_hit_count"] >= 1
    assert data["activities"][0]["enabled_by_label"] in {"protein one", "protein two"}
    assert "enabled_by_label" in (out / "grounding.tsv").read_text().splitlines()[0]
    assert "top_term_labels" in ranked.read_text().splitlines()[0]
    report_text = report.read_text()
    assert "# Test input set" in report_text
    assert "signal transduction" in report_text
    assert "## Visualizations" in report_text
    assert "subgraph_001.dot" in report_text


def test_parse_inline_yaml_inputs_preserves_zero(tmp_path):
    inputs = tmp_path / "inputs.yaml"
    inputs.write_text(
        "id: inline_set\n"
        "inputs:\n"
        "  - id: CHEBI:1\n"
        "    type: chemical\n"
        "    modality: metabolomics\n"
        "    value: 0\n"
        "    magnitude: 5\n"
    )

    parsed = parse_inputs(inputs)
    assert parsed[0].id == "CHEBI:1"
    assert parsed[0].value == 0
