# GO-CAM Enrichment

Command line prototype for GO-CAM-grounded network retrieval from mixed omics
object lists. It accepts genes, proteins, chemicals, complexes, GO terms, or
activity IDs, diffuses signal over a heterogeneous GO-CAM graph, and returns
ranked grounded subgraphs for downstream interpretation.

This is intentionally not a fixed-model GO-CAM enrichment tool. GO-CAM model IDs
are kept as provenance while signal can cross model boundaries through shared
biological objects.

## Install

From this repo:

```bash
uv sync
```

The local `../gocam` package is used as an editable dependency through
`tool.uv.sources`.

## Input

Use TSV, CSV, YAML, or a plain one-object-per-line file. A TSV with magnitudes
can look like this:

```tsv
id	type	modality	value	label
UniProtKB:P31749	protein	phosphoproteomics	2.1	AKT1
CHEBI:17234	chemical	metabolomics	-1.4	glucose
HGNC:3236	gene	transcriptomics	1.8	EGFR
```

Only `id` is required. `value` is optional; missing values are treated as 1.0.

YAML input sets can either inline objects under `inputs` or point to a TSV/CSV
with `inputs_path`. The gallery uses the latter so each example can carry richer
metadata:

```yaml
id: example_input_set
inputs_path: inputs.tsv
provenance:
  source_name: GO-CAM development corpus
expected_answer:
  should_surface:
    - id: GO:0032958
      label: inositol phosphate biosynthetic process
```

## Commands

Fetch GO-CAM YAML from the GO API:

```bash
uv run gocam-enrich fetch \
  --model-id 663d668500002178 \
  --out .cache/gocam/models.yaml
```

Build a reusable graph from local YAML:

```bash
uv run gocam-enrich build \
  --models ../gocam/data/gocam.yaml \
  --out .cache/gocam/graph.json
```

Run retrieval from a graph:

```bash
uv run gocam-enrich run inputs.tsv \
  --graph .cache/gocam/graph.json \
  --out results/
```

Run directly from YAML:

```bash
uv run gocam-enrich run inputs.tsv \
  --models ../gocam/data/gocam.yaml \
  --taxon NCBITaxon:9606 \
  --out results/
```

Run by fetching models from the API first:

```bash
uv run gocam-enrich run inputs.tsv \
  --source api \
  --model-id 663d668500002178 \
  --out results/
```

## Metabolomics Inputs

GO-CAM models name chemicals by ChEBI, usually the pH 7.3 major microspecies
(citrate(3-), not citric acid), while metabolomics platforms report neutral
forms or HMDB/KEGG ids. Exact matching misses most of them. Point
`--chemical-normalization` at the output of `genesets-workflows
prepare-chem2go` (monarch-initiative/genesets) to match inputs and graph
chemical nodes by canonical ChEBI id instead:

```bash
uv run gocam-enrich run metabolites.tsv \
  --graph .cache/gocam/graph.json \
  --chemical-normalization ../genesets/evals/chem2go/generated/current \
  --out results/
```

`HMDB0000094`, `CHEBI:30769` (citric acid) and `CHEBI:16947` (citrate(3-))
then all seed the same node; the `match` field in the outputs records
`exact`, `activity` or `chemical_normalized`.

Chemicals shared by many activities are shortcuts through the graph, so the
walk is damped around them:

- edges into a chemical node are scaled by `degree ** -alpha`
  (`--chemical-hub-alpha`, default 0.5; 0 disables);
- currency metabolites (water, H+, ATP/ADP, NAD(P)(H), CoA, SAM, common ions)
  receive no walk mass unless they are themselves inputs, so they cannot carry
  signal between unrelated activities (`--keep-currency` disables). The list
  comes from the prep directory's `currency.tsv`, or a built-in default.

## Outputs

`run` writes:

- `report.md`: human-readable run report with summary, provenance when
  available, expected-answer anchors when provided, ranked subgraphs, matched
  inputs, static visualization links, and leading grounded activities.
- `ranked_subgraphs.tsv`: one ranked row per returned subgraph.
- `subgraph_001.dot`: Graphviz source for an activity-centric static diagram
  showing the leading activities by score.
- `subgraph_001.svg`: static diagram rendered when Graphviz `dot` is installed.
- `subgraph_001.png`: static diagram rendered when Graphviz `dot` is installed.
- `subgraph_001.json`: grounded nodes, edges, activities, inputs, and paths.
- `subgraph_001.graphml`: graph export for Cytoscape/NetworkX tooling.
- `grounding.tsv`: activity-level grounding across all returned subgraphs.
- `unmapped_inputs.tsv`: input objects not found in the graph.

The TSV and JSON outputs include IDs plus resolved labels where GO-CAM provides
them, including activity/gene-product, molecular-function, biological-process,
cellular-component, matched input, and top-term labels.

## Output Model

The public output structure is defined in `src/gocam_enrichment/results.py`:

- `ResultBundle`
- `SubgraphResult`
- `ActivityGrounding`
- `TermSummary`
- `NodeResult`
- `EdgeResult`
- `InputResult`
- `PathResult`

Writers live in `src/gocam_enrichment/output_adapters.py`. Current adapters
write a markdown report, ranked TSV, grounding TSV, unmapped-input TSV,
activity-centric Graphviz DOT/SVG/PNG diagrams, per-subgraph JSON, and
per-subgraph GraphML. Future visual outputs should add an `OutputAdapter`
implementation rather than changing diffusion logic.

The JSON output is designed to be interpreted by an agent, but all claims can be
traced to GO-CAM activity IDs, source model IDs, RO predicates, and evidence
objects when present.
