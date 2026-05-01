# Input Sets

These directories are intended to exercise different parts of the GO-CAM
grounded retrieval graph:

- `gene_sets/`: gene/protein-only queries.
- `metabolite_sets/`: chemical-only queries.
- `go_term_sets/`: ontology-term-only semantic queries.
- `mixed_omics/`: inputs that combine genes/proteins, chemicals, and GO terms
  with signed magnitudes.

Each input set is represented as:

```text
<category>/<set-id>/
  metadata.yaml
  inputs.tsv
```

The `metadata.yaml` file is the richer object. It includes:

- `provenance`: where the set came from and how it was selected.
- `context`: taxon and graph assumptions.
- `expected_answer`: semantic anchors that a good retrieval should surface.
- `recommended_command`: a ready-to-run CLI invocation.
- `inputs_path`: the input file to parse.

The lightweight schema note is in `metadata_schema.yaml`.

The current examples use identifiers observed in the local GO-CAM corpus used
during development. They are not gold-standard biological benchmarks; they are
practical fixtures for exploring behavior and for designing future benchmarks.

The CLI also supports a fully inline YAML input set:

```yaml
id: example_inline_set
inputs:
  - id: UniProtKB:P31749
    type: protein
    modality: phosphoproteomics
    value: 2.1
    label: AKT1
  - id: CHEBI:17268
    type: chemical
    modality: metabolomics
    value: -1.4
    label: myo-inositol
```
