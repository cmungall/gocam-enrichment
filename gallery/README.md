# Input Set Gallery

This directory is a small catalog of example query inputs for `gocam-enrich`.
It is organized like a subrepo so it can later be split out, copied into
benchmarks, or rendered as a visual gallery.

Each set has its own folder:

```text
input_sets/<category>/<set-id>/
  metadata.yaml
  inputs.tsv
```

`metadata.yaml` records provenance, intended use, expected answer anchors, and a
runnable command. `inputs.tsv` keeps the simple object-list format understood by
the CLI:

```tsv
id	type	modality	value	label
```

The CLI can consume either file. Passing `metadata.yaml` works because the
metadata points at `inputs.tsv` with `inputs_path`.

Run examples against a local megagraph:

```bash
uv run gocam-enrich build \
  --models ../gocam/data/gocam.yaml \
  --out .cache/gocam/full-megagraph.json

uv run gocam-enrich run gallery/input_sets/mixed_omics/pombe_phosphoinositide_mixed/metadata.yaml \
  --graph .cache/gocam/full-megagraph.json \
  --taxon NCBITaxon:4896 \
  --out results/pombe_phosphoinositide
```

The catalog index is in `input_sets/manifest.yaml`; detailed metadata lives with
each input set.
