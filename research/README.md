# Graph-aware GO-CAM enrichment: investigation

Today, GO-CAM enrichment in AmiGO 3 (`amigo3/src/amigo3/enrichment.py`, `target="gocam"`)
treats each model as a flat gene set and runs a one-sided hypergeometric test.
This directory asks what graph-based alternatives buy us on the real corpus. It
contains:

- `parse_minerva.py` flattens the GO release's `noctua-models-json.tgz` into an activity-graph corpus.
- `methods.py` implements four prototype methods.
- `benchmark.py` is a simulation benchmark that plants signal in human models.
- `results/` holds the benchmark output (150 trials per condition).

Reproduce:

```bash
curl -sSLO http://current.geneontology.org/products/json/noctua-models-json.tgz
mkdir raw && tar xzf noctua-models-json.tgz -C raw
python parse_minerva.py raw corpus.json          # 9,061 GO-CAM + Reactome-converted models
python benchmark.py corpus.json out/ --trials 150 --perms 500   # needs networkx numpy scipy pandas tabulate
```

## 1. What the corpus looks like (GO release, Oct 2026, production models)

| Taxon | Models | Distinct genes | Genes/model (median) | Models with <3 genes | Causal edges/model (median) | Activities isolated (no causal edge) |
|---|---|---|---|---|---|---|
| Human | 1,525 | 2,330 | 3 | 27% | 2 | 12% |
| Mouse | 1,543 | 2,777 | 2 | 64% | 0 | 77% |
| Zebrafish | 1,996 | 2,229 | 1 | 92% | 0 | 96% |
| S. cerevisiae | 644 | 845 | 1 | 93% | 0 | 95% |
| S. pombe | 142 | 1,657 | 11 | 1% | 5.5 | 31% |
| Human (Reactome-converted) | 280 | 1,580 | 6 | 20% | 0 | 72% |

Consequences for method design:

- **Models are tiny.** Outside human and pombe, most "models" have one or two
  genes, so per-model ORA has almost no power. A graph view helps only where
  causal edges exist, which in practice means human, pombe, and partly mouse.
- **Models are highly redundant.** Of the 1,331 human models with at least 2
  genes, 648 have a near-twin (gene-set Jaccard ≥ 0.5), and 97 have an exact
  gene-set twin. The largest cluster has 95 GPCR signalling models that share
  the same G-protein and adenylyl-cyclase genes (e.g. UniProtKB:O43306 is in
  123 models). A study set containing a few G-protein genes lights up dozens of
  models with near-identical p-values. The genes that would discriminate between
  them, the receptor and ligand, sit at a specific position in the graph.
- **Models stitch into a megagraph.** Link genes by causal edges, chemical
  output→input, and complex co-membership, and let shared gene products join
  models. The 2,330 human genes then form one giant component of 1,681 genes,
  plus 352 small components. Real biology crosses model boundaries.
- **Human edges are signed.** About 100% of human causal edges carry a sign
  (RO:0002629/0002630 dominate). That makes sign-concordance methods viable
  when the input has a direction of change.

## 2. Methods prototyped (`methods.py`)

| Method | Unit | Null | Question it answers |
|---|---|---|---|
| `ora` | model gene set | hypergeometric (competitive) | Does the list over-represent the model's genes? (current AmiGO behaviour) |
| `ora+conn` | model | ORA × within-model connectivity test, conditional on k hits (random k-subsets of the model's genes); combined with Fisher | Are the hits causally adjacent inside the model, beyond what membership alone predicts? |
| `rwr` | model, scored on the megagraph | random walk with restart (restart 0.5) from the study genes; model score = mean propagated mass; z-scored against 500 degree-matched seed permutations | Is the model *near* the signal, even if it was not hit directly? |
| `module` | connected subgraph of the megagraph | none yet | Which connected cross-model subnetwork joins the hits (seeds joined by connectors ≤ 2 hops)? |

## 3. Simulation benchmark (human, `results/human_benchmark.md`)

The benchmark plants k ∈ {3, 5} signal genes in a source model (206 eligible
models: at least 6 genes and at least 4 internal links), then adds 20 or 100
noise genes drawn from the GO-CAM gene universe. There are three signal shapes:

- **scatter:** random genes of the model (no topology).
- **chain:** a connected causal walk inside the model.
- **cross:** a connected walk on the megagraph that starts in the model and
  may leave it. On average only 54–70% of the signal stays in the source
  model.

Selected numbers (R@1 / R@5 = the source model ranked first / in the top 5):

| Condition | ora R@1 | ora+conn R@1 | rwr R@1 | ora R@5 | ora+conn R@5 | rwr R@5 | module signal recall / precision |
|---|---|---|---|---|---|---|---|
| scatter k=5, noise 20 | 0.84 | 0.85 | 0.13 | 0.99 | 0.99 | 0.81 | 0.95 / 0.84 |
| scatter k=3, noise 100 | 0.41 | 0.45 | 0.18 | 0.87 | 0.81 | 0.55 | 0.36 / 0.08 |
| chain k=3, noise 20 | 0.63 | 0.65 | 0.10 | 0.94 | 0.96 | 0.80 | 0.99 / 0.82 |
| chain k=3, noise 100 | 0.30 | **0.45** | 0.11 | 0.78 | **0.87** | 0.38 | 0.50 / 0.11 |
| cross k=3, noise 20 | 0.30 | 0.31 | 0.07 | 0.59 | 0.61 | 0.42 | **0.95 / 0.83** |
| cross k=5, noise 20 | 0.25 | 0.27 | 0.07 | 0.52 | 0.51 | 0.46 | **1.00 / 0.89** |
| cross k=5, noise 100 | 0.22 | 0.25 | 0.14 | 0.47 | 0.52 | 0.42 | 0.70 / 0.27 |

Calibration on pure-noise study sets (fraction of sets with any model at BH-FDR < 0.05):

| Size | ora | ora+conn | rwr (z-normal) |
|---|---|---|---|
| 25 | 0.013 | 0 | **1.00** |
| 100 | 0.013 | 0 | **1.00** |

**Caveat.** Planting signal along graph edges favours graph methods by
construction. The `scatter` rows and the null rows are the no-harm and
calibration checks. These results test the mechanics, not biological validity
(see §5).

### Findings

1. **Flat ORA is a strong baseline when the signal sits inside one model.**
   Graph methods do not beat it for ranking a single model in the easy
   settings.
2. **The within-model connectivity test is a cheap, safe improvement.**
   - It helps where it should: a causal chain buried in heavy noise (chain
     k=3, noise 100: R@1 0.30 → 0.45, R@5 0.78 → 0.87).
   - It costs little when the hits are scattered (R@5 0.87 → 0.81 in the
     worst case).
   - It is conservative under the null.
   - Combining with Fisher weakens FDR-level power when connectivity
     contributes nothing. It is better reported as a separate column ("hits
     form a connected causal chain, p = …") than folded into one p-value.
3. **Cross-model signal is where model-level ORA breaks.**
   - Once signal spans models, the best single model ranks first only
     12–30% of the time and holds only 36–83% of the signal.
   - The `module` approach recovers 95–100% of the signal at 83–89% precision
     at low noise.
   - At 100 noise genes, connectors through hub genes swamp it (precision
     0.11–0.27). This is the hub artefact DOMINO (Levi 2021) warns about, so
     module extraction needs a permutation null and hub control before it can
     report significance.
4. **Naive diffusion is the weakest and the least calibrated option.**
   - RWR spreads mass into the overlapping twin models, so the source model
     rarely ranks first (R@1 ≈ 0.1).
   - z-normalising against degree-matched permutations, as diffuStats does
     with a Gaussian approximation, produced FDR hits on **every** pure-noise
     set: the null score distributions are heavily skewed.
   - The existing `gocam_enrichment.diffusion.diffuse()` has no null at all.
     Treat its scores as a layout and prioritisation signal, not as evidence
     of enrichment, until it has empirical p-values or permutation-based
     calibration.

## 4. Recommendations, in order of value per effort

1. **Fix the unit and the redundancy in the existing ORA** (AmiGO 3; gene list
   only):
   - Collapse near-duplicate models, or decorrelate them with a topGO-style
     elim/weight pass (Alexa 2006), so one G-protein hit does not yield 95
     models.
   - Count *activities* (steps) rather than genes, which handles paralog "OR"
     sets (cf. Hill 2023, Genetics; nmarkari/gocam_enrichment).
   - Show models with fewer than 3 genes separately; they cannot reach
     significance.
2. **Add the connectivity annotation** (`ora+conn` logic) as a second column:
   "k of n hits form a connected causal chain (p = …)". It is cheap, exact for
   small models, and calibrated.
3. **Megamodel subnetwork retrieval for cross-model signal.** This is the
   direction `gocam-enrichment` already takes. It needs:
   - a permutation null on module score or size (gene-label permutation that
     keeps the graph fixed, plus degree-preserving rewiring);
   - hub exclusion or degree penalties (cf. the existing
     `max_bridge_degree`);
   - DOMINO-style validation.
   Report models only as provenance for the returned module.
4. **Sign concordance when the input has direction** (fold-changes,
   phospho-changes):
   - Use IPA upstream-regulator / reverse causal reasoning / CausalR-style
     concordance over the signed causal edges.
   - Null: within-model edge-sign shuffling plus gene-label permutation.
   - This is the closest fit to GO-CAM semantics and is not available from
     flat sets at all. Not benchmarked here because it needs signed input data.
5. **Chemicals as a first-class layer.** 11% of human activities (44% of
   pombe, 90% of Reactome-converted) carry ChEBI inputs or outputs. Score each
   layer and combine with Stouffer (IMPaLA, multiGSEA), or propagate across
   output→input edges. Exclude currency metabolites, as `/megamodels` already
   does.
6. **Real-data evaluation before choosing.**
   - Simulation cannot decide between these methods.
   - Use perturbation or knockout datasets with a known target pathway
     (Nguyen 2019 design), plus shuffled-label null runs for every method.
   - The cGSA benchmark in `monarch-initiative/genesets/evals/cgsa` (DEG lists
     with ground-truth pathway names) is the nearest local candidate. It would
     need its pathway names mapped to GO-CAM models.

## 5. Literature map (abridged; DOIs verified against Crossref)

- **Topology-based pathway analysis**
  - Methods: SPIA (Tarca 2009, 10.1093/bioinformatics/btn577), CePa (Gu 2012, 10.1186/1752-0509-6-56), NetGSA (Shojaie 2009; Ma 2016, 10.1093/bioinformatics/btw410), DEGraph (Jacob 2012), TopologyGSA (Massa 2010), PADOG (Tarca 2012, overlap down-weighting).
  - Benchmarks: Ihnatova 2018 (10.1371/journal.pone.0191154), Nguyen 2019 Genome Biol (10.1186/s13059-019-1790-4), Bayerlová 2015. Gains over ORA are modest and depend on the dataset, null calibration matters as much as method choice, and pathway overlap causes false positives.
- **Propagation**
  - Review: Cowen 2017 Nat Rev Genet (10.1038/nrg.2017.38).
  - Methods: HotNet2 (Leiserson 2015), diffuStats (Picart-Armada 2017, 10.1093/bioinformatics/btx632), EnrichNet (Glaab 2012), NetPEA (Liu 2017).
  - Hub and normalisation issues: Biran 2019; Huang 2018 Cell Syst.
- **Active modules**
  - jActiveModules (Ideker 2002), BioNet (Dittrich 2008).
  - DOMINO (Levi 2021, 10.15252/msb.20209593): modules from permuted data are often enriched for the same terms as modules from real data.
- **Causal reasoning**
  - IPA upstream analysis (Krämer 2014, 10.1093/bioinformatics/btt703), RCR (Catlett 2013), CausalR (Bradley 2017, 10.1093/bioinformatics/btx425).
  - CARNIVAL (Liu 2019) and COSMOS (Dugourd 2021) for multi-omics, CausalPath (Babur 2021).
- **Path, edge and sub-pathway enrichment:** SubpathwayMiner (Li 2009), MinePath (Koumakis 2016), ESEA (Han 2015).
- **GO-CAM-specific**
  - Thomas 2019 Nat Genet (10.1038/s41588-019-0500-1) introduces GO-CAM but no enrichment method.
  - Hill 2023 Genetics (10.1093/genetics/iyad152) defines position-based subsets of mouse pathway models.
  - The unreviewed nmarkari/gocam_enrichment (step-centric) also exists.
  - No peer-reviewed topology-aware GO-CAM enrichment method was found; this appears to be an open gap.
- **Multi-omics:** IMPaLA (Kamburov 2011), multiGSEA (Canzler 2020), PaintOmics 4 (Liu 2022), MetaboAnalyst 5.0 (Pang 2021).
