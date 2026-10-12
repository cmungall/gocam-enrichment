"""Simulation benchmark: flat ORA vs graph-aware GO-CAM enrichment.

Usage: python benchmark.py CORPUS.json OUT_DIR [--taxon NCBITaxon:9606] [--trials 150]

Signal types (planted in a source model with >=6 genes and >=4 internal links):
  scatter  k random genes of the model (no topology: graph methods should not lose)
  chain    k genes forming a connected causal walk inside the model
  cross    k genes forming a connected walk on the megagraph starting in the model
           (may leave it: models are small and stitched by shared genes)
Each study set = signal + `noise` genes drawn uniformly from the GO-CAM gene universe.
Null sets (noise only) measure the false-positive rate at BH-FDR < 0.05.

Caveat: planting signal along graph edges favours graph methods by construction;
`scatter` and the null rows are the no-harm / calibration checks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from methods import bh, load_corpus, module, ora, ora_conn, prepare_rwr, rwr


def walk(G: nx.Graph, start: str, k: int, rng, allowed: set[str] | None = None) -> list[str] | None:
    chosen = [start]
    frontier = set(G[start])
    while len(chosen) < k:
        cand = sorted(n for n in frontier if n not in chosen and (allowed is None or n in allowed))
        if not cand:
            return None
        nxt = cand[rng.integers(len(cand))]
        chosen.append(nxt)
        frontier |= set(G[nxt])
    return chosen


def rank_of(p: dict[str, float], target: str) -> float:
    v = p[target]
    vals = np.array(list(p.values()))
    better, ties = (vals < v - 1e-15).sum(), (np.abs(vals - v) <= 1e-15).sum()
    return better + (ties + 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--taxon", default="NCBITaxon:9606")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--perms", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    c = load_corpus(args.corpus, taxon=args.taxon)
    t0 = time.time()
    prepare_rwr(c)
    print(f"{len(c.members)} models, {len(c.genes)} genes, {c.graph.number_of_edges()} links; kernel {time.time()-t0:.1f}s")
    comps = sorted((len(x) for x in nx.connected_components(c.graph)), reverse=True)
    print("megagraph components:", len(comps), "largest:", comps[:5])

    sources = []
    for m, gs in c.members.items():
        sub = nx.Graph(list(c.model_edges[m]))
        if len(gs) >= 6 and sub.number_of_edges() >= 4:
            sources.append(m)
    print("eligible source models:", len(sources))
    universe = np.array(c.genes)
    methods = {"ora": lambda s: ora(c, s), "ora+conn": lambda s: ora_conn(c, s, rng=rng),
               "rwr": lambda s: rwr(c, s, perms=args.perms, rng=rng)}
    rows = []
    for signal in ("scatter", "chain", "cross"):
        for k in (3, 5):
            for noise in (20, 100):
                for _ in range(args.trials):
                    m = sources[rng.integers(len(sources))]
                    gs = sorted(c.members[m])
                    if signal == "scatter":
                        sig = list(rng.choice(gs, min(k, len(gs)), replace=False))
                    else:
                        allowed = set(gs) if signal == "chain" else None
                        G = nx.Graph(list(c.model_edges[m])) if signal == "chain" else c.graph
                        starts = [g for g in gs if g in G]
                        sig = walk(G, starts[rng.integers(len(starts))], k, rng, allowed)
                        if sig is None:
                            continue
                    study = set(sig) | set(rng.choice(universe, noise, replace=False))
                    row = {"signal": signal, "k": k, "noise": noise, "model": m,
                           "frac_signal_in_model": len(set(sig) & c.members[m]) / k}
                    for name, f in methods.items():
                        p = f(study)
                        q = dict(zip(p, bh(np.array(list(p.values())))))
                        row[f"{name}_rank"] = rank_of(p, m)
                        row[f"{name}_sig"] = q[m] < 0.05
                        top = min(p, key=p.get)
                        row[f"{name}_top_signal_recall"] = len(c.members[top] & set(sig)) / k
                    mod = module(c, study)
                    row["module_signal_recall"] = len(mod & set(sig)) / k
                    row["module_precision"] = len(mod & set(sig)) / max(1, len(mod & study))
                    rows.append(row)
                print(signal, k, noise, "done", flush=True)
    for size in (25, 100):
        for _ in range(args.trials):
            study = set(rng.choice(universe, size, replace=False))
            row = {"signal": "null", "k": 0, "noise": size}
            for name, f in methods.items():
                p = f(study)
                row[f"{name}_any_fdr05"] = bool((bh(np.array(list(p.values()))) < 0.05).any())
            rows.append(row)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(args.out) / "rows.json", "w"), default=lambda o: o.item())
    summarize(rows, Path(args.out) / "summary.md")


def summarize(rows, path):
    import pandas as pd
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c.endswith("_sig") or c.endswith("_fdr05")]:
        df[col] = df[col].astype(float)
    sig = df[df.signal != "null"]
    out = []
    agg = {}
    for name in ("ora", "ora+conn", "rwr"):
        agg[f"{name} R@1"] = (f"{name}_rank", lambda r: (r <= 1).mean())
        agg[f"{name} R@5"] = (f"{name}_rank", lambda r: (r <= 5).mean())
        agg[f"{name} FDR<.05"] = (f"{name}_sig", "mean")
    agg["ora top-model signal recall"] = ("ora_top_signal_recall", "mean")
    agg["rwr top-model signal recall"] = ("rwr_top_signal_recall", "mean")
    agg["module signal recall"] = ("module_signal_recall", "mean")
    agg["module precision"] = ("module_precision", "mean")
    agg["signal in source model"] = ("frac_signal_in_model", "mean")
    agg["n"] = ("model", "size")
    t = sig.groupby(["signal", "k", "noise"]).agg(**agg).round(2)
    out.append("## Recovery of the planted source model\n\n" + t.T.to_markdown())
    null = df[df.signal == "null"]
    nt = null.groupby("noise")[[c for c in null.columns if c.endswith("_any_fdr05")]].mean().round(3)
    out.append("## Null study sets: fraction with any model at BH-FDR < 0.05\n\n" + nt.to_markdown())
    Path(path).write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
