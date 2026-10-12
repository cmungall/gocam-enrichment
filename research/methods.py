"""Prototype GO-CAM enrichment methods: flat ORA vs graph-aware alternatives.

All methods share one corpus view (`Corpus`): a gene-level "megagraph" in which
two genes are linked when an activity enabled by one is causally upstream of an
activity enabled by the other (any model), when one activity's chemical output is
another's input (same model), or when they co-enable one activity (complex).
Models stay as provenance: `members[m]` is the gene set of model m.

Methods
-------
ora           flat hypergeometric per model (what amigo3 does today)
conn          ORA hits re-tested for causal connectivity *within* the model,
              conditional on k (self-contained topology test), Fisher-combined with ORA
rwr           network-smoothed model score: RWR from the study genes on the
              megagraph, model score = mean propagated mass on its genes, empirical
              p from degree-matched seed permutations (competitive null)
module        seed-connected subnetwork (Steiner-style) on the megagraph; returns
              a cross-model module rather than ranking models
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.stats import combine_pvalues, hypergeom, norm


@dataclass
class Corpus:
    genes: list[str]
    index: dict[str, int]
    members: dict[str, set[str]]
    titles: dict[str, str]
    graph: nx.Graph  # gene-level megagraph (undirected, for diffusion)
    model_edges: dict[str, set[tuple[str, str]]]  # within-model gene-gene causal links
    kernel: np.ndarray | None = None
    degree_bins: list[np.ndarray] = field(default_factory=list)
    bin_of: np.ndarray | None = None


def load_corpus(path: str, *, taxon: str, sources=("gocam",), states=("production",), min_genes=1) -> Corpus:
    models = [
        m for m in json.load(open(path))
        if m["taxon"] == taxon and m["source"] in sources and m["state"] in states
    ]
    members, titles, model_edges = {}, {}, {}
    graph = nx.Graph()
    for m in models:
        acts = m["activities"]
        gs = {g for a in acts.values() for g in a["genes"]}
        if len(gs) < min_genes:
            continue
        members[m["id"]], titles[m["id"]] = gs, m["title"]
        links = set()
        for a in acts.values():  # complex co-enablers
            links.update(itertools.combinations(sorted(a["genes"]), 2))
        for s, o, _p, _sign in m["edges"]:
            if s in acts and o in acts:
                links.update((x, y) for x in acts[s]["genes"] for y in acts[o]["genes"] if x != y)
        producers = defaultdict(set)
        for a in acts.values():
            for c in a["outputs"]:
                producers[c].update(a["genes"])
        for a in acts.values():
            for c in a["inputs"]:
                links.update((x, y) for x in producers.get(c, ()) for y in a["genes"] if x != y)
        model_edges[m["id"]] = {tuple(sorted(e)) for e in links}
        graph.add_nodes_from(gs)
        graph.add_edges_from(model_edges[m["id"]])
    genes = sorted(graph.nodes)
    return Corpus(genes, {g: i for i, g in enumerate(genes)}, members, titles, graph, model_edges)


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def ora(c: Corpus, study: set[str], *, min_size=2) -> dict[str, float]:
    universe = set(c.genes)
    hits = study & universe
    N, n = len(universe), len(hits)
    p = {}
    for m, gs in c.members.items():
        if len(gs) < min_size:
            continue
        k = len(gs & hits)
        p[m] = hypergeom.sf(k - 1, N, len(gs), n) if k else 1.0
    return p


def connectivity_p(c: Corpus, m: str, hits: set[str], *, samples=500, rng=None) -> float:
    """P(edges among k random model genes >= observed), k = |hits in model|."""
    gs = sorted(c.members[m])
    h = [g for g in gs if g in hits]
    k = len(h)
    edges = c.model_edges[m]
    if k < 2 or not edges:
        return 1.0

    def stat(sub):
        s = set(sub)
        return sum(1 for a, b in edges if a in s and b in s)

    obs = stat(h)
    if obs == 0:
        return 1.0
    total = _choose(len(gs), k)
    if total <= samples:
        draws = list(itertools.combinations(gs, k))
    else:
        rng = rng or np.random.default_rng(0)
        draws = [rng.choice(gs, k, replace=False) for _ in range(samples)]
    ge = sum(1 for d in draws if stat(d) >= obs)
    return (ge + (0 if total <= samples else 1)) / (len(draws) + (0 if total <= samples else 1))


def _choose(n, k):
    from math import comb
    return comb(n, k)


def ora_conn(c: Corpus, study: set[str], *, min_size=2, rng=None) -> dict[str, float]:
    base = ora(c, study, min_size=min_size)
    hits = study & set(c.index)
    out = {}
    for m, p in base.items():
        if p < 0.05:
            pc = connectivity_p(c, m, hits, rng=rng)
            out[m] = combine_pvalues([p, pc], method="fisher")[1]
        else:
            out[m] = p
    return out


def prepare_rwr(c: Corpus, *, restart=0.5, nbins=10) -> None:
    n = len(c.genes)
    A = np.zeros((n, n))
    for a, b in c.graph.edges:
        i, j = c.index[a], c.index[b]
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(0)
    W = A / np.where(deg > 0, deg, 1.0)  # column-stochastic
    c.kernel = restart * np.linalg.inv(np.eye(n) - (1 - restart) * W)
    # degree bins for matched permutations
    qs = np.quantile(deg, np.linspace(0, 1, nbins + 1))
    c.bin_of = np.clip(np.searchsorted(qs, deg, side="right") - 1, 0, nbins - 1)
    c.degree_bins = [np.where(c.bin_of == b)[0] for b in range(nbins)]


def rwr(c: Corpus, study: set[str], *, perms=500, min_size=2, rng=None) -> dict[str, float]:
    rng = rng or np.random.default_rng(0)
    seeds = [c.index[g] for g in study if g in c.index]
    if not seeds:
        return {m: 1.0 for m, gs in c.members.items() if len(gs) >= min_size}
    ms = [m for m, gs in c.members.items() if len(gs) >= min_size]
    M = np.zeros((len(c.genes), len(ms)))
    for j, m in enumerate(ms):
        for g in c.members[m]:
            M[c.index[g], j] = 1.0 / len(c.members[m])
    S = np.zeros((len(c.genes), perms + 1))
    S[seeds, 0] = 1.0
    for p in range(1, perms + 1):
        for s in seeds:  # degree-matched gene-label permutation
            pool = c.degree_bins[c.bin_of[s]]
            S[rng.choice(pool), p] += 1.0
    scores = M.T @ (c.kernel @ S)  # models x (perms+1)
    obs, null = scores[:, 0], scores[:, 1:]
    # z-normalise against the degree-matched null (diffuStats-style); empirical
    # p-values floor at 1/(perms+1) and tie, which makes ranking meaningless.
    z = (obs - null.mean(1)) / np.maximum(null.std(1), 1e-12)
    return dict(zip(ms, norm.sf(z)))


def module(c: Corpus, study: set[str], *, max_hops=2) -> set[str]:
    """Largest seed-connected Steiner-style module: seeds joined via <=max_hops connectors."""
    seeds = [g for g in study if g in c.graph]
    if len(seeds) < 2:
        return set(seeds)
    H = nx.Graph()
    H.add_nodes_from(seeds)
    for s in seeds:
        lengths = nx.single_source_shortest_path(c.graph, s, cutoff=max_hops)
        for t in seeds:
            if t != s and t in lengths:
                nx.add_path(H, lengths[t])
    comps = sorted(nx.connected_components(H), key=lambda x: len(set(x) & set(seeds)), reverse=True)
    best = comps[0]
    # keep connectors only if they bridge >=2 seeds (prune dangling)
    return set(best)
