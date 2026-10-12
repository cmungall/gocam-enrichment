"""Flatten Noctua/Minerva JSON GO-CAMs into a compact activity-graph corpus.

Usage: python parse_minerva.py RAW_DIR OUT.json

Keeps curated GO-CAMs (16-hex model ids) and Reactome-converted models (R-HSA-*),
in any state; `state` is recorded so callers can filter to production.
"""

import json
import re
import sys
from pathlib import Path

CAUSAL = {
    "RO:0002629": +1, "RO:0002304": +1, "RO:0002213": +1, "RO:0002407": +1,
    "RO:0002630": -1, "RO:0002305": -1, "RO:0002212": -1, "RO:0002409": -1,
    "RO:0002413": +1,  # provides input for / directly provides input for
    "RO:0002411": 0, "RO:0002418": 0, "RO:0002211": 0, "RO:0012009": 0,
    "RO:0002578": 0, "RO:0002406": +1, "RO:0002408": -1,
}
ENABLED_BY, HAS_PART = "RO:0002333", "BFO:0000051"
PART_OF, OCCURS_IN = "BFO:0000050", "BFO:0000066"
HAS_INPUT, HAS_OUTPUT = {"RO:0002233", "RO:0004009"}, {"RO:0002234", "RO:0004008"}
NON_GENE = re.compile(r"^(GO|CHEBI|ECO|UBERON|CL|NCBITaxon|PO|ZFA|WBbt|EMAPA|BFO|RO|SO|PR|ComplexPortal|FBbt|CARO|BTO|MI)[:_]")


def root(ind):
    return {t.get("id") for t in ind.get("root-type", []) if t.get("type") == "class"}


def first_type(ind):
    for t in ind.get("type", []):
        if t.get("type") == "class":
            return t.get("id"), t.get("label")
    return None, None


def parse(path):
    d = json.load(open(path))
    ann = {}
    for a in d.get("annotations", []):
        ann.setdefault(a["key"], []).append(a["value"])
    ind = {i["id"]: i for i in d["individuals"]}
    facts = d["facts"]
    by_subj = {}
    for f in facts:
        by_subj.setdefault(f["subject"], []).append(f)

    def is_mf(i):
        return "GO:0003674" in root(ind[i]) if i in ind else False

    def genes_of(obj, depth=0):
        tid, lab = first_type(ind[obj])
        if tid and not NON_GENE.match(tid):
            return [(tid, lab)]
        out = []
        if depth < 2:
            for f in by_subj.get(obj, []):
                if f["property"] == HAS_PART and f["object"] in ind:
                    out += genes_of(f["object"], depth + 1)
        return out

    acts = {}
    for i in ind:
        if not is_mf(i):
            continue
        mf, mfl = first_type(ind[i])
        a = {"mf": mf, "mf_label": mfl, "genes": [], "bp": None, "cc": None, "inputs": [], "outputs": [], "complex": None}
        for f in by_subj.get(i, []):
            o = f["object"]
            if o not in ind:
                continue
            p = f["property"]
            if p == ENABLED_BY:
                tid, _ = first_type(ind[o])
                if tid and NON_GENE.match(tid):
                    a["complex"] = tid
                a["genes"] += [g for g, _ in genes_of(o)]
            elif p == PART_OF:
                a["bp"] = first_type(ind[o])[0]
            elif p == OCCURS_IN:
                a["cc"] = first_type(ind[o])[0]
            elif p in HAS_INPUT:
                t = first_type(ind[o])[0]
                if t and t.startswith("CHEBI:"):
                    a["inputs"].append(t)
            elif p in HAS_OUTPUT:
                t = first_type(ind[o])[0]
                if t and t.startswith("CHEBI:"):
                    a["outputs"].append(t)
        a["genes"] = sorted(set(a["genes"]))
        acts[i.split("/")[-1]] = a
    edges = []
    for f in facts:
        if f["property"] in CAUSAL and is_mf(f["subject"]) and is_mf(f["object"]):
            edges.append([f["subject"].split("/")[-1], f["object"].split("/")[-1], f["property"], CAUSAL[f["property"]]])
    taxon = (ann.get("https://w3id.org/biolink/vocab/in_taxon") or [None])[0]
    return {
        "id": d["id"].replace("gomodel:", ""),
        "title": (ann.get("title") or [""])[0].strip(),
        "state": (ann.get("state") or [""])[0],
        "taxon": taxon,
        "source": "reactome" if "R-HSA" in path.name else "gocam",
        "activities": acts,
        "edges": edges,
    }


def main(raw, out):
    models = []
    for p in sorted(Path(raw).glob("*.json")):
        if re.fullmatch(r"[0-9a-f]{16}", p.stem) or p.stem.startswith("R-HSA-"):
            m = parse(p)
            if m["activities"]:
                models.append(m)
    json.dump(models, open(out, "w"))
    print(len(models), "models")


if __name__ == "__main__":
    main(*sys.argv[1:])
