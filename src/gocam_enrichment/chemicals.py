"""Chemical identifier normalization for matching metabolomics inputs to GO-CAM.

GO-CAM models name chemicals by ChEBI, usually the major microspecies at
pH 7.3 (citrate(3-), not citric acid), while metabolomics platforms report
neutral forms or HMDB/KEGG/LIPID MAPS ids. Exact id matching therefore misses
most metabolites.

``ChemicalIndex.load`` reads the directory written by
``genesets-workflows prepare-chem2go`` (monarch-initiative/genesets):

- ``normalization.tsv``: ``chebi_id``, ``canonical_id``, ``method``
- ``xrefs.tsv``: ``xref``, ``chebi_id`` (HMDB, KEGG, LIPID MAPS, CAS, ...)
- ``currency.tsv``: ``chemical_id`` of currency metabolites

Inputs and graph chemical nodes are both mapped to canonical ids, so any two
ids in the same conjugate acid/base / tautomer group match.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

CHEBI = re.compile(r"^(?:https?://purl\.obolibrary\.org/obo/)?CHEBI[:_](\d+)$", re.IGNORECASE)
HMDB = re.compile(r"^(?:HMDB:)?HMDB(\d{5,7})$", re.IGNORECASE)
KEGG_COMPOUND = re.compile(r"^(?:KEGG(?:\.COMPOUND)?:|CPD:)?(C\d{5})$", re.IGNORECASE)
LIPIDMAPS = re.compile(r"^(?:LIPIDMAPS:)?(LM[A-Z]{2}[0-9A-Z]{8,10})$", re.IGNORECASE)
CAS = re.compile(r"^(?:CAS:)?(\d{2,7}-\d{2}-\d)$", re.IGNORECASE)

#: Currency metabolites (pH 7.3 ChEBI ids) used when no prep directory is
#: given: water, H+, O2, CO2, bicarbonate, H2O2, NH4+, phosphate,
#: diphosphate, ATP/ADP/AMP, GTP/GDP, UTP/UDP, CTP/CDP, NAD(P)(H),
#: FAD(H2), FMN(H2), CoA, acetyl-CoA, SAM, SAH, and common metal ions.
DEFAULT_CURRENCY = frozenset(
    f"CHEBI:{n}"
    for n in (
        15377, 15378, 15379, 16526, 17544, 16240, 28938, 43474, 33019,
        30616, 456216, 456215, 37565, 58189, 46398, 58223, 37563, 58069,
        57540, 57945, 58349, 57783, 57692, 58307, 58210, 57618,
        57287, 57288, 59789, 57856,
        29101, 29103, 17996, 18420, 29108, 29035, 29033, 29034, 29105,
    )
)


def parse_chemical_id(raw: str) -> str | None:
    """Return a CURIE for a chemical id, or None.

    Matches the conventions of ``genesets_workflows.chemid``.

    >>> parse_chemical_id("chebi_15377")
    'CHEBI:15377'
    >>> parse_chemical_id("HMDB00094")
    'hmdb:HMDB0000094'
    >>> parse_chemical_id("C00158")
    'kegg.compound:C00158'
    >>> parse_chemical_id("UniProtKB:P12345") is None
    True
    """
    value = raw.strip()
    match = CHEBI.match(value)
    if match:
        return f"CHEBI:{int(match.group(1))}"
    match = HMDB.match(value)
    if match:
        return f"hmdb:HMDB{int(match.group(1)):07d}"
    match = KEGG_COMPOUND.match(value)
    if match:
        return f"kegg.compound:{match.group(1).upper()}"
    match = LIPIDMAPS.match(value)
    if match:
        return f"lipidmaps:{match.group(1).upper()}"
    match = CAS.match(value)
    if match:
        return f"cas:{match.group(1)}"
    return None


def _chebi_number(curie: str) -> int:
    return int(curie.partition(":")[2])


@dataclass
class ChemicalIndex:
    to_canonical: dict[str, str] = field(default_factory=dict)
    xrefs: dict[str, list[str]] = field(default_factory=dict)
    currency: frozenset[str] = DEFAULT_CURRENCY

    @classmethod
    def load(cls, prep_dir: Path) -> "ChemicalIndex":
        to_canonical: dict[str, str] = {}
        with (prep_dir / "normalization.tsv").open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                to_canonical[row["chebi_id"]] = row["canonical_id"]
        xrefs: dict[str, list[str]] = {}
        xref_path = prep_dir / "xrefs.tsv"
        if xref_path.exists():
            with xref_path.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    xrefs.setdefault(row["xref"], []).append(row["chebi_id"])
        currency = DEFAULT_CURRENCY
        currency_path = prep_dir / "currency.tsv"
        if currency_path.exists():
            with currency_path.open(newline="") as handle:
                currency = frozenset(row["chemical_id"] for row in csv.DictReader(handle, delimiter="\t"))
        return cls(to_canonical, xrefs, currency)

    def canonical(self, raw: str) -> str | None:
        """Canonical ChEBI id for any supported id, or None.

        Without a loaded table, ChEBI ids map to themselves.

        >>> index = ChemicalIndex({"CHEBI:30769": "CHEBI:16947", "CHEBI:16947": "CHEBI:16947"},
        ...                       {"hmdb:HMDB0000094": ["CHEBI:30769"]})
        >>> index.canonical("CHEBI:30769"), index.canonical("HMDB0000094")
        ('CHEBI:16947', 'CHEBI:16947')
        >>> ChemicalIndex().canonical("chebi:17234")
        'CHEBI:17234'
        """
        curie = parse_chemical_id(raw)
        if curie is None:
            return None
        if curie.startswith("CHEBI:"):
            if not self.to_canonical:
                return curie
            return self.to_canonical.get(curie)
        targets = {self.to_canonical[c] for c in self.xrefs.get(curie, ()) if c in self.to_canonical}
        return min(targets, key=_chebi_number) if targets else None

    def is_currency(self, raw: str) -> bool:
        canonical = self.canonical(raw)
        return canonical is not None and canonical in self.currency
