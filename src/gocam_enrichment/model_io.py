from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

try:
    from gocam.datamodel import Model
    from gocam.translation.minerva_wrapper import MinervaWrapper
except ImportError as exc:  # pragma: no cover - exercised only in broken envs
    raise RuntimeError(
        "The gocam package is required. In this repo, run with `uv run` so "
        "the adjacent ../gocam package is installed as an editable dependency."
    ) from exc


@dataclass(frozen=True)
class InputObject:
    """One query object supplied by a user."""

    id: str
    type: str | None = None
    modality: str | None = None
    value: float = 1.0
    label: str | None = None

    @property
    def abs_weight(self) -> float:
        return abs(self.value) if self.value != 0 else 1.0


def _model_from_payload(payload: object) -> Model | None:
    if payload is None:
        return None
    if isinstance(payload, Model):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a GO-CAM model mapping, got {type(payload).__name__}")
    if "id" not in payload:
        raise ValueError("Expected a GO-CAM model with an `id` field")
    return Model.model_validate(payload)


def load_models(paths: Iterable[Path], *, taxon: str | None = None) -> list[Model]:
    """Load one or more YAML/JSON GO-CAM model files."""

    models: list[Model] = []
    for path in paths:
        suffix = path.suffix.lower()
        text = path.read_text()
        if suffix == ".json":
            payload = json.loads(text)
            if isinstance(payload, list):
                docs = payload
            else:
                docs = [payload]
        else:
            docs = list(yaml.safe_load_all(text))
        for doc in docs:
            model = _model_from_payload(doc)
            if model is None:
                continue
            if taxon is not None and model.taxon != taxon:
                continue
            models.append(model)
    return models


def fetch_models(
    *,
    model_ids: list[str] | None = None,
    taxon: str | None = None,
    max_models: int | None = None,
) -> list[Model]:
    """Fetch GO-CAM models from the GO API through gocam's MinervaWrapper."""

    wrapper = MinervaWrapper()
    ids: Iterable[str]
    if model_ids:
        ids = model_ids
    else:
        ids = wrapper.models_ids()

    models: list[Model] = []
    fetched = 0
    for model_id in ids:
        if max_models is not None and fetched >= max_models:
            break
        fetched += 1
        model = wrapper.fetch_model(model_id)
        if taxon is not None and model.taxon != taxon:
            continue
        models.append(model)
    return models


def write_models_yaml(models: Iterable[Model], path: Path) -> None:
    """Write models as a multi-document YAML file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for model in models:
            handle.write("---\n")
            yaml.safe_dump(model.model_dump(exclude_none=True), handle, sort_keys=False)


def parse_inputs(path: Path) -> list[InputObject]:
    """Parse TSV, CSV, YAML, or a simple one-object-per-line input file."""

    if path.suffix.lower() in {".yaml", ".yml"}:
        return _parse_yaml_inputs(path)

    raw_lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not raw_lines:
        return []

    delimiter = "\t" if path.suffix.lower() == ".tsv" or "\t" in raw_lines[0] else ","
    first_fields = raw_lines[0].split(delimiter)
    normalized_headers = {field.strip().lower() for field in first_fields}

    if "id" not in normalized_headers and "curie" not in normalized_headers:
        return [_plain_input_line(line, delimiter) for line in raw_lines]

    reader = csv.DictReader(raw_lines, delimiter=delimiter)
    objects: list[InputObject] = []
    for row in reader:
        object_id = (row.get("id") or row.get("curie") or "").strip()
        if not object_id:
            continue
        objects.append(
            InputObject(
                id=object_id,
                type=_clean(row.get("type")),
                modality=_clean(row.get("modality")),
                value=_parse_value(_first_present(row.get("value"), row.get("magnitude"))),
                label=_clean(row.get("label") or row.get("name")),
            )
        )
    return objects


def load_input_metadata(path: Path) -> dict[str, object]:
    """Load optional YAML input-set metadata for reports."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        return {}
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if key != "inputs"}


def _parse_yaml_inputs(path: Path) -> list[InputObject]:
    payload = yaml.safe_load(path.read_text())
    if payload is None:
        return []

    if isinstance(payload, list):
        return [_input_from_yaml_item(item) for item in payload]

    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML input mapping or list in {path}")

    inline_inputs = payload.get("inputs")
    if inline_inputs is not None:
        if not isinstance(inline_inputs, list):
            raise ValueError(f"`inputs` must be a list in {path}")
        return [_input_from_yaml_item(item) for item in inline_inputs]

    inputs_path = payload.get("inputs_path") or payload.get("input_file")
    if inputs_path:
        resolved = Path(str(inputs_path))
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        return parse_inputs(resolved)

    raise ValueError(f"YAML input file must contain `inputs` or `inputs_path`: {path}")


def _input_from_yaml_item(item: object) -> InputObject:
    if isinstance(item, str):
        return InputObject(id=item)
    if not isinstance(item, dict):
        raise ValueError(f"Expected input object mapping, got {type(item).__name__}")

    object_id = item.get("id") or item.get("curie")
    if not object_id:
        raise ValueError("YAML input object is missing `id`")

    return InputObject(
        id=str(object_id).strip(),
        type=_clean(item.get("type")),
        modality=_clean(item.get("modality")),
        value=_parse_value(_first_present(item.get("value"), item.get("magnitude"))),
        label=_clean(item.get("label") or item.get("name")),
    )


def _plain_input_line(line: str, delimiter: str) -> InputObject:
    parts = [part.strip() for part in line.split(delimiter)]
    object_id = parts[0]
    value = _parse_value(parts[1]) if len(parts) > 1 and parts[1] else 1.0
    label = parts[2] if len(parts) > 2 and parts[2] else None
    return InputObject(id=object_id, value=value, label=label)


def _parse_value(value: object | None) -> float:
    if value is None or value == "":
        return 1.0
    try:
        return float(value)
    except ValueError:
        return 1.0


def _first_present(*values: object | None) -> object | None:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
