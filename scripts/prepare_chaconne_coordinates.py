#!/usr/bin/env python3
"""Prepare anatomical anchor coordinates for the pinned FlyWire v783 graph.

Python 3 standard library only. Run from the directory containing .runtime, or
pass explicit paths. Output columns are root_id,x_um,y_um,z_um; the exact integer
IDs and row order come from Shiu's pinned Completeness_783.csv or its verified
RINGSCSR graph. Missing coordinate sources download automatically; existing
inputs must match their hashes. Use --no-download for strictly offline operation.

Windows / launcher command (no Python packages required):
  python scripts/prepare_chaconne_coordinates.py --graph .runtime/java_graph/graph.bin
  Add --output PATH to select a different destination.

Primary coordinates: flyconnectome/flywire_annotations v3.2.0, commit
a83b2776d60d5764cef36b927f5f9679c16c47a2, Supplemental_file1_neuron_annotations.tsv.
Its pos_x/y/z fields are neuron anchors (usually on the backbone), NOT somas,
in 4 x 4 x 40 nm voxel units. soma_x/y/z are deliberately not substituted.
Documentation: https://github.com/flyconnectome/flywire_annotations/blob/
a83b2776d60d5764cef36b927f5f9679c16c47a2/supplemental_files/README.md

For the 14 graph neurons absent from that TSV, use an actual marked position
from Codex FAFB v783 coordinates.csv.gz, GCS generation 1733428431583877.
Those positions are already nanometers. Choose the lexicographically smallest
(x_nm, y_nm, z_nm, supervoxel_id), never an average or an invented coordinate.
The JSON sidecar records every fallback selection and all source hashes.

Data license: FlyWire public-release data are CC BY-NC 4.0 per
https://flywire.ai/guidelines ; https://creativecommons.org/licenses/by-nc/4.0/
Preserve source attribution and applicable data terms when sharing derivatives.
Citations: Schlegel et al. Nature (2024), Dorkenwald et al. Nature (2024), plus
the v3.2.0 annotation release's requested credits: Matsliah et al. Nature (2024),
Berg et al. Cell (2026), and Tastekin et al. Cell (2026).
This script changes coordinate units and joins existing anatomical positions;
it does not turn a marked anchor into a measured soma or a physiological model.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import urllib.request


ANNOTATION_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"
SOURCE = {
    "completeness": {
        "bytes": 3327347,
        "sha256": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
        "url": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Completeness_783.csv",
    },
    "annotations": {
        "bytes": 31720298,
        "sha256": "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
        "url": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/" + ANNOTATION_COMMIT + "/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    },
    "codex_coordinates": {
        "bytes": 5314546,
        "sha256": "14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05",
        "url": "https://storage.googleapis.com/download/storage/v1/b/flywire-data/o/codex%2Fdata%2Ffafb%2F783%2Fcoordinates.csv.gz?generation=1733428431583877&alt=media",
    },
}
EXPECTED_GRAPH_ROWS = 138639
EXPECTED_ANNOTATION_ROWS = 139248
EXPECTED_CODEX_ROWS = 238909
EXPECTED_CODEX_IDS = 139255
PINNED_GRAPH = {
    "bytes": 183322052,
    "sha256": "61203093733cffe0cde4fb60c5059fb2b76ddb1bb8576e574320ef765079ccb9",
    "description": "RINGSCSR graph exported from the pinned Shiu FlyWire v783 files",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, source: dict) -> None:
    if path.stat().st_size != source["bytes"] or file_sha256(path) != source["sha256"]:
        raise ValueError("Pinned source size/hash mismatch: " + str(path))


def ensure_source(path: Path, source: dict, download: bool) -> None:
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                with urllib.request.urlopen(source["url"], timeout=60) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            verify(temporary, source)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    verify(path, source)


def exact_id(value: str) -> str:
    if not value or not value.isascii() or not value.isdecimal() or int(value) >= 2**64:
        raise ValueError("Invalid exact uint64 ID: " + repr(value))
    return value


def coordinate(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid coordinate: " + repr(value)) from exc
    if not number.is_finite():
        raise ValueError("Nonfinite coordinate: " + repr(value))
    return number


def decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def prepare(completeness: Path, annotations: Path, codex_coordinates: Path,
            output: Path, *, download: bool = True, graph: Path | None = None) -> dict:
    paths = {"annotations": annotations, "codex_coordinates": codex_coordinates}
    sources = dict(SOURCE)
    if graph is None:
        paths["completeness"] = completeness
    else:
        paths["graph"] = graph
        sources["graph"] = PINNED_GRAPH
    manifest_path = output.with_suffix(".json")
    if output.resolve() == manifest_path.resolve() or any(
        dest.resolve() == source.resolve()
        for dest in (output, manifest_path) for source in paths.values()
    ):
        raise ValueError("Output and manifest must be distinct from input files")
    for name, path in paths.items():
        ensure_source(path, sources[name], download and name != "graph")

    if graph is not None:
        with graph.open("rb") as handle:
            if handle.read(8) != b"RINGSCSR":
                raise ValueError("Invalid graph magic")
            n, m = struct.unpack("<QQ", handle.read(16))
            if n != EXPECTED_GRAPH_ROWS or m != 15091983:
                raise ValueError("Unexpected graph dimensions")
            ids = [str(value[0]) for value in struct.iter_unpack("<Q", handle.read(n * 8))]
    else:
        with completeness.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if next(reader) != ["", "Completed"]:
                raise ValueError("Unexpected Completeness header")
            ids = []
            for row in reader:
                if len(row) != 2 or row[1] != "True":
                    raise ValueError("Invalid Completeness row")
                ids.append(exact_id(row[0]))
    wanted = set(ids)
    if len(ids) != EXPECTED_GRAPH_ROWS or len(wanted) != len(ids):
        raise ValueError("Wrong graph dimensions or duplicate IDs")

    points = {}
    annotation_ids = set()
    with annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"root_id", "pos_x", "pos_y", "pos_z"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing annotation coordinate columns")
        for row in reader:
            root_id = exact_id(row["root_id"])
            if root_id in annotation_ids:
                raise ValueError("Duplicate annotation root ID")
            annotation_ids.add(root_id)
            if root_id in wanted:
                xyz_voxel = tuple(coordinate(row["pos_" + axis]) for axis in "xyz")
                points[root_id] = tuple(v * scale for v, scale in zip(
                    xyz_voxel, (Decimal("0.004"), Decimal("0.004"), Decimal("0.04"))))
    if len(annotation_ids) != EXPECTED_ANNOTATION_ROWS:
        raise ValueError("Wrong annotation dimensions")
    primary_count = len(points)
    missing = wanted - points.keys()
    fallback = {}
    codex_ids = set()
    codex_rows = 0
    with gzip.open(codex_coordinates, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["root_id", "position", "supervoxel_id"]:
            raise ValueError("Unexpected Codex coordinates header")
        for row in reader:
            codex_rows += 1
            root_id = exact_id(row["root_id"])
            codex_ids.add(root_id)
            if root_id not in missing:
                continue
            position = row["position"].strip()
            if not (position.startswith("[") and position.endswith("]")):
                raise ValueError("Malformed Codex position")
            parts = position[1:-1].split()
            if len(parts) != 3:
                raise ValueError("Codex position must have three coordinates")
            xyz_nm = tuple(coordinate(value) for value in parts)
            supervoxel = int(exact_id(row["supervoxel_id"]))
            candidate = (*xyz_nm, supervoxel)
            if root_id not in fallback or candidate < fallback[root_id]:
                fallback[root_id] = candidate
    if codex_rows != EXPECTED_CODEX_ROWS or len(codex_ids) != EXPECTED_CODEX_IDS:
        raise ValueError("Wrong Codex coordinates dimensions")
    for root_id, candidate in fallback.items():
        points[root_id] = tuple(value / Decimal(1000) for value in candidate[:3])
    unavailable = sorted(wanted - points.keys(), key=int)
    if unavailable:
        raise ValueError("No real coordinate available for graph IDs: " + ",".join(unavailable))
    if primary_count != 138625 or len(fallback) != 14 or len(points) != len(ids):
        raise ValueError("Unexpected pinned-source join coverage")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["root_id", "x_um", "y_um", "z_um"])
            for root_id in ids:
                writer.writerow([root_id, *(decimal_text(v) for v in points[root_id])])
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    bounds = {axis: {"min": decimal_text(min(p[i] for p in points.values())),
                     "max": decimal_text(max(p[i] for p in points.values()))}
              for i, axis in enumerate(("x_um", "y_um", "z_um"))}
    report = {
        "schema": "rings-chaconne-anatomical-anchors-v1", "dataset": "FlyWire FAFB v783",
        "coordinate_kind": "anatomical marked anchors, not soma positions",
        "units": "micrometers", "axes": ["x", "y", "z"],
        "row_order": "Exact pinned graph / Completeness_783.csv order; join by root_id",
        "rows": len(ids), "annotation_anchor_rows": primary_count,
        "codex_fallback_rows": len(fallback), "unavailable_ids": unavailable,
        "annotation_release": "v3.2.0", "annotation_commit": ANNOTATION_COMMIT,
        "annotation_voxel_to_um": ["0.004", "0.004", "0.04"],
        "codex_nm_to_um": "0.001", "codex_generation": "1733428431583877",
        "fallback_rule": "Lexicographic minimum (x_nm,y_nm,z_nm,supervoxel_id) among actual marked points",
        "fallback_selections": [{"root_id": root_id,
             "position_nm": [decimal_text(v) for v in fallback[root_id][:3]],
             "supervoxel_id": str(fallback[root_id][3])}
             for root_id in sorted(fallback, key=int)],
        "bounds_um": bounds,
        "data_license": "CC BY-NC 4.0; https://flywire.ai/guidelines",
        "sources": {name: dict(sources[name], local_path=str(path)) for name, path in paths.items()},
        "output": {"path": str(output), "bytes": output.stat().st_size,
                   "sha256": file_sha256(output)},
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, help="Use exact IDs from this pinned RINGSCSR graph instead of Completeness")
    parser.add_argument("--completeness", type=Path, default=Path(".runtime/shiu_v783/Completeness_783.csv"))
    parser.add_argument("--annotations", type=Path, default=Path(".runtime/coords-src/Supplemental_file1_neuron_annotations.tsv"))
    parser.add_argument("--codex-coordinates", type=Path, default=Path(".runtime/coords-src/coordinates.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/chaconne-coordinates.csv"))
    fetch = parser.add_mutually_exclusive_group()
    fetch.add_argument("--download", dest="download", action="store_true", help="Download missing pinned inputs (default)")
    fetch.add_argument("--no-download", dest="download", action="store_false", help="Fail if any input is absent")
    parser.set_defaults(download=True)
    args = parser.parse_args()
    try:
        result = prepare(args.completeness, args.annotations, args.codex_coordinates,
                         args.output, download=args.download, graph=args.graph)
    except (OSError, ValueError) as exc:
        parser.exit(1, "prepare_chaconne_coordinates: " + str(exc) + "\n")
    print(json.dumps({key: result[key] for key in (
        "rows", "annotation_anchor_rows", "codex_fallback_rows", "unavailable_ids", "output")}, indent=2))


if __name__ == "__main__":
    main()
