"""Strict, provenance-aware CSV import of directed connectomes.

The matrix convention is W[postsynaptic, presynaptic]. IDs remain strings.
Weights are anatomical counts with an explicit sign convention, not measured
physiological synaptic efficacies.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Connectome:
    node_ids: tuple[str, ...]
    weights: sparse.csr_matrix
    metadata: dict

    def __post_init__(self):
        self.node_ids = tuple(self.node_ids)
        if not self.node_ids or any(not isinstance(x, str) or not x.strip() for x in self.node_ids):
            raise ValueError("node_ids must be nonempty, exact strings")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("Duplicate neuron IDs")
        if np.issubdtype(getattr(self.weights, "dtype", np.asarray(self.weights).dtype), np.complexfloating):
            raise ValueError("Complex weights are not supported")
        self.weights = sparse.csr_matrix(self.weights, dtype=np.float64, copy=True)
        if self.weights.shape != (len(self.node_ids), len(self.node_ids)):
            raise ValueError("Matrix dimensions do not match node_ids")
        self.weights.sum_duplicates()
        self.weights.eliminate_zeros()
        self.weights.sort_indices()
        if not np.isfinite(self.weights.data).all():
            raise ValueError("Non-finite weights")
        # Explicitly avoid 16-bit neuron indices, including >65,535-neuron graphs.
        self.weights.indices = self.weights.indices.astype(np.int64)
        self.weights.indptr = self.weights.indptr.astype(np.int64)

    @property
    def index_by_id(self) -> dict[str, int]:
        return {node_id: i for i, node_id in enumerate(self.node_ids)}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column(frame: pd.DataFrame, alternatives: tuple[str, ...]) -> str:
    present = [name for name in alternatives if name in frame.columns]
    if len(present) != 1:
        raise ValueError(f"Expected exactly one column from {alternatives}; found {present}")
    return present[0]


def _read_csv(path: str | Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    # dtype=str is essential: FlyWire root IDs exceed IEEE-754 exact integers.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names")
    return frame


def _signs(nodes: pd.DataFrame, unknown_nt: str,
           nt_signs: Mapping[str, float] | None) -> tuple[np.ndarray, dict]:
    if unknown_nt not in {"error", "zero"}:
        raise ValueError("unknown_nt must be 'error' or explicit 'zero'")
    if "sign" in nodes:
        signs = pd.to_numeric(nodes["sign"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(signs).all() or not np.isin(signs, [-1, 0, 1]).all():
            raise ValueError("Node signs must be finite -1, 0 or +1")
        return signs, {"policy": "explicit_presynaptic_node_sign", "zero_sign_nodes": int((signs == 0).sum())}
    if "nt_type" not in nodes:
        raise ValueError("nodes.csv requires sign or nt_type")
    # No universal neurotransmitter-to-sign mapping is biologically valid.
    # The caller must state the model assumption; no modulators are guessed.
    if nt_signs is not None and not isinstance(nt_signs, Mapping):
        raise ValueError("nt_signs must be a mapping of transmitter names to signs")
    policy = {str(k).strip().upper(): float(v) for k, v in (nt_signs or {}).items()}
    if any(not np.isfinite(v) or v not in {-1, 0, 1} for v in policy.values()):
        raise ValueError("nt_signs values must be -1, 0 or +1")
    labels = nodes["nt_type"].str.strip().str.upper()
    unknown = sorted(set(labels) - set(policy))
    if unknown and unknown_nt == "error":
        raise ValueError(f"Unmapped neurotransmitters {unknown}; supply explicit nt_signs or node sign column")
    signs = np.array([policy.get(label, 0) for label in labels], dtype=float)
    return signs, {"policy": "explicit_nt_mapping", "mapping": policy,
                   "unknown_nt": unknown_nt, "unmapped_types": unknown,
                   "zero_sign_nodes": int((signs == 0).sum())}


def load_csv(nodes_path: str | Path, connections_path: str | Path, *,
             normalize: str = "row_abs", source=None, unknown_nt: str = "error",
             nt_signs: Mapping[str, float] | None = None) -> Connectome:
    """Load canonical or FlyWire CSVs without integer-ID precision loss.

    Nodes: neuron_id (or root_id), plus sign or nt_type. Connections: pre_id,
    post_id (or pre_root_id, post_root_id), syn_count. Duplicate directed edges
    are summed before normalization. Self edges and isolated nodes are retained.
    normalize='row_abs' divides each row by max(1, sum(abs(row))); 'none'
    retains signed counts. Unknown neurotransmitters are errors by default.
    """
    if normalize not in {"row_abs", "none"}:
        raise ValueError("normalize must be 'row_abs' or 'none'")
    nodes = _read_csv(nodes_path)
    edges = _read_csv(connections_path)
    node_col = _column(nodes, ("neuron_id", "root_id"))
    pre_col = _column(edges, ("pre_id", "pre_root_id"))
    post_col = _column(edges, ("post_id", "post_root_id"))
    if "syn_count" not in edges:
        raise ValueError("Missing syn_count column")
    ids = tuple(nodes[node_col])
    if not ids or any(not value or value != value.strip() for value in ids):
        raise ValueError("IDs must be nonempty exact strings without surrounding whitespace")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate neuron IDs")
    index = {node_id: i for i, node_id in enumerate(ids)}
    pre = edges[pre_col].map(index)
    post = edges[post_col].map(index)
    if pre.isna().any() or post.isna().any():
        unknown = sorted((set(edges[pre_col]) | set(edges[post_col])) - set(ids))
        raise ValueError(f"Connections reference unknown neuron IDs: {unknown[:8]}")
    counts = pd.to_numeric(edges["syn_count"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("syn_count must be finite and nonnegative")
    signs, policy = _signs(nodes, unknown_nt, nt_signs)
    pre = pre.to_numpy(dtype=np.int64)
    post = post.to_numpy(dtype=np.int64)
    weights = sparse.csr_matrix((counts * signs[pre], (post, pre)), shape=(len(ids), len(ids)))
    weights.sum_duplicates()
    weights.eliminate_zeros()
    if not np.isfinite(weights.data).all():
        raise ValueError("Connection aggregation overflowed")
    row_abs = np.asarray(abs(weights).sum(axis=1)).ravel()
    if not np.isfinite(row_abs).all():
        raise ValueError("Row sums overflowed")
    if normalize == "row_abs":
        weights = sparse.diags(1 / np.maximum(1, row_abs)) @ weights
    metadata = {
        "schema": "rings-fly-connectome-v1", "orientation": "post_by_pre",
        "node_count": len(ids), "input_edge_rows": len(edges), "stored_edges": int(weights.nnz),
        "normalization": normalize, "normalization_rule": "max(1,row_l1)" if normalize == "row_abs" else None,
        "sign_policy": policy, "source": source,
        "source_files": {
            "nodes": {"filename": Path(nodes_path).name, "sha256": file_sha256(nodes_path)},
            "connections": {"filename": Path(connections_path).name, "sha256": file_sha256(connections_path)},
        },
        "weight_interpretation": "Signed anatomical counts, not measured physiological efficacies",
    }
    return Connectome(ids, weights, metadata)


def save_connectome(graph: Connectome, directory: str | Path) -> None:
    """Write sparse NPZ + exact string IDs + a hash-verifiable JSON manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "weights.npz", graph.weights, compressed=True)
    (directory / "nodes.json").write_text(json.dumps(graph.node_ids, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = dict(graph.metadata)
    metadata["bundle_schema"] = "rings-fly-connectome-bundle-v1"
    metadata["orientation"] = "post_by_pre"
    metadata["bundle_sha256"] = {name: file_sha256(directory / name) for name in ("weights.npz", "nodes.json")}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_connectome(directory: str | Path) -> Connectome:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("bundle_schema") != "rings-fly-connectome-bundle-v1" or metadata.get("orientation") != "post_by_pre":
        raise ValueError("Unsupported connectome bundle schema/orientation")
    for name in ("weights.npz", "nodes.json"):
        if metadata.get("bundle_sha256", {}).get(name) != file_sha256(directory / name):
            raise ValueError(f"SHA-256 mismatch: {name}")
    ids = json.loads((directory / "nodes.json").read_text(encoding="utf-8"))
    return Connectome(tuple(ids), sparse.load_npz(directory / "weights.npz"), metadata)
