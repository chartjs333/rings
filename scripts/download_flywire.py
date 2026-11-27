#!/usr/bin/env python3
"""Download pinned Shiu/FlyWire v783 data and export a real induced subgraph.

Requires the optional pyarrow dependency. This does not execute upstream code.
The default is a bounded 256-neuron anatomical smoke test, not a complete feeding
circuit or a validated behavioral model. Use --max-nodes 0 for the whole table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
REPOSITORY = "philshiu/Drosophila_brain_model"
FILES = {
    "Completeness_783.csv": (3327347, "b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b", "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    "Connectivity_783.parquet": (100804642, "d386555d1a5f40ebfa1380bcb05b1fab044855fd", "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
}
# Exact IDs from the author's example.ipynb at COMMIT. Its experiment used
# v630; missing IDs in v783 are reported, never replaced or silently relabeled.
SUGAR_IDS = (
    "720575940624963786", "720575940630233916", "720575940637568838",
    "720575940638202345", "720575940617000768", "720575940630797113",
    "720575940632889389", "720575940621754367", "720575940621502051",
    "720575940640649691", "720575940639332736", "720575940616885538",
    "720575940639198653", "720575940620900446", "720575940617937543",
    "720575940632425919", "720575940633143833", "720575940612670570",
    "720575940628853239", "720575940629176663", "720575940611875570",
)
MN9_ID = "720575940660219265"


def verify(path: Path, size: int, git_blob: str, sha256: str) -> None:
    a = hashlib.sha1(f"blob {size}\0".encode())
    b = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            a.update(chunk)
            b.update(chunk)
    if total != size or a.hexdigest() != git_blob or b.hexdigest() != sha256:
        raise ValueError(f"Pinned source integrity mismatch: {path.name}")


def download(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, integrity in FILES.items():
        path = cache / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{name}"
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            verify(temporary, *integrity)
            temporary.replace(path)
        verify(path, *integrity)


def export_subgraph(cache: Path, output: Path, max_nodes: int = 256) -> dict:
    import pyarrow.parquet as pq

    if max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative")
    with (cache / "Completeness_783.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["", "Completed"]:
            raise ValueError(f"Unexpected pinned completeness schema: {header}")
        ids = [row[0] for row in reader]
    if len(ids) != len(set(ids)) or any(not value.isdecimal() for value in ids):
        raise ValueError("Invalid/duplicate exact root IDs")
    numeric_ids = np.array(ids, dtype=np.int64)  # integer-to-integer only
    index = {value: i for i, value in enumerate(ids)}
    seeds = [index[value] for value in SUGAR_IDS if value in index]
    seed_mask = np.zeros(len(ids), dtype=bool)
    seed_mask[seeds] = True
    positive, negative = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
    scores = np.zeros(len(ids), dtype=np.float64)
    columns = ["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory", "Excitatory x Connectivity"]
    parquet = pq.ParquetFile(cache / "Connectivity_783.parquet")
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise ValueError("Unexpected pinned connectivity schema")

    def batches():
        for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
            yield {name: batch.column(i).to_numpy(zero_copy_only=False) for i, name in enumerate(columns)}

    total_synapses = 0
    for data in batches():
        pre, post = data[columns[2]], data[columns[3]]
        counts, signs, signed = data[columns[4]], data[columns[5]], data[columns[6]]
        if any(not np.issubdtype(x.dtype, np.integer) for x in data.values()):
            raise ValueError("Pinned table must have integer indices, IDs, counts and signs")
        if (pre < 0).any() or (post < 0).any() or (pre >= len(ids)).any() or (post >= len(ids)).any():
            raise ValueError("Invalid source index")
        if not np.array_equal(numeric_ids[pre], data[columns[0]]) or not np.array_equal(numeric_ids[post], data[columns[1]]):
            raise ValueError("Source root-ID/index mismatch")
        if (counts < 0).any() or not np.isin(signs, [-1, 1]).all() or not np.array_equal(counts * signs, signed):
            raise ValueError("Invalid signed source connectivity")
        positive |= np.bincount(pre[signs == 1], minlength=len(ids)) > 0
        negative |= np.bincount(pre[signs == -1], minlength=len(ids)) > 0
        mask = seed_mask[pre]
        scores += np.bincount(post[mask], weights=counts[mask], minlength=len(ids))
        total_synapses += int(counts.sum())
    if (positive & negative).any():
        raise ValueError("Source has inconsistent signs for one presynaptic neuron")
    node_signs = positive.astype(np.int8) - negative.astype(np.int8)
    selected = seed_mask.copy()
    if MN9_ID in index:
        selected[index[MN9_ID]] = True
    if max_nodes == 0:
        selected[:] = True
    else:
        if max_nodes < selected.sum():
            raise ValueError("max_nodes is smaller than the retained source-defined seeds")
        candidates = np.lexsort((numeric_ids, -scores))
        for candidate in candidates:
            if selected.sum() >= max_nodes or scores[candidate] == 0:
                break
            selected[candidate] = True
    output.mkdir(parents=True, exist_ok=True)
    with (output / "nodes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["neuron_id", "sign"])
        writer.writerows((value, int(node_signs[i])) for i, value in enumerate(ids) if selected[i])
    retained = boundary_edges = retained_synapses = 0
    with (output / "connections.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pre_id", "post_id", "syn_count"])
        for data in batches():
            pre, post, counts = data[columns[2]], data[columns[3]], data[columns[4]]
            keep = selected[pre] & selected[post]
            boundary_edges += int((selected[pre] ^ selected[post]).sum())
            retained += int(keep.sum())
            retained_synapses += int(counts[keep].sum())
            writer.writerows((ids[int(a)], ids[int(b)], int(c)) for a, b, c in zip(pre[keep], post[keep], counts[keep]))
    manifest = {
        "kind": "real_neuron_level_induced_subgraph" if max_nodes else "real_neuron_level_whole_source_table",
        "dataset": "FlyWire v783 as distributed with Shiu et al.",
        "repository": REPOSITORY, "commit": COMMIT,
        "upstream_example": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/example.ipynb",
        "source_files": {name: {"bytes": value[0], "git_blob_sha1": value[1], "sha256": value[2]} for name, value in FILES.items()},
        "full_source_neurons": len(ids), "full_source_directed_rows": parquet.metadata.num_rows,
        "full_source_synapses": total_synapses, "retained_neurons": int(selected.sum()),
        "retained_directed_rows": retained, "retained_synapses": retained_synapses,
        "excluded_boundary_edge_rows": boundary_edges,
        "selection": "Source example sugar GRNs present in v783, source MN9 when present, then strongest summed direct sugar-GRN targets until max_nodes; all induced edges retained",
        "max_nodes": max_nodes, "source_sugar_ids": list(SUGAR_IDS),
        "stimulus_ids": [value for value in SUGAR_IDS if value in index],
        "missing_source_sugar_ids_in_v783": [value for value in SUGAR_IDS if value not in index],
        "mn9_id": MN9_ID, "mn9_present": MN9_ID in index,
        "sign_policy": "Exact upstream Excitatory column; validated against signed count column; neurons with no outgoing source edges get zero sign, affecting no source edges",
        "scope_warning": "Subset boundaries omit external inputs and outputs; neither complete feeding circuit nor behavioral validation. Ring dynamics are a new computational experiment.",
        "export_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("nodes.csv", "connections.csv")},
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".runtime/shiu_v783"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/flywire_sugar_subset"))
    parser.add_argument("--max-nodes", type=int, default=256)
    args = parser.parse_args()
    download(args.cache)
    manifest = export_subgraph(args.cache, args.output, args.max_nodes)
    print(json.dumps({key: manifest[key] for key in ("retained_neurons", "retained_directed_rows", "retained_synapses", "excluded_boundary_edge_rows", "missing_source_sugar_ids_in_v783")}, indent=2))


if __name__ == "__main__":
    main()
