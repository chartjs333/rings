#!/usr/bin/env python3
"""Reproduce a bounded real-neuron FlyWire wiring smoke experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from rings_fly.connectome import load_csv
from rings_fly.cli import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".runtime/flywire_sugar_subset"))
    parser.add_argument("--out", type=Path, default=Path(".runtime/flywire_smoke"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "source_manifest.json").read_text())
    for name, expected in manifest["export_sha256"].items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Export hash mismatch: {name}")
    graph = load_csv(args.data / "nodes.csv", args.data / "connections.csv", normalize="row_abs",
                     source=f"{manifest['repository']}@{manifest['commit']}:FlyWire-v783-induced-subgraph")
    graph.metadata["subset_provenance"] = manifest
    indices = graph.index_by_id
    stimulus = manifest["stimulus_ids"]
    strength = abs(graph.weights[:, [indices[x] for x in stimulus]]).sum(axis=1).A1
    for neuron in stimulus:
        strength[indices[neuron]] = 0
    if strength.max() <= 0:
        raise ValueError("No direct target of the source-defined stimulus remains")
    target = graph.node_ids[int(strength.argmax())]
    readouts = [target]
    if manifest["mn9_present"] and manifest["mn9_id"] in indices:
        readouts.append(manifest["mn9_id"])
    result = run_experiment(graph, out=args.out, stimulate=stimulus, readout=readouts,
                            silence=stimulus, steps=400)
    print(json.dumps({"summary": str(args.out / "summary.json"), "neurons": result["n_nodes"],
                      "edges": result["n_edges"], "stimulated_neurons": len(stimulus),
                      "readout_ids": readouts, "biological_validation": False}, indent=2))


if __name__ == "__main__":
    main()
