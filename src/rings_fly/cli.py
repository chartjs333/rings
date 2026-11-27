"""Import a connectome and run matched phase/fixed-gate perturbation experiments.

Copyright (C) 2026 Harutyun Madoev. GPL-3.0-or-later; see LICENSE.txt.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from .connectome import Connectome, load_csv, load_connectome, save_connectome
from .dynamics import DynamicsConfig, RingFlyDynamics, GateSummary


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fresh_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _indices(graph: Connectome, ids: list[str], role: str) -> list[int]:
    mapping = graph.index_by_id
    missing = [node for node in ids if node not in mapping]
    if missing:
        raise ValueError(f"Unknown {role} IDs: {missing}")
    return list(dict.fromkeys(mapping[node] for node in ids))


def _number(value):
    return float(value) if np.isfinite(value) else None


def _parameter(value):
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array.tolist()


def run_experiment(graph: Connectome, *, out: Path, stimulate: list[str],
                   readout: list[str], silence: list[str], steps: int = 400,
                   onset: int | None = None, offset: int | None = None,
                   amplitude: float = 1.0, config: DynamicsConfig | None = None) -> dict:
    """Four matched conditions when a silencing target is supplied; no fitting."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and nonnegative")
    if not stimulate or not readout:
        raise ValueError("Specify at least one stimulus and one readout neuron")
    if len(set(readout)) > 32:
        raise ValueError("Select at most 32 readout neurons; full state traces are not saved")
    onset = steps // 10 if onset is None else onset
    offset = max(onset + 1, steps * 8 // 10) if offset is None else offset
    if not 0 <= onset < offset <= steps:
        raise ValueError("Require 0 <= onset < offset <= steps")
    stim_idx = _indices(graph, stimulate, "stimulus")
    read_idx = _indices(graph, readout, "readout")
    silence_idx = _indices(graph, silence, "silencing")
    config = config or DynamicsConfig()
    # Validate the dynamics before creating any output artifact.
    RingFlyDynamics(graph.weights, config)
    _fresh_directory(out)
    conditions = [("rings", config, [])]
    if silence_idx:
        conditions.append(("rings_silenced", config, silence_idx))
    fixed = replace(config, phase_sensitivity=0.0)
    conditions.append(("fixed_gate", fixed, []))
    if silence_idx:
        conditions.append(("fixed_gate_silenced", fixed, silence_idx))
    report = {
        "schema": "rings-fly-experiment-v1",
        "model": "experimental_rectified_phase_graph",
        "biological_validation": False,
        "output_units": "dimensionless_nonnegative_activity",
        "n_nodes": len(graph.node_ids), "n_edges": int(graph.weights.nnz),
        "data": graph.metadata,
        "parameters": {
            "dt_s": config.dt_s, "carrier_hz": _parameter(config.carrier_hz),
            "phase_sensitivity": _parameter(config.phase_sensitivity),
            "coupling": config.coupling, "half_width": _parameter(config.half_width),
            "initial_phase": 0.0,
        },
        "stimulus": {"ids": stimulate, "amplitude": amplitude, "onset_step": onset, "offset_step_exclusive": offset},
        "silenced_ids": silence, "readout_ids": [graph.node_ids[i] for i in read_idx],
        "steps": steps, "duration_s": steps * config.dt_s,
        "conditions": {},
    }
    external = np.zeros(len(graph.node_ids), dtype=float)
    for name, setting, targets in conditions:
        net = RingFlyDynamics(graph.weights, setting)
        net.set_silenced(targets)
        summary = GateSummary(len(graph.node_ids))
        with (out / f"{name}_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "time_s", "neuron_id", "activity", "gate_open", "phase_rad"])
            for step in range(steps):
                external.fill(0.0)
                if onset <= step < offset:
                    external[stim_idx] = amplitude
                result = net.step(external)
                summary.update(result)
                for index in read_idx:
                    writer.writerow([step + 1, (step + 1) * setting.dt_s, graph.node_ids[index],
                                     float(result.activity[index]), int(result.gates[index]), float(result.phase[index])])
        stats = summary.result()
        report["conditions"][name] = {
            "phase_sensitivity": _parameter(setting.phase_sensitivity),
            "silenced_ids": [graph.node_ids[i] for i in targets],
            "readouts": {graph.node_ids[i]: {
                "mean_activity": float(stats.mean_activity[i]), "gate_duty_cycle": float(stats.duty_cycle[i]),
                "p01": _number(stats.p01[i]), "p10": _number(stats.p10[i]),
                "valid01": bool(stats.valid01[i]), "valid10": bool(stats.valid10[i]),
            } for i in read_idx},
        }
    _json(out / "summary.json", report)
    return report


def _toy_graph() -> Connectome:
    """Synthetic wiring fixture, deliberately not described as Drosophila data."""
    # rows=destination, columns=source: sensory->relay->output; inhibitor->output.
    weights = sparse.csr_matrix(([1.0, .5, -.5], ([1, 2, 2], [0, 1, 3])), shape=(4, 4))
    return Connectome(("toy-sensory", "toy-relay", "toy-output", "toy-inhibitor"), weights,
                      {"source": "synthetic four-node wiring fixture", "synthetic": True,
                       "normalization": "row_abs", "biological_circuit": False})


def _add_run_options(parser):
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--onset", type=int)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--carrier-hz", type=float, default=10.0)
    parser.add_argument("--phase-sensitivity", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=.4)
    parser.add_argument("--half-width", type=float, default=math.pi / 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-csv", help="Read neuron and connection CSV files into a sparse graph")
    imp.add_argument("--neurons", type=Path, required=True)
    imp.add_argument("--connections", type=Path, required=True)
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument("--normalization", choices=["row_abs", "none"], default="row_abs")
    imp.add_argument("--source", required=True, help="Dataset version/source URL for provenance")
    imp.add_argument("--unknown-nt", choices=["error", "zero"], default="error")
    imp.add_argument("--nt-signs", type=Path,
                     help="JSON object of explicit neurotransmitter sign assumptions; unnecessary when CSV has sign column")
    demo = sub.add_parser("demo", help="Run a clearly labelled synthetic wiring check")
    _add_run_options(demo)
    run = sub.add_parser("run", help="Run phase/fixed-gate and optional silencing comparisons on an imported graph")
    _add_run_options(run)
    run.add_argument("--connectome", type=Path, required=True)
    run.add_argument("--stimulate", action="append", required=True, metavar="NEURON_ID")
    run.add_argument("--readout", action="append", required=True, metavar="NEURON_ID")
    run.add_argument("--silence", action="append", default=[], metavar="NEURON_ID")
    args = parser.parse_args(argv)
    try:
        if args.command == "import-csv":
            nt_signs = json.loads(args.nt_signs.read_text(encoding="utf-8")) if args.nt_signs else None
            graph = load_csv(args.neurons, args.connections, normalize=args.normalization,
                             source=args.source, unknown_nt=args.unknown_nt, nt_signs=nt_signs)
            _fresh_directory(args.out)
            save_connectome(graph, args.out)
            print(json.dumps({"graph": str(args.out), "n_nodes": len(graph.node_ids), "n_edges": int(graph.weights.nnz)}))
            return 0
        config = DynamicsConfig(dt_s=args.dt_ms / 1000, carrier_hz=args.carrier_hz,
                                phase_sensitivity=args.phase_sensitivity,
                                coupling=args.coupling, half_width=args.half_width)
        if args.command == "demo":
            graph = _toy_graph()
            stimulate, readout, silence = ["toy-sensory"], ["toy-relay", "toy-output"], ["toy-relay"]
        else:
            graph = load_connectome(args.connectome)
            stimulate, readout, silence = args.stimulate, args.readout, args.silence
        report = run_experiment(graph, out=args.out, stimulate=stimulate, readout=readout,
                                silence=silence, steps=args.steps, onset=args.onset, offset=args.offset,
                                amplitude=args.amplitude, config=config)
        print(json.dumps({"summary": str(args.out / "summary.json"), "conditions": list(report["conditions"]),
                          "n_nodes": report["n_nodes"], "biological_validation": False}))
        return 0
    except (ValueError, OSError, FloatingPointError) as error:
        print(f"rings-fly: {error}", file=sys.stderr)
        return 2

