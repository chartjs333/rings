"""End-to-end checks at the CSV, experiment and artifact boundaries."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from rings_fly.cli import main, run_experiment
from rings_fly.connectome import Connectome, load_connectome
from rings_fly.dynamics import DynamicsConfig


class IntegrationTests(unittest.TestCase):
    def test_demo_is_explicitly_synthetic_and_silencing_blocks_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["demo", "--out", str(out), "--steps", "200",
                               "--onset", "0", "--offset", "150"])
            self.assertEqual(status, 0)
            report = json.loads((out / "summary.json").read_text())
            self.assertFalse(report["biological_validation"])
            self.assertTrue(report["data"]["synthetic"])
            self.assertEqual(report["output_units"], "dimensionless_nonnegative_activity")
            self.assertEqual(set(report["conditions"]),
                             {"rings", "rings_silenced", "fixed_gate", "fixed_gate_silenced"})
            for model in ("rings", "fixed_gate"):
                self.assertGreater(report["conditions"][model]["readouts"]["toy-output"]["mean_activity"], 0)
                self.assertEqual(report["conditions"][model + "_silenced"]["readouts"]["toy-output"]["mean_activity"], 0)
            with (out / "rings_trace.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 400)  # 200 steps × two readouts.
            self.assertEqual({row["neuron_id"] for row in rows}, {"toy-relay", "toy-output"})
            self.assertEqual(float(rows[-1]["time_s"]), 0.2)

    def test_unknown_neuron_fails_before_creating_output(self):
        graph = Connectome(("known",), sparse.csr_matrix((1, 1)), {})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "Unknown stimulus"):
                run_experiment(graph, out=out, stimulate=["unknown"], readout=["known"],
                               silence=[], steps=10)
            self.assertFalse(out.exists())

    def test_per_node_parameters_are_recorded_without_scalar_coercion(self):
        graph = Connectome(("a", "b"),
                           sparse.csr_matrix(([1.0], ([1], [0])), shape=(2, 2)), {})
        config = DynamicsConfig(carrier_hz=np.array([0.0, 1.0]),
                                phase_sensitivity=np.array([0.0, 2.0]),
                                half_width=np.array([np.pi, np.pi / 2]))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "arrays"
            run_experiment(graph, out=out, stimulate=["a"], readout=["b"], silence=[],
                           steps=10, onset=0, offset=8, config=config)
            report = json.loads((out / "summary.json").read_text())
            self.assertEqual(report["parameters"]["carrier_hz"], [0.0, 1.0])
            self.assertEqual(report["parameters"]["phase_sensitivity"], [0.0, 2.0])
            self.assertEqual(report["conditions"]["rings"]["phase_sensitivity"], [0.0, 2.0])
            self.assertEqual(report["conditions"]["fixed_gate"]["phase_sensitivity"], 0.0)

    def test_cli_import_accepts_explicit_nt_mapping_and_preserves_large_ids(self):
        first, second = "720575940000000001", "720575940000000002"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nodes, edges, mapping, bundle = (root / name for name in
                                            ("nodes.csv", "edges.csv", "signs.json", "bundle"))
            nodes.write_text(f"root_id,nt_type\n{first},ACH\n{second},GABA\n")
            edges.write_text(f"pre_root_id,post_root_id,syn_count\n{first},{second},3\n")
            mapping.write_text(json.dumps({"ACH": 1, "GABA": -1}))
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["import-csv", "--neurons", str(nodes), "--connections", str(edges),
                               "--out", str(bundle), "--source", "synthetic-test-fixture",
                               "--nt-signs", str(mapping)])
            self.assertEqual(status, 0)
            graph = load_connectome(bundle)
            self.assertEqual(graph.node_ids, (first, second))
            self.assertEqual(graph.weights[1, 0], 1.0)
            self.assertEqual(graph.weights[0, 1], 0.0)
            self.assertEqual(graph.metadata["sign_policy"]["mapping"], {"ACH": 1.0, "GABA": -1.0})


if __name__ == "__main__":
    unittest.main()
