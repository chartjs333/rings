import tempfile
import unittest
from pathlib import Path

import numpy as np

from rings_fly.connectome import Connectome, load_csv, load_connectome, save_connectome


class ConnectomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.nodes = self.root / "nodes.csv"
        self.edges = self.root / "connections.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, nodes, edges):
        self.nodes.write_text(nodes, encoding="utf-8")
        self.edges.write_text(edges, encoding="utf-8")

    def test_direction_duplicate_edges_large_ids_and_isolated_node(self):
        # Synthetic fixture, deliberately distinct IDs that float64 conflates.
        a, b, c = "720575940600000001", "720575940600000002", "720575940600000003"
        self.write(f"root_id,sign\n{a},1\n{b},-1\n{c},1\n",
                   f"pre_root_id,post_root_id,syn_count\n{a},{b},2\n{a},{b},3\n{b},{a},4\n")
        graph = load_csv(self.nodes, self.edges, normalize="none")
        self.assertEqual(graph.node_ids, (a, b, c))
        np.testing.assert_array_equal(graph.weights.toarray(), [[0, -4, 0], [5, 0, 0], [0, 0, 0]])
        self.assertEqual(graph.index_by_id[b], 1)
        self.assertGreaterEqual(graph.weights.indices.dtype.itemsize, 4)

    def test_row_normalization_after_aggregation_and_signing(self):
        self.write("neuron_id,sign\na,1\nb,-1\nc,1\n",
                   "pre_id,post_id,syn_count\na,c,3\nb,c,1\na,b,0.5\n")
        graph = load_csv(self.nodes, self.edges)
        np.testing.assert_allclose(graph.weights.toarray(), [[0, 0, 0], [.5, 0, 0], [.75, -.25, 0]])
        self.assertEqual(graph.metadata["normalization_rule"], "max(1,row_l1)")

    def test_transmitter_mapping_requires_explicit_policy(self):
        self.write("neuron_id,nt_type\na,ACH\nb,DA\n", "pre_id,post_id,syn_count\na,b,2\nb,a,1\n")
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            load_csv(self.nodes, self.edges)
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            load_csv(self.nodes, self.edges, nt_signs={"ACH": 1})
        graph = load_csv(self.nodes, self.edges, nt_signs={"ACH": 1}, unknown_nt="zero")
        self.assertEqual(graph.metadata["sign_policy"]["unmapped_types"], ["DA"])
        self.assertEqual(graph.weights[0, 1], 0)

    def test_invalid_counts_and_unknown_ids_rejected(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value):
                self.write("neuron_id,sign\na,1\n", f"pre_id,post_id,syn_count\na,a,{value}\n")
                with self.assertRaises(ValueError):
                    load_csv(self.nodes, self.edges)
        self.edges.write_text("pre_id,post_id,syn_count\na,missing,1\n")
        with self.assertRaisesRegex(ValueError, "unknown neuron"):
            load_csv(self.nodes, self.edges)

    def test_duplicate_nodes_rejected(self):
        self.write("neuron_id,sign\na,1\na,-1\n", "pre_id,post_id,syn_count\na,a,1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_csv(self.nodes, self.edges)

    def test_duplicate_headers_complex_weights_and_invalid_nt_mapping(self):
        self.write("neuron_id,sign,sign\na,1,-1\n", "pre_id,post_id,syn_count\na,a,1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate CSV"):
            load_csv(self.nodes, self.edges)
        with self.assertRaisesRegex(ValueError, "Complex"):
            Connectome(("a",), np.array([[1 + 2j]]), {})
        self.write("neuron_id,nt_type\na,ACH\n", "pre_id,post_id,syn_count\na,a,1\n")
        with self.assertRaisesRegex(ValueError, "mapping"):
            load_csv(self.nodes, self.edges, nt_signs=[1])

    def test_bundle_roundtrip_and_hash_tampering(self):
        self.write("neuron_id,sign\na,1\nb,-1\n", "pre_id,post_id,syn_count\na,b,2\n")
        graph = load_csv(self.nodes, self.edges, source={"kind": "synthetic_test"})
        destination = self.root / "graph"
        save_connectome(graph, destination)
        loaded = load_connectome(destination)
        self.assertEqual(loaded.node_ids, graph.node_ids)
        np.testing.assert_array_equal(loaded.weights.toarray(), graph.weights.toarray())
        (destination / "nodes.json").write_text('["different", "b"]\n')
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_connectome(destination)


if __name__ == "__main__":
    unittest.main()
