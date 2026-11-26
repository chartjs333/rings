# Data sources and scientific scope

## Real neuron-level smoke test

`scripts/download_flywire.py` obtains the **actual FlyWire v783 neuron and
connectivity tables distributed by Philip Shiu and colleagues**, not a synthetic
network and not a cell-type-averaged graph. The source is pinned to:

- Repository: <https://github.com/philshiu/Drosophila_brain_model>
- Commit: `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
- Source experiment: [example.ipynb](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/example.ipynb)
- Source model schema: [model.py](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py)

| File | Bytes | SHA-256 |
|---|---:|---|
| Completeness_783.csv | 3,327,347 | `bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311` |
| Connectivity_783.parquet | 100,804,642 | `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347` |

Both files were downloaded and verified against their pinned Git blob hashes
and SHA-256 digests during implementation. The data are kept under ignored
`.runtime/`, not distributed as repository source code.

The completeness table's first column is the exact FlyWire root ID, and its row
order defines the upstream neuron index. The Parquet table has integer columns
`Presynaptic_ID`, `Postsynaptic_ID`, `Presynaptic_Index`, `Postsynaptic_Index`,
`Connectivity`, `Excitatory`, and `Excitatory x Connectivity`. The adapter checks
that both indices map to the stated exact IDs, counts are nonnegative, signs are
±1, and signed counts equal `Connectivity * Excitatory`. No IDs pass through
floating point. Source signs are retained; no new neurotransmitter sign is guessed.

### Reproduce the bounded example

Install the optional Parquet reader, then run from the repository root:

```bash
python -m pip install pyarrow
python scripts/download_flywire.py --max-nodes 256
```

The first download is about 104 MB. The adapter streams Parquet batches rather
than materializing all source edges as a pandas DataFrame. It writes:

- `.runtime/flywire_sugar_subset/nodes.csv`
- `.runtime/flywire_sugar_subset/connections.csv`
- `.runtime/flywire_sugar_subset/source_manifest.json`

Selection is explicit: retain sugar-sensing neuron IDs from the author's pinned
example that exist in v783, retain the example's MN9 if present, then add targets
ranked by total direct synaptic input from those sugar neurons until 256 nodes.
Ties use exact numerical root-ID order. **All edges between selected neurons**
are retained. The source example used v630; any listed IDs missing from v783 are
recorded and never silently replaced. No threshold is applied to selected edges.

This is an anatomical **induced-subgraph smoke test**. External inputs/outputs
are omitted and counted in the manifest. It is not a complete feeding circuit,
does not reproduce the original Shiu LIF dynamics, and does not validate feeding
behavior merely because a node bears the source's MN9 identifier.

`--max-nodes 0` exports the full pinned source table. This option is available
for subsequent work; a whole-brain dynamical run is not implied by the small
example. `--cache` and `--output` select different local directories. Existing
downloads are reused only after integrity verification.

## Canonical CSV import

The normal loader accepts:

```text
nodes.csv:       neuron_id,sign
connections.csv: pre_id,post_id,syn_count
```

`root_id`, `pre_root_id`, and `post_root_id` are supported aliases. Every neuron,
including isolated neurons, is retained in the declared node order. Connection
duplicates are summed before normalization. Unknown IDs, invalid counts and
duplicate IDs are errors. The sparse matrix is `W[post, pre]`, so `W @ activity`
propagates from presynaptic to postsynaptic nodes.

A node's explicit `sign` is -1, 0, or +1 and is applied exactly once. Alternatively
nodes may supply `nt_type`, but the caller must supply `nt_signs` explicitly:

```python
from rings_fly.connectome import load_csv, save_connectome
graph = load_csv(
    "nodes.csv", "connections.csv",
    nt_signs={"ACH": 1, "GABA": -1, "GLUT": -1},
    unknown_nt="error",
    normalize="row_abs",
)
save_connectome(graph, ".runtime/imported_graph")
```

That example sign map is a **chosen model approximation**, not a universal rule
about receptors. Unmapped transmitters (including modulators) cause an error.
The explicit `unknown_nt="zero"` alternative removes their outgoing numerical
influence and records which types were affected. For the downloaded Shiu data,
the adapter instead uses the already supplied per-neuron source signs.

`row_abs` divides each row by `max(1, sum(abs(row)))`; `none` preserves signed
counts. This changes the numerical interaction strengths and is recorded in
metadata. Counts and normalized weights are not measured physiological synaptic
efficacies. With row normalization, incoming absolute row sums are at most one;
this is a numerical bound, not proof that gated phase dynamics reproduce biology.

Saved bundles contain `weights.npz`, `nodes.json` and `metadata.json`, with
SHA-256 checks on the matrix and exact ID array. Loading verifies these hashes.

## Licences and attribution

The [FlyWire public-release guidelines](https://join.flywire.ai/guidelines)
state **CC BY-NC 4.0** for public-release data. Do not assume that the upstream
repository's MIT code licence or this project's code licence relicenses the
underlying connectome. Source data and locally derived subsets remain separate
from this repository's implementation. Review the applicable dataset terms for
the intended reuse; no grant for commercial data use is made here.

The integration code is original. No Eon GPL implementation was copied.

Primary scientific references:

- Shiu et al. (2024), *A Drosophila computational brain model reveals sensorimotor
  processing*, Nature: <https://doi.org/10.1038/s41586-024-07763-9>.
- Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature:
  <https://doi.org/10.1038/s41586-024-07558-y>.
- The pinned author repository and example above document the exact downloaded
  files and source-defined neurons used by this implementation.

The synthetic graphs in unit tests test software properties only. They must not
be described as biological datasets or evidence of biological validity.
