# Reproducible real FlyWire phase-SIMD benchmark

This is a new experiment based on commit
`71921ec86bed8993ca0d8f34b43eb0f850752c61` on
`codex/rings-fly-connectome`. That revision includes the reference graph engine
and original `wrap()` / `gated_value()` functions, but no AVX2/AVX-512 graph
variants and no source for the previously reported synthetic-graph benchmark.
The original source files remain unchanged.

The recorded full-graph run, all 18 timings, validation and source hashes are
in [the report](../benchmarks/flywire_v783/report.md) and
[machine-readable results](../benchmarks/flywire_v783/benchmark_results.json).
Large source tables, exported CSR binaries, executables and state dumps are
reproduced by the commands below and remain outside Git.

`cpp/benchmark_simd.cpp` uses `rings::RingGraph` directly as its reference.
AVX2 (four doubles) and AVX-512 (eight doubles) vectorize drive/phase arithmetic
across independent neurons. They keep ordered scalar CSR dot products and call
the original wrap/gate functions for each lane. No fast math, FMA contraction,
float32, activity skipping, or approximate wrapping is used. This tests these
specific implementations, not the maximum possible benefit of either ISA.

## Build and prepare

Requires Linux, Python with numpy/scipy/pyarrow, g++ with OpenMP, a little-endian
64-bit x86 host supporting AVX2 and AVX-512F, and the two supplied v783 files.
Use an environment in which `import pyarrow` succeeds. Replace the input paths
below with the locations of your two pinned v783 files. In the Work session,
the installed dependency directory is the workspace's `.runtime-deps` and can
be added to `PYTHONPATH`.

From the repository root:

```sh
mkdir -p build
g++ -O3 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra \
    cpp/benchmark_simd.cpp -o build/benchmark_simd
python scripts/prepare_flywire_benchmark.py \
    --completeness '../upload/Completeness_783(1).txt' \
    --connectivity '../upload/Connectivity_783(1).parquet' \
    --output .runtime/flywire_benchmark/graph.bin
```

The exporter checks the original repository's pinned SHA-256 and Git blob
hashes, exact root IDs and indexes, signs, signed counts and completeness.
It includes all 138,639 nodes and 15,091,983 directed edge rows. There are
54,492,922 summed anatomical synapse counts. This file is the full supplied
source table; no claim is made that it is every connection in a living brain.

The graph uses `W[post,pre]`, source completeness row order, and ascending
presynaptic indexes within each receiving row. Signed duplicate counts are
aggregated before direct division by `max(1,sum(abs(row)))`, matching
`cpp/main.cpp`. Exported ID types are uint64, indexes uint32 and weights
float64. A JSON manifest records source hashes and the exported binary hash.
The binary layout is eight bytes `RINGSCSR`, little-endian uint64 node/edge
counts, uint64 IDs, uint64 CSR row offsets, uint32 pre indexes, float64 weights.

## Verify and measure

```sh
build/benchmark_simd --self-test
build/benchmark_simd --graph .runtime/flywire_benchmark/graph.bin \
    --stimulus 720575940624963786 --threads 1 --steps 400 --validate
build/benchmark_simd --graph .runtime/flywire_benchmark/graph.bin \
    --stimulus 720575940624963786 --threads 4 --steps 400 --validate
python scripts/run_simd_benchmark.py --binary build/benchmark_simd \
    --graph .runtime/flywire_benchmark/graph.bin \
    --out .runtime/flywire_benchmark/timings
```

The self-test covers SIMD tails, wrap boundaries, huge finite phases, signed
inputs and silencing. Full-graph validation compares all activity, phase and
gate elements after every step. Final state files from all timed configurations
are SHA-256 compared as an additional cross-thread and cross-repeat check.

The default timing protocol is 400 timed steps, 50 discarded warmup steps in a
separate network, and three repetitions per configuration (18 runs in total).
Runs are sequential; six configurations are shuffled with seed 783 in each
repetition. Each timed run starts from zero phase/activity and open gates.
Constant input amplitude 1 is supplied to exact root ID
`720575940624963786`, the example in `cpp/README.md` (source row index 69093).
This is a computational stimulation protocol, not a calibrated physiological
experiment. Other neurons receive zero external input.

Default graph parameters are dt=0.001 s, carrier=10 Hz, gain=1,
coupling=0.4 and half-width=pi/2. Timing covers only the step loop; loading,
initialization, warmup, validation and state serialization are excluded.
OpenMP uses `OMP_PROC_BIND=close`, explicit singleton logical-CPU places from
`sched_getaffinity()` and dynamic teams off. The host does not expose physical
core/socket topology, so `OMP_PLACES=cores` produces a runtime warning. That
initial series was stopped and excluded. The final runner verifies actual
thread count, observed CPU IDs, and absence of runtime diagnostics on every run.
Results report median elapsed time, equivalent steps/s, ranges, host/compiler
details, source/executable hashes, all individual runs and state hashes.

Graph adaptation changes the original CIFAR grid's connection operator and
rectifies activity; reusing its phase/gate functions does not establish that
the two models are equivalent. Comparison against previous results obtained
on synthetic topology or another CPU cannot isolate the effect of topology.
These timings alone do not locate the performance bottleneck and do not
validate biological behavior.
