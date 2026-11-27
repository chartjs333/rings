# Java RINGS graph implementation and C++ comparison

This Java 21 implementation ports the graph dynamics from
`cpp/rings_graph.hpp`, preserving binary64 operation order, the original
phase wrap formula, signed row-normalized CSR weights, synchronous state
updates and positive-zero closed gates. It does not port the CIFAR classifier.

The default CPU implementation uses primitive arrays and persistent platform
threads. Four persistent virtual workers are available as a separate control;
there is no virtual thread or object per neuron. Graph row offsets remain
64-bit and edge indexes 32-bit, matching the C++ graph representation.

## Build

From the repository root, with JDK 21+:

```sh
mkdir -p build/java
javac --release 21 -d build/java java/RingsBenchmark.java
java -cp build/java RingsBenchmark --self-test
```

The Java CLI is platform independent, including Windows. The comparison
orchestrator uses Linux process affinity and should run under Linux or WSL2.
The C++ reference remains unchanged. See
[full-data preparation](../docs/BENCHMARK_FLYWIRE.md).

```sh
g++ -O3 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra cpp/benchmark_simd.cpp -o build/benchmark_simd
g++ -O3 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra cpp/trace_reference.cpp -o build/trace_reference
python scripts/run_java_comparison.py --graph .runtime/flywire_benchmark/graph.bin --out .runtime/java_comparison
```

The graph loader reads the existing little-endian RINGSCSR file. No dataset
subsampling, float32 conversion or change to the stimulus is applied.

## Direct Java run

```sh
java -Xms1g -Xmx1g -XX:-UseFMA -cp build/java RingsBenchmark --graph .runtime/flywire_benchmark/graph.bin --mode platform --threads 4 --steps 400 --warmup 600 --stimulus 720575940624963786 --output .runtime/java_state.bin
```

Select `--mode virtual` for the virtual-worker control. In the controlled
comparison the virtual carrier parallelism and maximum pool size are both
set to four. The default one-worker platform path executes directly without
a worker-coordinator barrier.

## Validation and timing protocol

- 138,639 nodes, 15,091,983 edges; the same signed normalized graph as C++.
- dt=0.001 s, carrier=10 Hz, gain=1, coupling=0.4, half-width=pi/2.
- Initial phase/activity zero; constant unit input to root ID 720575940624963786.
- A validation-only C++ trace contains all activity, phase and gate values
  after every one of 400 steps. Java platform 1/4 and virtual 4 compare every
  element bitwise. The temporary trace is about 943 MB and is deleted after
  successful validation; it is never part of timed measurements or artifacts.
- Three repetitions for C++ 1/4, Java platform 1/4 and Java virtual 4:
  15 measured processes, run serially and interleaved with seed 783.
- Every process performs 600 discarded warmup steps in a separate network,
  followed by a fresh state and 400 timed steps. This is a specified warmup,
  not proof that all JIT activity has stopped. JVM compiler and GC counters
  around the timed region are recorded.
- Both languages use the same first N allowed logical CPUs through taskset.
  OpenMP dynamic teams and per-thread binding are disabled; affinity is
  controlled at process level. JVM helper threads share that CPU budget.
- Timings exclude graph loading, initialization, warmup, validation, state
  export and summary scans. Final state hashes must match across all runs.
- Record median and range. Ratios use the C++ run from the same host and
  experiment, never the historical AMD numbers.

The full comparison requires four allowed logical CPUs, g++ with OpenMP,
JDK 21+, Python, NumPy/SciPy/pyarrow for data preparation, and about 2 GB of
free temporary disk space in addition to the dependencies. The C++ reference
mode does not require AVX-512. Do not invoke the older SIMD self-test on a
runner that lacks AVX-512.

[GitHub Actions workflow](../.github/workflows/java-benchmark.yml) runs this
comparison in one job and preserves compact results as an artifact. Results
must be taken from an observed successful run; source code alone is not a
performance measurement.
