# Native RINGS graph prototype
Reuses original rings.hpp wrap() and gated_value() directly. The fixed four-neighbor RGB operator is replaced with signed, row-normalized CSR incoming connections. The rectification matches the earlier Python graph adaptation, not the original signed image output. C++17 simulation and CSV import; optional OpenMP. No Python runtime required after CSV export.
```sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python scripts/download_flywire.py --max-nodes 0 --output .runtime/full
OMP_NUM_THREADS=1 build/rings_fly_cpp .runtime/full/nodes.csv .runtime/full/connections.csv 400 720575940624963786
OMP_NUM_THREADS=4 build/rings_fly_cpp .runtime/full/nodes.csv .runtime/full/connections.csv 400 720575940624963786
```
Python is currently needed only for the upstream Parquet export. The CLI uses constant unit stimulation at explicitly supplied IDs. Timing excludes loading and includes stepping only. The full-source-graph C++/AVX2/AVX-512 benchmark is documented in [BENCHMARK_FLYWIRE.md](../docs/BENCHMARK_FLYWIRE.md), with [recorded measurements](../benchmarks/flywire_v783/report.md). Python performance parity is not claimed. Markov summaries and body integration remain future work. The graph is borrowed by the engine and must outlive it without mutation. This is a computational model, not validated physiology.
