// Validation-only trace of the unchanged C++ reference; never used for timings.
#define main rings_existing_benchmark_main
#include "benchmark_simd.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("Usage: trace_reference graph.bin trace.bin steps threads stimulus_id");
        const auto steps = parse_uint(argv[3]);
        const auto threads64 = parse_uint(argv[4]);
        if (!steps || !threads64 || threads64 > 1024) throw std::runtime_error("Invalid steps/threads");
        const int threads = static_cast<int>(threads64);
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        const Graph graph = load_graph(argv[1]);
        RingGraph net(graph);
        std::vector<double> input(graph.ids.size(), 0.0);
        const auto stimulus = parse_uint(argv[5]);
        const auto it = std::find(graph.ids.begin(), graph.ids.end(), stimulus);
        if (it == graph.ids.end()) throw std::runtime_error("Unknown stimulus ID");
        input[it - graph.ids.begin()] = 1.0;
        std::ofstream output(argv[2], std::ios::binary);
        if (!output) throw std::runtime_error("Cannot create trace");
        const std::uint64_t n = graph.ids.size();
        output.write("RINGSTRC", 8);
        output.write(reinterpret_cast<const char*>(&n), 8);
        output.write(reinterpret_cast<const char*>(&steps), 8);
        for (std::uint64_t step = 0; step < steps; ++step) {
            net.step(input);
            output.write(reinterpret_cast<const char*>(net.activity.data()), n * 8);
            output.write(reinterpret_cast<const char*>(net.phase.data()), n * 8);
            output.write(reinterpret_cast<const char*>(net.gate.data()), n);
            if (!output) throw std::runtime_error("Trace write failed");
        }
        output.close();
        if (!output) throw std::runtime_error("Trace close failed");
        std::cout << "{\"trace_steps\":" << steps << ",\"nodes\":" << n
                  << ",\"threads\":" << threads << ",\"bytes\":" << (24 + steps * n * 17)
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
