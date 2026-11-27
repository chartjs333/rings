#!/usr/bin/env python3
"""Same-host, sequential Java/C++ full-FlyWire comparison. Linux runner."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def command_json(command, env):
    start = time.perf_counter()
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError("Command failed: " + repr(command) + "\nstdout:\n" + run.stdout + "\nstderr:\n" + run.stderr)
    if run.stderr.strip():
        raise RuntimeError("Unexpected diagnostic: " + run.stderr)
    result = json.loads(run.stdout)
    result["command"] = command
    result["process_wall_seconds"] = time.perf_counter() - start
    return result

def read_text(path):
    p = Path(path)
    return p.read_text().strip() if p.exists() else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--cpp", type=Path, default=Path("build/benchmark_simd"))
    parser.add_argument("--trace-cpp", type=Path, default=Path("build/trace_reference"))
    parser.add_argument("--java-classpath", type=Path, default=Path("build/java"))
    parser.add_argument("--out", type=Path, default=Path(".runtime/java_comparison"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("Invalid benchmark counts")
    args.graph = args.graph.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "runs.json").exists():
        raise ValueError("Choose a fresh output directory; existing runs will not be overwritten")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        raise RuntimeError("Four allowed logical CPUs are required")
    if not shutil.which("taskset"):
        raise RuntimeError("taskset is required")
    with args.graph.open("rb") as handle:
        magic, n, m = struct.unpack("<8sQQ", handle.read(24))
    if magic != b"RINGSCSR" or (n, m) != (138639, 15091983):
        raise ValueError("Expected the full supplied FlyWire graph")
    stimulus = "720575940624963786"
    env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_PROC_BIND="FALSE")
    env.pop("OMP_PLACES", None)
    configurations = [("cpp", "reference", 1), ("cpp", "reference", 4),
                      ("java", "platform", 1), ("java", "platform", 4),
                      ("java", "virtual", 4)]

    def prefix(language, mode, threads):
        pin = ["taskset", "-c", ",".join(map(str, cpus[:threads]))]
        if language == "cpp":
            return pin + [str(args.cpp.resolve()), "--mode", "reference"]
        return pin + ["java", "-Xms1g", "-Xmx1g", "-XX:-UseFMA",
                      "-XX:ActiveProcessorCount=" + str(threads),
                      "-Djdk.virtualThreadScheduler.parallelism=" + str(threads),
                      "-Djdk.virtualThreadScheduler.maxPoolSize=" + str(threads),
                      "-cp", str(args.java_classpath.resolve()), "RingsBenchmark",
                      "--mode", mode]

    common = ["--graph", str(args.graph), "--stimulus", stimulus,
              "--steps", str(args.steps)]
    trace = args.out / "reference.trace"
    validation = {"cpp_trace": command_json(
        ["taskset", "-c", ",".join(map(str, cpus[:4])), str(args.trace_cpp.resolve()),
         str(args.graph), str(trace), str(args.steps), "4", stimulus], env),
        "java": []}
    for mode, threads in [("platform", 1), ("platform", 4), ("virtual", 4)]:
        record = command_json(prefix("java", mode, threads) + common +
                              ["--threads", str(threads), "--validate-against", str(trace)], env)
        record.update(mode=mode, threads=threads)
        validation["java"].append(record)
        (args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print("VALIDATION " + json.dumps(record), flush=True)
        if not record["passed"]:
            raise RuntimeError("Java/C++ per-step state mismatch")
    trace.unlink()  # only compact validation records are retained
    records = []
    rng = random.Random(783)
    for repeat in range(1, args.repeats + 1):
        order = configurations.copy()
        rng.shuffle(order)
        for language, mode, threads in order:
            state = args.out / f"{language}_{mode}_{threads}t_r{repeat}.bin"
            command = prefix(language, mode, threads) + common + [
                "--threads", str(threads), "--warmup", str(args.warmup),
                "--output", str(state)]
            result = command_json(command, env)
            if not result["finite"] or result["threads"] != threads:
                raise RuntimeError("Invalid state or worker count")
            if language == "cpp" and result["actual_threads"] != threads:
                raise RuntimeError("Unexpected OpenMP team count")
            result.update(language=language, mode=mode, threads=threads, repeat=repeat,
                          cpu_set=cpus[:threads], state_sha256=sha256(state))
            records.append(result)
            (args.out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
            print("TIMING " + json.dumps(result), flush=True)
    equal = len({r["state_sha256"] for r in records}) == 1
    summary = []
    for language, mode, threads in configurations:
        group = [r for r in records if (r["language"], r["mode"], r["threads"]) ==
                 (language, mode, threads)]
        seconds = [r["seconds"] for r in group]
        summary.append({"language": language, "mode": mode, "threads": threads,
                        "median_seconds": statistics.median(seconds),
                        "median_steps_per_second": args.steps / statistics.median(seconds),
                        "min_steps_per_second": args.steps / max(seconds),
                        "max_steps_per_second": args.steps / min(seconds)})
    cpp_speeds = {r["threads"]: r["median_steps_per_second"] for r in summary if r["language"] == "cpp"}
    for row in summary:
        row["speed_ratio_vs_cpp_same_threads"] = row["median_steps_per_second"] / cpp_speeds[row["threads"]]
    cpu_info = {}
    for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in ("model name", "flags"):
                cpu_info[key.strip()] = value.strip()
    java_version = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    sources = ["java/RingsBenchmark.java", "cpp/benchmark_simd.cpp", "cpp/trace_reference.cpp",
               "cpp/rings_graph.hpp", "original/rings_cifar10/rings.hpp",
               "scripts/prepare_flywire_benchmark.py", "scripts/run_java_comparison.py"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: sha256(ROOT / p) for p in sources},
        "source_dataset": "Full supplied FlyWire v783 table, verified against pinned hashes",
        "nodes": n, "edges": m, "graph_sha256": sha256(args.graph),
        "host": {"platform": platform.platform(), "cpu": cpu_info, "allowed_cpu_ids": cpus,
                 "cpu_max": read_text("/sys/fs/cgroup/cpu.max"),
                 "memory_max": read_text("/sys/fs/cgroup/memory.max"),
                 "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
                 "runner_os": os.environ.get("RUNNER_OS")},
        "java_version": (java_version.stdout + java_version.stderr).strip(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "protocol": {"steps": args.steps, "warmup_steps_separate_network": args.warmup,
                     "repeats": args.repeats, "fresh_process_each_repeat": True,
                     "process_affinity": "same first N allowed logical CPUs for both languages",
                     "openmp_binding": "process CPU set; OMP_PROC_BIND=FALSE",
                     "java_heap": "-Xms1g -Xmx1g", "java_disable_fma": True,
                     "java_active_processor_count": "requested worker count",
                     "virtual_carriers": "requested worker count",
                     "stimulus": stimulus, "input": "constant unit input",
                     "dt": .001, "hz": 10, "gain": 1, "coupling": .4, "width": "pi/2",
                     "initial_phase_activity": 0,
                     "timing_scope": "step loop only; excludes load, initialization, warmup, validation and output",
                     "ordering": "sequential; seed783 shuffled configurations in each repetition",
                     "cpp_flags": "-O3 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra"},
        "validation": validation,
        "all_final_state_hashes_equal": equal,
        "summary": summary, "runs": records,
        "limitations": [
            "Exploratory implementation comparison on one hosted runner, not a universal language ranking.",
            "600 warmup steps (or the selected value) is the protocol, not proof of asymptotic JVM steady state.",
            "Java parallel mode uses persistent workers plus a coordinating main thread; OpenMP runtime differs.",
            "Virtual mode tests four persistent compute workers, not one thread per neuron.",
            "The old AMD measurements are not the baseline for these ratios.",
            "Same CPU set is enforced per process; OS scheduling inside that set is not identical across runtimes.",
        ],
    }
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Java versus C++ on full FlyWire v783", "",
             f"Source commit: {report['source_commit']}", "",
             f"CPU: {cpu_info.get('model name', 'unknown')}", "",
             f"{n:,} nodes; {m:,} edges; float64; {args.steps} measured steps; "
             f"{args.warmup} warmup steps; {args.repeats} repetitions.", "",
             "| Runtime | Workers | Median steps/s | Range | Speed / C++ |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['language']} {row['mode']} | {row['threads']} | "
                     f"{row['median_steps_per_second']:.2f} | "
                     f"{row['min_steps_per_second']:.2f}–{row['max_steps_per_second']:.2f} | "
                     f"{row['speed_ratio_vs_cpp_same_threads']:.3f} |")
    lines += ["", f"All final state files identical: {equal}.",
              "Java states were compared bitwise against the C++ trace after every measured step.",
              "", "## Limitations", ""]
    lines += ["- " + value for value in report["limitations"]]
    lines += ["", "JVM compilation and garbage-collection counters are in each raw Java timing record.",
              "Full machine-readable results are in comparison.json.", ""]
    markdown = "\n".join(lines)
    (args.out / "report.md").write_text(markdown)
    print("COMPARISON_SUMMARY " + json.dumps({"summary": summary, "all_final_states_equal": equal}), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(markdown)
    if not equal:
        raise RuntimeError("Final state files differ")

if __name__ == "__main__":
    main()
