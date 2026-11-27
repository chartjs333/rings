#!/usr/bin/env python3
"""Run sequential, interleaved real-graph timings and compare all state hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--stimulus', default='720575940624963786')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if len(allowed_cpus) < 4:
        raise RuntimeError('At least four allowed logical CPUs are required')
    places = ','.join('{'+str(cpu)+'}' for cpu in allowed_cpus)
    env = dict(os.environ, OMP_PROC_BIND='close', OMP_PLACES=places, OMP_DYNAMIC='FALSE')
    modes = ['reference', 'avx2', 'avx512']
    rng = random.Random(783)
    records = []
    for repeat in range(a.repeats):
        order = [(mode, threads) for mode in modes for threads in (1, 4)]
        rng.shuffle(order)
        for mode, threads in order:
            state = a.out / f'{mode}_{threads}t_repeat{repeat+1}.bin'
            command = [str(a.binary.resolve()), '--graph', str(a.graph.resolve()),
                       '--mode', mode, '--threads', str(threads), '--steps', str(a.steps),
                       '--warmup', str(a.warmup), '--stimulus', a.stimulus,
                       '--output', str(state.resolve())]
            result = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
            if result.stderr.strip():
                raise RuntimeError('Unexpected benchmark diagnostic: '+result.stderr)
            record = json.loads(result.stdout)
            if record['actual_threads'] != threads or record['thread_cpu_ids'] != allowed_cpus[:threads]:
                raise RuntimeError('Actual OpenMP team/CPU placement differs from the protocol')
            if not record['finite']:
                raise RuntimeError('Nonfinite benchmark state')
            record.update(repeat=repeat+1, state_sha256=digest(state), command=command,
                          stderr=result.stderr)
            records.append(record)
            (a.out / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(record), flush=True)
    summaries = []
    for mode in modes:
        for threads in (1, 4):
            group = [r for r in records if r['mode'] == mode and r['threads'] == threads]
            seconds = [r['seconds'] for r in group]
            summaries.append({'mode': mode, 'threads': threads,
                              'median_seconds': statistics.median(seconds),
                              'median_steps_per_second': a.steps/statistics.median(seconds),
                              'min_steps_per_second': a.steps/max(seconds),
                              'max_steps_per_second': a.steps/min(seconds)})
    cpu = {}
    for line in Path('/proc/cpuinfo').read_text().split('\n\n')[0].splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key.strip() in ('model name', 'flags'):
                cpu[key.strip()] = value.strip()
    sources = ['cpp/rings_graph.hpp', 'original/rings_cifar10/rings.hpp',
               'cpp/benchmark_simd.cpp', 'scripts/prepare_flywire_benchmark.py',
               'scripts/run_simd_benchmark.py']
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'source_sha256': {name: digest(root/name) for name in sources},
               'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
               'compile_flags': '-O3 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra',
               'executable_sha256': digest(a.binary), 'cpu': cpu,
               'platform': platform.platform(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'omp_environment': {k: env[k] for k in ('OMP_PROC_BIND', 'OMP_PLACES', 'OMP_DYNAMIC')},
               'protocol': {'timed_steps': a.steps, 'discarded_warmup_steps': a.warmup,
                            'reset_after_warmup': True, 'repeats': a.repeats,
                            'stimulus_id': a.stimulus, 'stimulus': 'constant unit input',
                            'initial_phase_and_activity': 0, 'dt': .001, 'hz': 10,
                            'gain': 1, 'coupling': .4, 'width': 'pi/2',
                            'timing': 'step loop only, no load, warmup, validation, or output',
                            'order': 'seed783 shuffled six configurations per repetition'},
               'graph_sha256': digest(a.graph),
               'all_18_final_state_hashes_equal': len({r['state_sha256'] for r in records}) == 1,
               'summary': summaries, 'runs': records}
    (a.out / 'benchmark_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'summary': summaries, 'all_final_states_equal':
                      summary['all_18_final_state_hashes_equal']}, indent=2), flush=True)
    if not summary['all_18_final_state_hashes_equal']:
        raise RuntimeError('Final state mismatch between configurations')


if __name__ == '__main__':
    main()
