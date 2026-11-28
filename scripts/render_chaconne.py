#!/usr/bin/env python3
"""Build the Java MIDI renderer and render from the existing FlyWire graph (JDK 21+)."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / ".runtime/java_graph/graph.bin")
    p.add_argument("--coords", type=Path, default=ROOT / ".runtime/chaconne-coordinates.csv")
    p.add_argument("--midi", type=Path, default=ROOT / "music/bach-chaconne-bwv1004.mid")
    p.add_argument("--output", type=Path, default=ROOT / ".runtime/chaconne.wav")
    p.add_argument("--duration-ms", type=int, help="Render only the first N milliseconds")
    p.add_argument("--tail-ms", type=int, default=250, help="Extra simulation time after the MIDI ends")
    p.add_argument("--spread-cents", type=float, default=50)
    p.add_argument("--threshold", type=float, default=0.001)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--stimulus", default="720575940624963786")
    args = p.parse_args()
    graph, coords, midi, output = (x.resolve() for x in (args.graph, args.coords, args.midi, args.output))
    if not graph.is_file():
        p.error(f"Missing graph {graph}. Prepare it once using docs/JAVA_WINDOWS.md, or pass --graph.")
    if not midi.is_file():
        p.error(f"Missing MIDI file: {midi}")
    java, javac = shutil.which("java"), shutil.which("javac")
    if not java:
        p.error("Install JDK 21 or newer and put java on PATH.")
    if not coords.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_chaconne_coordinates.py"),
                        "--graph", str(graph), "--output", str(coords)], cwd=ROOT, check=True)
    classes = ROOT / "build/chaconne"
    classes.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "java/RingsBenchmark.java", *sorted((ROOT / "java").glob("Chaconne*.java"))]
    compiler = [javac, "--release", "21"] if javac else [java, "-m", "jdk.compiler/com.sun.tools.javac.Main"]
    subprocess.run([*compiler, "-encoding", "UTF-8", "-d", str(classes), *map(str, sources)], check=True)
    cmd = [java, "-Xmx2g", "-XX:-UseFMA", "-cp", str(classes), "Chaconne",
           "--graph", str(graph), "--coords", str(coords), "--midi", str(midi), "--output", str(output),
           "--threshold", str(args.threshold), "--spread-cents", str(args.spread_cents),
           "--stimulus", args.stimulus, "--threads", str(args.threads), "--tail-ms", str(args.tail_ms)]
    if args.duration_ms is not None:
        cmd.extend(["--duration-ms", str(args.duration_ms)])
    subprocess.run(cmd, check=True)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
