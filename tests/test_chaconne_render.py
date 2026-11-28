"""Portable JDK 21 launcher checks; the complete score uses only FOUR graph nodes.

No FlyWire data, network download, Python packages, or audio device is required.
All WAVs, coordinates and graph fixtures are temporary and removed after testing.
"""
from array import array
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
STIMULUS = 720575940624963786


def write_graph(path):
    """Independent little-endian RINGSCSR fixture, four nodes and four edges."""
    ids = [STIMULUS + i for i in range(4)]
    rows = [0, 0, 1, 2, 4]
    pre = [0, 0, 1, 2]
    weights = [1.0, 0.5, 0.5, 0.5]
    path.write_bytes(
        b"RINGSCSR" + struct.pack("<QQ", len(ids), len(pre))
        + struct.pack("<4Q", *ids) + struct.pack("<5Q", *rows)
        + struct.pack("<4I", *pre) + struct.pack("<4d", *weights)
    )


def vlq(value):
    result = [value & 127]
    while value > 127:
        value >>= 7
        result.insert(0, 128 | (value & 127))
    return bytes(result)


def write_chord(path):
    """Four simultaneous pitches, one second long; 1000 ticks/quarter at 60 BPM."""
    pitches = (55, 60, 64, 69)
    track = bytearray(b"\x00\xff\x51\x03\x0f\x42\x40")
    for pitch in pitches:
        track.extend(bytes((0, 0x90, pitch, 90)))
    for i, pitch in enumerate(pitches):
        track.extend(vlq(1000 if i == 0 else 0))
        track.extend(bytes((0x80, pitch, 0)))
    track.extend(b"\x00\xff\x2f\x00")
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 1000)
                     + b"MTrk" + struct.pack(">I", len(track)) + track)
    return pitches


def reject_nonfinite(value):
    raise ValueError("Nonfinite JSON metadata: " + value)


class ChaconneRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.java = shutil.which("java")
        if not cls.java:
            raise unittest.SkipTest("JDK 21 or later is required")
        cls.temp = tempfile.TemporaryDirectory(prefix="rings-chaconne-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.graph = cls.directory / "graph.bin"
        cls.coords = cls.directory / "coordinates.csv"
        write_graph(cls.graph)
        cls.coords.write_text("root_id,x_um,y_um,z_um\n" + "".join(
            f"{STIMULUS+i},{10*i},0,0\n" for i in range(4)), encoding="utf-8")
        cls.short_wav = cls.directory / "short.wav"
        cls.short_result = cls.launch(cls.short_wav, "--duration-ms", "1000")
        if cls.short_result.returncode:
            raise AssertionError("Normal Python launcher failed:\n" + cls.short_result.stdout
                                 + cls.short_result.stderr)
        cls.classes = ROOT / "build/chaconne"
        # Use the normal JDK compiler in Windows CI; the module fallback mirrors
        # the launcher for environments that expose the compiler without javac.
        javac = shutil.which("javac")
        compiler = [javac, "--release", "21"] if javac else [
            cls.java, "-m", "jdk.compiler/com.sun.tools.javac.Main"]
        result = subprocess.run([
            *compiler, "-encoding", "UTF-8", "-cp", str(cls.classes), "-d", str(cls.classes),
            str(ROOT / "tests/java/ChaconneScoreTest.java")], cwd=ROOT,
            capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError("Score test compilation failed:\n" + result.stdout + result.stderr)

    @classmethod
    def launch(cls, output, *extra):
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/render_chaconne.py"),
            "--graph", str(cls.graph), "--coords", str(cls.coords),
            "--output", str(output), "--stimulus", str(STIMULUS), "--threads", "1",
            *extra], cwd=cls.directory, capture_output=True, text=True, timeout=900)

    def metadata(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.lstrip().startswith("{")]
        self.assertEqual(len(lines), 1, result.stdout)
        data = json.loads(lines[0], parse_constant=reject_nonfinite)
        self.assertEqual(data["nodes"], 4)
        self.assertEqual(data["edges"], 4)
        self.assertEqual(data["located"], 4)
        self.assertGreater(data["audible_values"], 0)
        self.assertTrue(math.isfinite(data["distance_max_um"]))
        return data

    def assert_wav(self, path, simulated_ms):
        # The final simulation endpoint has an explicit extra 1 ms release.
        expected_frames = ((simulated_ms + 1) * 44100 + 999) // 1000
        lo, hi = 0, 0
        with wave.open(str(path), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()),
                             (1, 2, 44100))
            self.assertEqual(audio.getcomptype(), "NONE")
            self.assertEqual(audio.getnframes(), expected_frames)
            frames_read = 0
            while raw := audio.readframes(65536):
                values = array("h", raw)
                if sys.byteorder != "little":
                    values.byteswap()
                frames_read += len(values)
                lo, hi = min(lo, min(values)), max(hi, max(values))
            self.assertEqual(frames_read, expected_frames)
        # PCM16 cannot encode NaN/Inf; finite JSON and finite nonzero signed PCM
        # together verify that the renderer produced an actual bounded waveform.
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        self.assertLessEqual(max(-lo, hi), 31130)  # global 95% peak normalization
        self.assert_no_staging_files()

    def assert_no_staging_files(self):
        self.assertEqual(list(self.directory.glob("rings-audio-*")), [])
        self.assertEqual(list(self.directory.glob(".chaconne-audio-*")), [])

    def test_launcher_renders_truncated_checked_in_score(self):
        data = self.metadata(self.short_result)
        self.assertEqual(data["score_ms"], 770000)
        self.assertEqual(data["simulated_ms"], 1000)
        self.assertEqual(data["notes_in_score"], 3085)
        self.assert_wav(self.short_wav, 1000)

    def test_full_checked_in_score_on_four_node_graph(self):
        output = self.directory / "complete-score-small-graph.wav"
        data = self.metadata(self.launch(output))
        self.assertEqual(data["score_ms"], 770000)
        self.assertEqual(data["simulated_ms"], 770250)
        self.assertEqual(data["notes_in_score"], 3085)
        self.assertEqual((data["min_pitch"], data["max_pitch"]), (55, 91))
        self.assertEqual(data["rendered_pitches"], 36)
        self.assert_wav(output, 770250)

    def test_four_simultaneous_pitches_reach_the_audio(self):
        midi = self.directory / "four-note-chord.mid"
        pitches = write_chord(midi)
        output = self.directory / "four-note-chord.wav"
        data = self.metadata(self.launch(output, "--midi", str(midi), "--spread-cents", "0"))
        self.assertEqual(data["score_ms"], 1000)
        self.assertEqual(data["notes_in_score"], 4)
        self.assertEqual(data["rendered_pitches"], 4)
        self.assert_wav(output, 1250)
        with wave.open(str(output), "rb") as audio:
            samples = array("h", audio.readframes(22050))
        if sys.byteorder != "little":
            samples.byteswap()
        samples = [sample / 32768 for sample in samples[::4]]
        window = [0.5 - 0.5 * math.cos(2 * math.pi * i / (len(samples)-1))
                  for i in range(len(samples))]
        for pitch in pitches:
            frequency = 440 * 2 ** ((pitch - 69) / 12)
            real = sum(s*w*math.cos(2*math.pi*frequency*i/11025)
                       for i, (s, w) in enumerate(zip(samples, window)))
            imag = sum(s*w*math.sin(2*math.pi*frequency*i/11025)
                       for i, (s, w) in enumerate(zip(samples, window)))
            magnitude = 2 * math.hypot(real, imag) / sum(window)
            self.assertGreater(magnitude, 0.01, f"Missing audible chord pitch {pitch}")

    def test_missing_input_preserves_existing_output(self):
        output = self.directory / "preserved.wav"
        output.write_bytes(b"existing-output-must-stay")
        result = self.launch(output, "--midi", str(self.directory / "missing.mid"),
                             "--duration-ms", "10")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), b"existing-output-must-stay")
        self.assertFalse(any(line.lstrip().startswith("{") for line in result.stdout.splitlines()))
        self.assert_no_staging_files()

    def test_failed_final_move_removes_staged_files(self):
        output = self.directory / "nonempty-directory.wav"
        output.mkdir()
        (output / "keep.txt").write_text("keep", encoding="utf-8")
        result = self.launch(output, "--duration-ms", "20")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertFalse(any(line.lstrip().startswith("{") for line in result.stdout.splitlines()))
        self.assert_no_staging_files()

    def test_dedicated_java_unit_checks(self):
        for class_name, args in (
            ("ChaconneScoreTest", []),
            ("ChaconneAudio", ["--self-test"]),
            ("ChaconneSimulation", ["--self-test"]),
        ):
            with self.subTest(java_class=class_name):
                result = subprocess.run([
                    self.java, "-XX:-UseFMA", "-cp", str(self.classes), class_name, *args],
                    cwd=self.directory, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if class_name == "ChaconneSimulation":
                    data = json.loads(result.stdout, parse_constant=reject_nonfinite)
                    self.assertTrue(data["sparse_self_test"])
                    self.assertEqual(data["threads"], [1, 4])
                else:
                    self.assertIn("passed", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
