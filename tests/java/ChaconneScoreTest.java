import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import javax.sound.midi.MetaMessage;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

/** Synthetic MIDI fixtures: no external score or audio device is needed. */
public final class ChaconneScoreTest {
    private static Path directory;
    private static int fixture;

    public static void main(String[] args) throws Exception {
        Files.createDirectories(Path.of("build"));
        directory = Files.createTempDirectory(Path.of("build"), "midi-score-test-");
        try {
            tempoMapAndRests();
            overlappingChannelsAndVelocityZero();
            sameKeyOverlap();
            sustainAndRetrigger();
            channelModeControllers();
            quantization();
            trailingHeldNote();
            unsupportedHoldWarning();
            unsupportedTimingAndEmpty();
            System.out.println("ChaconneScoreTest: 9 synthetic MIDI checks passed");
        } finally {
            try (var paths = Files.walk(directory)) {
                for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
            }
        }
    }

    private static void tempoMapAndRests() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 480);
        Track conductor = sequence.createTrack(), notes = sequence.createTrack();
        tempo(conductor, 0, 500_000); tempo(conductor, 480, 1_000_000);
        note(notes, 240, 0, 60, 90, true); note(notes, 960, 0, 60, 0, false);
        note(notes, 1080, 0, 70, 90, true); note(notes, 1200, 0, 70, 0, false);
        end(conductor, 1440);
        var score = read(sequence);
        check(score.minPitch() == 60 && score.maxPitch() == 70 && score.pitchCount() == 11, "chromatic range");
        check(score.durationMillis() == 2500, "tempo-aware duration includes trailing rest");
        check(score.notes().get(0).startMillis() == 250 && score.notes().get(0).endMillis() == 1500,
                "note spanning tempo change");
        check(score.notes().get(1).startMillis() == 1750 && score.notes().get(1).endMillis() == 2000,
                "second tempo segment");
        var cursor = score.cursor(); boolean[] gates = new boolean[11];
        cursor.fill(249, gates); check(none(gates), "leading rest");
        cursor.fill(250, gates); check(gates[0] && !gates[10], "first note on");
        cursor.fill(1499, gates); check(gates[0], "first note before end");
        cursor.fill(1500, gates); check(none(gates), "exclusive end");
        cursor.fill(1749, gates); check(none(gates), "internal rest");
        cursor.fill(1750, gates); check(gates[10] && !gates[0], "second independent gate");
        cursor.fill(2000, gates); check(none(gates), "trailing rest begins");
        cursor.fill(2500, gates); check(none(gates), "end has no held gate");
        boolean rejected = false;
        try { cursor.fill(2499, gates); } catch (IllegalArgumentException expected) { rejected = true; }
        check(rejected, "backward cursor rejected");
    }

    private static void overlappingChannelsAndVelocityZero() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000);
        Track a = sequence.createTrack(), b = sequence.createTrack();
        note(a, 0, 0, 60, 80, true); note(a, 4, 0, 60, 0, false);
        note(b, 2, 1, 60, 70, true); note(b, 8, 1, 60, 0, true); end(a, 10);
        var score = read(sequence); var cursor = score.cursor(); boolean[] gates = new boolean[1];
        check(score.notes().size() == 2, "two channel voices");
        check(score.metadata().get("velocityZeroNoteOffCount").equals(1), "velocity zero is off");
        cursor.fill(0, gates); check(gates[0], "first channel starts");
        cursor.fill(2, gates); check(gates[0], "ending one channel does not kill another");
        cursor.fill(4, gates); check(!gates[0], "all channels released");
    }

    private static void sameKeyOverlap() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000); Track track = sequence.createTrack();
        note(track, 0, 0, 60, 30, true); note(track, 2, 0, 60, 70, true);
        note(track, 4, 0, 60, 0, false); note(track, 6, 0, 60, 0, false);
        var score = read(sequence);
        check(score.notes().get(0).velocity() == 30 && score.notes().get(0).endMillis() == 2, "FIFO first voice");
        check(score.notes().get(1).velocity() == 70 && score.notes().get(1).endMillis() == 3, "FIFO second voice");
        var cursor = score.cursor(); boolean[] gates = new boolean[1];
        cursor.fill(2, gates); check(gates[0], "overlap reference count");
        cursor.fill(3, gates); check(!gates[0], "overlap ends");
        check(score.metadata().get("sameChannelPitchOverlapCount").equals(1), "overlap pairing disclosed");
    }

    private static void sustainAndRetrigger() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000); Track track = sequence.createTrack();
        note(track, 0, 0, 60, 90, true); note(track, 0, 1, 64, 90, true);
        control(track, 1, 0, 64, 127);
        note(track, 4, 0, 60, 0, false); note(track, 4, 1, 64, 0, false);
        note(track, 6, 0, 60, 80, true); note(track, 8, 0, 60, 0, false);
        control(track, 10, 0, 64, 0); end(track, 12);
        var score = read(sequence);
        check(score.notes().stream().filter(n -> n.pitch() == 60).allMatch(n -> n.endMillis() == 5), "sustain and retrigger release together");
        check(score.notes().stream().filter(n -> n.pitch() == 64).allMatch(n -> n.endMillis() == 2), "sustain isolated by channel");
        var cursor = score.cursor(); boolean[] gates = new boolean[5];
        cursor.fill(2, gates); check(gates[0] && !gates[4], "sustained only first channel");
        cursor.fill(4, gates); check(gates[0], "both key releases remain sustained");
        cursor.fill(5, gates); check(none(gates), "pedal up");
        check(score.metadata().get("sustainPedalUsed").equals(true), "sustain metadata");
    }

    private static void channelModeControllers() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000); Track track = sequence.createTrack();
        note(track, 0, 0, 60, 80, true); control(track, 1, 0, 64, 127);
        note(track, 2, 0, 60, 0, false); control(track, 4, 0, 121, 0);
        note(track, 6, 0, 62, 80, true); control(track, 7, 0, 64, 127);
        control(track, 8, 0, 123, 0); control(track, 10, 0, 64, 0);
        note(track, 12, 0, 64, 80, true); control(track, 13, 0, 64, 127);
        note(track, 14, 0, 64, 0, false); control(track, 16, 0, 120, 0);
        var score = read(sequence);
        check(score.notes().get(0).endMillis() == 2, "reset controllers releases pedal-held note");
        check(score.notes().get(1).endMillis() == 5, "all notes off respects sustain");
        check(score.notes().get(2).endMillis() == 8, "all sound off overrides sustain");
    }

    private static void quantization() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000); Track track = sequence.createTrack();
        note(track, 1, 0, 60, 80, true); note(track, 2, 0, 60, 0, false);
        note(track, 3, 0, 64, 80, true); note(track, 5, 0, 64, 0, false); end(track, 7);
        var score = read(sequence);
        check(score.notes().get(0).startMillis() == 1 && score.notes().get(0).endMillis() == 1, "half millisecond rounds up exactly");
        check(score.notes().get(1).startMillis() == 2 && score.notes().get(1).endMillis() == 3, "second half ties up");
        check(score.durationMillis() == 4 && score.metadata().get("collapsedNoteCount").equals(1), "ceil duration and disclosed collapsed note");
        var cursor = score.cursor(); boolean[] gates = new boolean[5];
        cursor.fill(1, gates); check(none(gates), "zero-duration note is not a stuck gate");
        cursor.fill(2, gates); check(gates[4] && !gates[0], "quantized gate");
        cursor.fill(3, gates); check(none(gates), "quantized gate ends");
    }

    private static void trailingHeldNote() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000); Track track = sequence.createTrack();
        note(track, 0, 0, 60, 80, true); end(track, 10);
        var score = read(sequence);
        check(score.notes().get(0).endMillis() == 5, "dangling note clipped to sequence end");
        check(score.metadata().get("notesClosedAtSequenceEnd").equals(1), "dangling note disclosed");
    }

    private static void unsupportedHoldWarning() throws Exception {
        Sequence sequence = new Sequence(Sequence.PPQ, 1000); Track track = sequence.createTrack();
        note(track, 0, 0, 60, 80, true); control(track, 1, 0, 66, 127);
        note(track, 4, 0, 60, 0, false);
        var score = read(sequence);
        check(score.metadata().get("unsupportedHoldEventsIgnored").equals(1), "unsupported sostenuto is disclosed");
        check(score.warnings().stream().anyMatch(s -> s.contains("Sostenuto")), "sostenuto warning present");
    }

    private static void unsupportedTimingAndEmpty() throws Exception {
        Sequence smpte = new Sequence(Sequence.SMPTE_24, 10); Track track = smpte.createTrack();
        note(track, 0, 0, 60, 80, true); note(track, 1, 0, 60, 0, false);
        boolean rejected = false;
        try { read(smpte); } catch (IOException expected) { rejected = expected.getMessage().contains("SMPTE"); }
        check(rejected, "SMPTE explicitly rejected");
        Sequence empty = new Sequence(Sequence.PPQ, 480); end(empty.createTrack(), 480);
        rejected = false;
        try { read(empty); } catch (IOException expected) { rejected = expected.getMessage().contains("no positive-velocity"); }
        check(rejected, "empty score rejected");
    }

    private static ChaconneScore.Score read(Sequence sequence) throws Exception {
        Path path = directory.resolve("fixture-" + fixture++ + ".mid");
        check(MidiSystem.write(sequence, 1, path.toFile()) > 0, "MIDI fixture written");
        return ChaconneScore.read(path);
    }

    private static void note(Track track, long tick, int channel, int pitch, int velocity, boolean on) throws Exception {
        ShortMessage message = new ShortMessage();
        message.setMessage(on ? ShortMessage.NOTE_ON : ShortMessage.NOTE_OFF, channel, pitch, velocity);
        track.add(new MidiEvent(message, tick));
    }

    private static void control(Track track, long tick, int channel, int controller, int value) throws Exception {
        ShortMessage message = new ShortMessage();
        message.setMessage(ShortMessage.CONTROL_CHANGE, channel, controller, value);
        track.add(new MidiEvent(message, tick));
    }

    private static void tempo(Track track, long tick, int micros) throws Exception {
        MetaMessage message = new MetaMessage();
        byte[] data = {(byte) (micros >>> 16), (byte) (micros >>> 8), (byte) micros};
        message.setMessage(0x51, data, 3); track.add(new MidiEvent(message, tick));
    }

    private static void end(Track track, long tick) throws Exception {
        MetaMessage message = new MetaMessage(); message.setMessage(0x2f, new byte[0], 0);
        track.add(new MidiEvent(message, tick));
    }

    private static boolean none(boolean[] gates) {
        for (boolean gate : gates) if (gate) return false;
        return true;
    }

    private static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }
}
