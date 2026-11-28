import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MetaMessage;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiFileFormat;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

/**
 * Portable MIDI-to-note-gate reader. Supports format 0/1 PPQ files, tempo maps,
 * channels, overlapping voices, and sustain. It does not synthesize audio.
 *
 * Every note uses [startMillis, endMillis): the gate is closed at its end.
 * Event times are rounded to the nearest millisecond, with exact half ties up.
 * Notes that collapse to zero milliseconds are retained in notes(), reported,
 * and do not open a gate. Absolute times and rests are never compressed.
 */
public final class ChaconneScore {
    private ChaconneScore() {}

    public record NoteInterval(int pitch, int channel, int velocity, int track,
                               long startTick, long endTick,
                               long startMillis, long endMillis) {}

    public record TempoChange(long tick, int microsPerQuarter, int track) {}

    public record Score(int minPitch, int maxPitch, long durationMillis,
                        List<NoteInterval> notes, Map<String, Object> metadata,
                        List<String> warnings) {
        public Score {
            notes = List.copyOf(notes);
            metadata = Map.copyOf(metadata);
            warnings = List.copyOf(warnings);
        }

        public int pitchCount() { return maxPitch - minPitch + 1; }

        /** A separate cursor per playback; its pitch index is pitch - minPitch. */
        public Cursor cursor() { return new Cursor(this); }
    }

    private record Event(long tick, int track, int index, MidiMessage message) {}
    private record Start(int pitch, int channel, int velocity, int track,
                         long tick, BigInteger timeNumerator) {}
    private record GateEvent(long millis, int pitchIndex, int delta) {}

    /** Keeps note counts, so one ending voice cannot close another's gate. */
    public static final class Cursor {
        private final List<GateEvent> events;
        private final int[] counts;
        private int next;
        private long previousMillis = -1;

        private Cursor(Score score) {
            counts = new int[score.pitchCount()];
            var collected = new ArrayList<GateEvent>(score.notes().size() * 2);
            for (NoteInterval note : score.notes()) {
                if (note.startMillis() == note.endMillis()) continue;
                int pitch = note.pitch() - score.minPitch();
                collected.add(new GateEvent(note.startMillis(), pitch, 1));
                collected.add(new GateEvent(note.endMillis(), pitch, -1));
            }
            collected.sort(Comparator.comparingLong(GateEvent::millis)
                    .thenComparingInt(GateEvent::delta)
                    .thenComparingInt(GateEvent::pitchIndex));
            events = List.copyOf(collected);
        }

        /** Fill all semitone gates at a nondecreasing absolute score time. */
        public void fill(long millis, boolean[] gates) {
            if (millis < 0 || millis < previousMillis)
                throw new IllegalArgumentException("Cursor time must be nonnegative and nondecreasing");
            if (gates.length != counts.length)
                throw new IllegalArgumentException("Wrong gate array length");
            while (next < events.size() && events.get(next).millis() <= millis) {
                GateEvent event = events.get(next++);
                counts[event.pitchIndex()] += event.delta();
                if (counts[event.pitchIndex()] < 0)
                    throw new IllegalStateException("Unbalanced note intervals");
            }
            for (int i = 0; i < gates.length; i++) gates[i] = counts[i] > 0;
            previousMillis = millis;
        }
    }

    public static Score read(Path path) throws IOException {
        try {
            MidiFileFormat format = MidiSystem.getMidiFileFormat(path.toFile());
            if (format.getType() != 0 && format.getType() != 1)
                throw new IOException("Only MIDI formats 0 and 1 are supported; format 2 has independent sequences");
            Sequence sequence = MidiSystem.getSequence(path.toFile());
            if (sequence.getDivisionType() != Sequence.PPQ)
                throw new IOException("SMPTE MIDI timing is unsupported; use a PPQ MIDI score");
            if (sequence.getResolution() <= 0)
                throw new IOException("Invalid PPQ resolution");
            return new Reader(sequence, format.getType()).read();
        } catch (InvalidMidiDataException exception) {
            throw new IOException("Invalid MIDI score: " + exception.getMessage(), exception);
        } catch (ArithmeticException exception) {
            throw new IOException("MIDI timing exceeds supported millisecond range", exception);
        }
    }

    private static final class Reader {
        private final Sequence sequence;
        private final int midiType;
        // Time numerator / PPQ is microseconds. Integer arithmetic avoids drift.
        private final BigInteger ppq;
        private final BigInteger millisDenominator;
        private final Map<Integer, ArrayDeque<Start>> held = new HashMap<>();
        private final List<List<Start>> sustained = new ArrayList<>(16);
        private final boolean[] pedal = new boolean[16];
        private final List<NoteInterval> notes = new ArrayList<>();
        private final List<TempoChange> tempoChanges = new ArrayList<>();
        private final List<String> warnings = new ArrayList<>();
        private final TreeSet<Integer> channels = new TreeSet<>();
        private final List<String> trackNames = new ArrayList<>();
        private int minPitch = 128, maxPitch = -1;
        private int noteOns, velocityZeroOffs, unmatchedOffs, closedAtEnd;
        private int sameKeyOverlaps, sustainEvents, pitchBends, unsupportedHolds, sysexEvents;
        private int collapsed, conflictingTempos;
        private boolean sustainUsed;
        private long lastTempoTick = -1;
        private int lastTempo = -1;

        Reader(Sequence sequence, int midiType) {
            this.sequence = sequence;
            this.midiType = midiType;
            ppq = BigInteger.valueOf(sequence.getResolution());
            millisDenominator = ppq.multiply(BigInteger.valueOf(1000));
            for (int channel = 0; channel < 16; channel++) sustained.add(new ArrayList<>());
        }

        Score read() throws IOException {
            List<Event> events = new ArrayList<>();
            Track[] tracks = sequence.getTracks();
            for (int track = 0; track < tracks.length; track++) {
                for (int index = 0; index < tracks[track].size(); index++) {
                    MidiEvent event = tracks[track].get(index);
                    if (event.getTick() < 0) throw new IOException("Negative MIDI tick");
                    events.add(new Event(event.getTick(), track, index, event.getMessage()));
                }
            }
            events.sort(Comparator.comparingLong(Event::tick)
                    .thenComparingInt(Event::track).thenComparingInt(Event::index));
            long tick = 0;
            int tempo = 500_000; // MIDI default: 120 quarter notes per minute.
            BigInteger time = BigInteger.ZERO;
            for (Event event : events) {
                time = time.add(BigInteger.valueOf(event.tick() - tick)
                        .multiply(BigInteger.valueOf(tempo)));
                tick = event.tick();
                MidiMessage message = event.message();
                if (message instanceof MetaMessage meta) {
                    if (meta.getType() == 0x51) {
                        byte[] bytes = meta.getData();
                        if (bytes.length != 3) throw new IOException("Invalid Set Tempo meta event");
                        int nextTempo = ((bytes[0] & 255) << 16)
                                | ((bytes[1] & 255) << 8) | (bytes[2] & 255);
                        if (nextTempo == 0) throw new IOException("MIDI tempo must be positive");
                        if (lastTempoTick == tick && lastTempo != nextTempo) conflictingTempos++;
                        lastTempoTick = tick;
                        lastTempo = nextTempo;
                        tempo = nextTempo;
                        tempoChanges.add(new TempoChange(tick, tempo, event.track()));
                    } else if (meta.getType() == 0x03) {
                        trackNames.add("track " + event.track() + ": "
                                + new String(meta.getData(), java.nio.charset.StandardCharsets.UTF_8));
                    }
                } else if (message instanceof ShortMessage shortMessage) {
                    process(shortMessage, event.track(), tick, time);
                } else if (message instanceof javax.sound.midi.SysexMessage) {
                    sysexEvents++;
                }
            }
            long endTick = sequence.getTickLength();
            time = time.add(BigInteger.valueOf(endTick - tick).multiply(BigInteger.valueOf(tempo)));
            for (int channel = 0; channel < 16; channel++) {
                for (int pitch = 0; pitch < 128; pitch++) {
                    ArrayDeque<Start> voices = held.get(key(channel, pitch));
                    if (voices == null) continue;
                    while (!voices.isEmpty()) { finish(voices.removeFirst(), endTick, time); closedAtEnd++; }
                }
                for (Start start : sustained.get(channel)) { finish(start, endTick, time); closedAtEnd++; }
                sustained.get(channel).clear();
            }
            if (noteOns == 0) throw new IOException("MIDI score contains no positive-velocity Note On events");
            notes.sort(Comparator.comparingLong(NoteInterval::startMillis)
                    .thenComparingLong(NoteInterval::startTick)
                    .thenComparingInt(NoteInterval::pitch)
                    .thenComparingInt(NoteInterval::channel)
                    .thenComparingInt(NoteInterval::track)
                    .thenComparingLong(NoteInterval::endTick));
            if (sameKeyOverlaps > 0)
                warnings.add("Overlapping Note On events for one channel/pitch use FIFO Note Off pairing: " + sameKeyOverlaps);
            if (unmatchedOffs > 0) warnings.add("Unmatched Note Off events ignored: " + unmatchedOffs);
            if (closedAtEnd > 0) warnings.add("Voices still sounding at sequence end were closed there: " + closedAtEnd);
            if (collapsed > 0) warnings.add("Notes collapsed to zero milliseconds by quantization: " + collapsed);
            if (conflictingTempos > 0)
                warnings.add("Conflicting same-tick tempo events: last in track/event order wins: " + conflictingTempos);
            if (unsupportedHolds > 0)
                warnings.add("Sostenuto CC66 and Hold2 CC69 are unsupported and ignored: " + unsupportedHolds);
            if (pitchBends > 0)
                warnings.add("Pitch bend ignored; gates represent written MIDI note numbers: " + pitchBends);
            if (sysexEvents > 0)
                warnings.add("System Exclusive messages ignored, including any device-specific resets/tuning: " + sysexEvents);
            if (channels.contains(9)) warnings.add("MIDI channel 10 is present; its note numbers are retained without interpreting percussion sounds");
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("midiType", midiType);
            metadata.put("division", "PPQ");
            metadata.put("ppq", sequence.getResolution());
            metadata.put("trackCount", tracks.length);
            metadata.put("durationTicks", endTick);
            metadata.put("noteOnCount", noteOns);
            metadata.put("noteIntervalCount", notes.size());
            metadata.put("velocityZeroNoteOffCount", velocityZeroOffs);
            metadata.put("tempoEventCount", tempoChanges.size());
            metadata.put("tempoChanges", List.copyOf(tempoChanges));
            metadata.put("initialDefaultMicrosPerQuarter", 500_000);
            metadata.put("sustainPedalSupported", true);
            metadata.put("sustainPedalUsed", sustainUsed);
            metadata.put("sustainEventCount", sustainEvents);
            metadata.put("unmatchedNoteOffCount", unmatchedOffs);
            metadata.put("notesClosedAtSequenceEnd", closedAtEnd);
            metadata.put("sameChannelPitchOverlapCount", sameKeyOverlaps);
            metadata.put("collapsedNoteCount", collapsed);
            metadata.put("pitchBendEventsIgnored", pitchBends);
            metadata.put("unsupportedHoldEventsIgnored", unsupportedHolds);
            metadata.put("sysexEventsIgnored", sysexEvents);
            metadata.put("channelsZeroBased", List.copyOf(channels));
            metadata.put("trackNames", List.copyOf(trackNames));
            metadata.put("quantization", "nearest millisecond, exact half ties up; intervals [start,end); duration ceil");
            metadata.put("sameTickOrder", "tick, track index, original event index");
            return new Score(minPitch, maxPitch, ceilMillis(time), notes, metadata, warnings);
        }

        private void process(ShortMessage message, int track, long tick, BigInteger time) {
            int channel = message.getChannel(), pitch = message.getData1(), value = message.getData2();
            switch (message.getCommand()) {
                case ShortMessage.NOTE_ON -> {
                    if (value == 0) { velocityZeroOffs++; noteOff(channel, pitch, tick, time); }
                    else {
                        channels.add(channel);
                        minPitch = Math.min(minPitch, pitch); maxPitch = Math.max(maxPitch, pitch);
                        noteOns++;
                        ArrayDeque<Start> voices = held.computeIfAbsent(key(channel, pitch), ignored -> new ArrayDeque<>());
                        if (!voices.isEmpty()) sameKeyOverlaps++;
                        voices.addLast(new Start(pitch, channel, value, track, tick, time));
                    }
                }
                case ShortMessage.NOTE_OFF -> noteOff(channel, pitch, tick, time);
                case ShortMessage.PITCH_BEND -> pitchBends++;
                case ShortMessage.CONTROL_CHANGE -> {
                    switch (pitch) {
                        case 64 -> {
                            sustainEvents++;
                            boolean down = value >= 64;
                            sustainUsed |= down;
                            if (pedal[channel] && !down) releaseSustained(channel, tick, time);
                            pedal[channel] = down;
                        }
                        case 120 -> allSoundOff(channel, tick, time);
                        case 121 -> { pedal[channel] = false; releaseSustained(channel, tick, time); }
                        case 123, 124, 125, 126, 127 -> allNotesOff(channel, tick, time);
                        case 66, 69 -> unsupportedHolds++;
                        default -> { /* Other controllers do not change note-number gates. */ }
                    }
                }
                default -> { /* Program, pressure and other messages do not alter note gates. */ }
            }
        }

        private void noteOff(int channel, int pitch, long tick, BigInteger time) {
            ArrayDeque<Start> voices = held.get(key(channel, pitch));
            if (voices == null || voices.isEmpty()) { unmatchedOffs++; return; }
            Start start = voices.removeFirst();
            if (pedal[channel]) sustained.get(channel).add(start);
            else finish(start, tick, time);
        }

        private void allNotesOff(int channel, long tick, BigInteger time) {
            for (int pitch = 0; pitch < 128; pitch++) {
                ArrayDeque<Start> voices = held.get(key(channel, pitch));
                if (voices == null) continue;
                while (!voices.isEmpty()) noteOff(channel, pitch, tick, time);
            }
        }

        private void allSoundOff(int channel, long tick, BigInteger time) {
            for (int pitch = 0; pitch < 128; pitch++) {
                ArrayDeque<Start> voices = held.get(key(channel, pitch));
                if (voices == null) continue;
                while (!voices.isEmpty()) finish(voices.removeFirst(), tick, time);
            }
            releaseSustained(channel, tick, time);
        }

        private void releaseSustained(int channel, long tick, BigInteger time) {
            for (Start start : sustained.get(channel)) finish(start, tick, time);
            sustained.get(channel).clear();
        }

        private void finish(Start start, long endTick, BigInteger time) {
            long begin = nearestMillis(start.timeNumerator()), end = nearestMillis(time);
            if (begin == end) collapsed++;
            notes.add(new NoteInterval(start.pitch(), start.channel(), start.velocity(), start.track(),
                    start.tick(), endTick, begin, end));
        }

        private long nearestMillis(BigInteger numerator) {
            return numerator.shiftLeft(1).add(millisDenominator)
                    .divide(millisDenominator.shiftLeft(1)).longValueExact();
        }

        private long ceilMillis(BigInteger numerator) {
            return numerator.add(millisDenominator).subtract(BigInteger.ONE)
                    .divide(millisDenominator).longValueExact();
        }

        private static int key(int channel, int pitch) { return channel * 128 + pitch; }
    }
}
