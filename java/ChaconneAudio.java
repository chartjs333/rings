import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/** Offline, bounded-memory activity sonification; independent of the graph engine.
 *
 * Input files are pitch-MIDI.bin, one file per MIDI pitch. All integers/doubles
 * use Java DataOutputStream big endian. A record is int step, int count, followed
 * by count (int cellIndex, double activity) pairs. Steps are strictly increasing
 * 1-based END-of-millisecond timestamps; omitted steps/cells mean zero activity.
 * EOF ends a track. Missing pitch files are rests. There is no file header.
 *
 * Distances are exact, not bucketed. Each (pitch, cell) has a fixed oscillator
 * pair f * 2^(+/-spreadCents * distance / maxDistance / 1200), mixed equally.
 * Zero distance, zero spread or zero global maximum uses one exact base tone.
 * Every oscillator follows absolute audio time, including silent intervals.
 * Activity envelopes are linearly interpolated between millisecond endpoints.
 * An explicit final 1 ms interpolation to silence extends the output by 1 ms.
 *
 * The sum is accumulated in float64, spooled as float32 PCM, then globally peak
 * normalized to 95% of signed PCM16 full scale. This is one common gain for the
 * complete mixture, not independent normalization of neurons, pitches or frames.
 * No dynamic limiter/compressor is applied. A pair is omitted if either of its
 * frequencies is at or above Nyquist. At 44.1 kHz all MIDI pitches 0..127 with
 * the supported maximum spread of 50 cents fit below Nyquist.
 */
public final class ChaconneAudio {
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final double PCM_TARGET = 0.95 * 32767.0;
    private static final int BUFFER_BYTES = 1 << 20;

    private ChaconneAudio() {}

    public record RenderStats(long samples, int sampleRate, double seconds,
                              double rawPeak, double pcmGain,
                              long distinctVoices, long omittedNyquistVoices) {}

    public static void render(Path trackDir, Path output, int durationMs,
            int minPitch, int maxPitch, double[] distanceByCellUm,
            double maxDistanceUm, double spreadCents, int sampleRate) throws IOException {
        renderWithStats(trackDir, output, durationMs, minPitch, maxPitch,
                distanceByCellUm, maxDistanceUm, spreadCents, sampleRate);
    }

    /** Same rendering operation, with measurements for the caller's report. */
    public static RenderStats renderWithStats(Path trackDir, Path output, int durationMs,
            int minPitch, int maxPitch, double[] distanceByCellUm,
            double maxDistanceUm, double spreadCents, int sampleRate) throws IOException {
        if (!Files.isDirectory(trackDir)) throw new IOException("Missing audio track directory: " + trackDir);
        if (durationMs < 0 || minPitch < 0 || maxPitch > 127 || minPitch > maxPitch)
            throw new IllegalArgumentException("Invalid duration or MIDI pitch range");
        if (sampleRate < 1000 || sampleRate > 384000)
            throw new IllegalArgumentException("Sample rate must be 1000..384000 Hz");
        if (distanceByCellUm == null || !Double.isFinite(maxDistanceUm) || maxDistanceUm < 0
                || !Double.isFinite(spreadCents) || spreadCents < 0 || spreadCents > 50)
            throw new IllegalArgumentException("Invalid distances or spread (allowed: 0..50 cents)");
        final long totalSamples = sampleBoundary((long) durationMs + 1, sampleRate);
        if (totalSamples > (0xffffffffL - 36L) / 2L)
            throw new IOException("Audio exceeds classic RIFF/WAV size limit");
        Path absoluteOutput = output.toAbsolutePath();
        Path parent = absoluteOutput.getParent();
        Files.createDirectories(parent);
        Path spool = Files.createTempFile(parent, ".chaconne-audio-", ".f32");
        Path wav = Files.createTempFile(parent, ".chaconne-audio-", ".wav");
        List<Track> tracks = new ArrayList<>();
        try {
            for (int pitch = minPitch; pitch <= maxPitch; pitch++) {
                Path file = trackDir.resolve("pitch-" + pitch + ".bin");
                if (Files.exists(file)) tracks.add(new Track(file, pitch, durationMs,
                        distanceByCellUm, maxDistanceUm, spreadCents, sampleRate));
            }
            double peak = synthesize(tracks, spool, durationMs, sampleRate);
            double gain = peak == 0 ? 0 : PCM_TARGET / peak;
            writeWav(spool, wav, totalSamples, sampleRate, gain);
            try {
                Files.move(wav, absoluteOutput, StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException ex) {
                Files.move(wav, absoluteOutput, StandardCopyOption.REPLACE_EXISTING);
            }
            long voices = 0, omitted = 0;
            for (Track track : tracks) {
                voices += track.voices.size();
                omitted += track.omitted;
            }
            return new RenderStats(totalSamples, sampleRate, (double) totalSamples / sampleRate,
                    peak, gain, voices, omitted);
        } finally {
            IOException failure = null;
            for (Track track : tracks) {
                try { track.close(); } catch (IOException ex) { failure = ex; }
            }
            try { Files.deleteIfExists(spool); } catch (IOException ex) { failure = ex; }
            try { Files.deleteIfExists(wav); } catch (IOException ex) { failure = ex; }
            if (failure != null) throw failure;
        }
    }

    private static long sampleBoundary(long millisecond, int sampleRate) {
        return (millisecond * sampleRate + 999L) / 1000L;
    }

    private static double synthesize(List<Track> tracks, Path spool, int durationMs,
                                     int sampleRate) throws IOException {
        double[] mix = new double[(sampleRate + 999) / 1000 + 1];
        double peak = 0;
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(
                Files.newOutputStream(spool), BUFFER_BYTES))) {
            for (long tick = 0; tick <= durationMs; tick++) {
                long start = sampleBoundary(tick, sampleRate);
                int count = (int) (sampleBoundary(tick + 1, sampleRate) - start);
                java.util.Arrays.fill(mix, 0, count, 0.0);
                double fractionAtStart = (start * 1000L - tick * sampleRate) / (double) sampleRate;
                for (Track track : tracks) {
                    track.endpoint(tick + 1);
                    track.mix(mix, start, count, fractionAtStart, sampleRate);
                }
                for (int j = 0; j < count; j++) {
                    float sample = (float) mix[j];
                    if (!Float.isFinite(sample)) throw new IOException("Nonfinite audio mixture");
                    peak = Math.max(peak, Math.abs((double) sample));
                    out.writeFloat(sample);
                }
            }
        }
        return peak;
    }

    private static final class Track implements AutoCloseable {
        final DataInputStream in;
        final int pitch, durationMs, sampleRate;
        final double[] distances;
        final double maxDistance, spread;
        final HashMap<Integer, Voice> voices = new HashMap<>();
        final ArrayList<Voice> active = new ArrayList<>();
        int nextStep = Integer.MAX_VALUE, nextCount, previousStep;
        long omitted;

        Track(Path file, int pitch, int durationMs, double[] distances,
              double maxDistance, double spread, int sampleRate) throws IOException {
            this.pitch = pitch; this.durationMs = durationMs; this.distances = distances;
            this.maxDistance = maxDistance; this.spread = spread; this.sampleRate = sampleRate;
            in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file), BUFFER_BYTES));
            try { advanceHeader(); } catch (IOException | RuntimeException ex) { in.close(); throw ex; }
        }

        void advanceHeader() throws IOException {
            int first = in.read();
            if (first == -1) { nextStep = Integer.MAX_VALUE; nextCount = 0; return; }
            try {
                nextStep = (first << 24) | (in.readUnsignedByte() << 16)
                        | (in.readUnsignedByte() << 8) | in.readUnsignedByte();
                nextCount = in.readInt();
            } catch (EOFException ex) { throw new IOException("Truncated audio frame header for pitch " + pitch, ex); }
            if (nextStep <= previousStep || nextStep > durationMs || nextCount <= 0
                    || nextCount > distances.length)
                throw new IOException("Invalid audio frame header for pitch " + pitch);
            previousStep = nextStep;
        }

        void endpoint(long step) throws IOException {
            for (Voice voice : active) voice.nextAmplitude = 0;
            if (step != nextStep) return;
            for (int k = 0; k < nextCount; k++) {
                int cell = in.readInt();
                double amplitude = in.readDouble();
                if (cell < 0 || cell >= distances.length || !Double.isFinite(amplitude) || amplitude < 0)
                    throw new IOException("Invalid cell/activity at pitch " + pitch + ", step " + step);
                Voice voice = voices.get(cell);
                if (voice == null) {
                    double distance = distances[cell];
                    if (!Double.isFinite(distance) || distance < 0
                            || distance > maxDistance + Math.max(1e-12, maxDistance * 1e-12))
                        throw new IOException("Invalid/unmapped distance for cell " + cell);
                    double cents = maxDistance == 0 ? 0 : spread * Math.min(1.0, distance / maxDistance);
                    double base = 440.0 * Math.pow(2.0, (pitch - 69) / 12.0);
                    voice = new Voice(base, cents, sampleRate);
                    voices.put(cell, voice);
                    if (!voice.audible) omitted++;
                }
                if (voice.endpointStep == step)
                    throw new IOException("Repeated cell in audio frame for pitch " + pitch);
                voice.endpointStep = step;
                if (!voice.audible || amplitude == 0) continue;
                if (voice.amplitude == 0) active.add(voice);
                voice.nextAmplitude = amplitude;
            }
            advanceHeader();
        }

        void mix(double[] mix, long startSample, int count, double fractionAtStart, int rate) {
            for (int i = active.size() - 1; i >= 0; i--) {
                Voice voice = active.get(i);
                voice.add(mix, startSample, count, fractionAtStart, rate);
                voice.amplitude = voice.nextAmplitude;
                if (voice.amplitude == 0) {
                    int last = active.size() - 1;
                    active.set(i, active.get(last));
                    active.remove(last);
                }
            }
        }

        public void close() throws IOException { in.close(); }
    }

    private static final class Voice {
        final double omegaLow, omegaHigh, sinStepLow, cosStepLow, sinStepHigh, cosStepHigh;
        final boolean pair, audible;
        double sinLow, cosLow = 1, sinHigh, cosHigh = 1;
        double amplitude, nextAmplitude;
        long nextSample, lastAnchor, endpointStep = -1;

        Voice(double baseFrequency, double cents, int sampleRate) {
            pair = cents != 0;
            double ratio = Math.pow(2.0, cents / 1200.0);
            double low = baseFrequency / ratio;
            double high = baseFrequency * ratio;
            audible = low > 0 && high < sampleRate / 2.0;
            omegaLow = TWO_PI * low / sampleRate;
            omegaHigh = TWO_PI * high / sampleRate;
            sinStepLow = Math.sin(omegaLow); cosStepLow = Math.cos(omegaLow);
            sinStepHigh = Math.sin(omegaHigh); cosStepHigh = Math.cos(omegaHigh);
        }

        void anchor(long sample) {
            // Absolute time preserves phase across silence. Periodic anchoring
            // also bounds oscillator recurrence drift without a phase restart.
            double lowPhase = Math.IEEEremainder(omegaLow * sample, TWO_PI);
            sinLow = Math.sin(lowPhase); cosLow = Math.cos(lowPhase);
            if (pair) {
                double highPhase = Math.IEEEremainder(omegaHigh * sample, TWO_PI);
                sinHigh = Math.sin(highPhase); cosHigh = Math.cos(highPhase);
            }
            nextSample = sample; lastAnchor = sample;
        }

        void add(double[] mix, long start, int count, double fractionAtStart, int sampleRate) {
            if (nextSample != start || start - lastAnchor >= 4096) anchor(start);
            double value = amplitude + (nextAmplitude - amplitude) * fractionAtStart;
            double increment = (nextAmplitude - amplitude) * (1000.0 / sampleRate);
            double sl = sinLow, cl = cosLow, sh = sinHigh, ch = cosHigh;
            if (pair) {
                for (int j = 0; j < count; j++) {
                    mix[j] += value * (0.5 * (sl + sh));
                    double newSl = sl * cosStepLow + cl * sinStepLow;
                    cl = cl * cosStepLow - sl * sinStepLow; sl = newSl;
                    double newSh = sh * cosStepHigh + ch * sinStepHigh;
                    ch = ch * cosStepHigh - sh * sinStepHigh; sh = newSh;
                    value += increment;
                }
            } else {
                for (int j = 0; j < count; j++) {
                    mix[j] += value * sl;
                    double newSl = sl * cosStepLow + cl * sinStepLow;
                    cl = cl * cosStepLow - sl * sinStepLow; sl = newSl;
                    value += increment;
                }
            }
            sinLow = sl; cosLow = cl; sinHigh = sh; cosHigh = ch;
            nextSample = start + count;
        }
    }

    private static void writeWav(Path spool, Path output, long samples, int sampleRate,
                                 double gain) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(
                     Files.newInputStream(spool), BUFFER_BYTES));
             OutputStream out = new BufferedOutputStream(Files.newOutputStream(output), BUFFER_BYTES)) {
            out.write(new byte[]{'R','I','F','F'}); writeLE32(out, 36 + samples * 2);
            out.write(new byte[]{'W','A','V','E','f','m','t',' '}); writeLE32(out, 16);
            writeLE16(out, 1); writeLE16(out, 1); writeLE32(out, sampleRate);
            writeLE32(out, sampleRate * 2L); writeLE16(out, 2); writeLE16(out, 16);
            out.write(new byte[]{'d','a','t','a'}); writeLE32(out, samples * 2);
            for (long i = 0; i < samples; i++) {
                long pcm = Math.round(in.readFloat() * gain);
                // The global peak gives a 5% margin; hitting the range is an error,
                // rather than silently clipping and changing the mixture.
                if (pcm < -32768 || pcm > 32767) throw new IOException("Unexpected PCM clipping");
                writeLE16(out, (int) pcm);
            }
            if (in.read() != -1) throw new IOException("Audio spool length mismatch");
        }
    }

    private static void writeLE16(OutputStream out, int value) throws IOException {
        out.write(value & 255); out.write((value >>> 8) & 255);
    }
    private static void writeLE32(OutputStream out, long value) throws IOException {
        out.write((int) value & 255); out.write((int) (value >>> 8) & 255);
        out.write((int) (value >>> 16) & 255); out.write((int) (value >>> 24) & 255);
    }

    /** Standalone deterministic checks: java ChaconneAudio --self-test. */
    public static void main(String[] args) throws Exception {
        if (args.length != 1 || !args[0].equals("--self-test"))
            throw new IllegalArgumentException("Usage: java ChaconneAudio --self-test; call render(...) from the graph runner");
        Path dir = Files.createTempDirectory(Path.of("."), ".chaconne-audio-test-");
        try {
            int rate = 8000, duration = 120;
            Path tracks = Files.createDirectory(dir.resolve("tracks"));
            Path silence = dir.resolve("silence.wav");
            RenderStats silenceStats = renderWithStats(tracks, silence, duration, 69, 72,
                    new double[]{0, 100}, 100, 50, rate);
            check(silenceStats.rawPeak() == 0, "Silent render peak");
            check(silenceStats.samples() == 968, "Release duration/sample count");
            for (short sample : readPcm(silence)) check(sample == 0, "Silent PCM");

            writeTestTrack(tracks, 69, 0, 1.0, duration);
            Path base = dir.resolve("base.wav"), zeroMax = dir.resolve("zero-max.wav");
            render(tracks, base, duration, 69, 72, new double[]{0,100}, 100,50,rate);
            render(tracks, zeroMax, duration, 69,72,new double[]{0,100},0,50,rate);
            check(java.util.Arrays.equals(Files.readAllBytes(base),Files.readAllBytes(zeroMax)),
                    "Zero distance must equal exact base tone regardless of global Dmax");
            compareAnalytic(base, duration, rate, new double[]{440}, new double[]{1});

            Files.delete(tracks.resolve("pitch-69.bin"));
            writeTestTrack(tracks, 69, 1, 1.0, duration);
            Path detuned = dir.resolve("detuned.wav");
            render(tracks,detuned,duration,69,72,new double[]{0,100},100,50,rate);
            double ratio = Math.pow(2,50.0/1200);
            compareAnalytic(detuned,duration,rate,new double[]{440/ratio,440*ratio},new double[]{.5,.5});

            Files.delete(tracks.resolve("pitch-69.bin"));
            writeTestTrack(tracks,69,0,1.0,duration);
            writeTestTrack(tracks,72,0,.5,duration);
            Path chord = dir.resolve("chord.wav");
            render(tracks,chord,duration,69,72,new double[]{0},0,50,rate);
            compareAnalytic(chord,duration,rate,new double[]{440,440*Math.pow(2,3.0/12)},new double[]{1,.5});

            Files.delete(tracks.resolve("pitch-72.bin"));
            try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(tracks.resolve("pitch-69.bin")))) {
                for(int step:new int[]{1,79}) {
                    out.writeInt(step);out.writeInt(1);out.writeInt(0);out.writeDouble(1.0);
                }
            }
            Path sparse = dir.resolve("sparse.wav");
            render(tracks,sparse,duration,69,72,new double[]{0},0,50,44100);
            short[] actual=readPcm(sparse);double[] expected=new double[actual.length];double sparsePeak=0;
            for(int i=0;i<expected.length;i++) {
                double timeMs=i*1000.0/44100;
                double envelope=Math.max(0,1-Math.abs(timeMs-1))+Math.max(0,1-Math.abs(timeMs-79));
                expected[i]=(float)(envelope*Math.sin(TWO_PI*440*i/44100));
                sparsePeak=Math.max(sparsePeak,Math.abs(expected[i]));
            }
            for(int i=0;i<actual.length;i++)
                check(Math.abs(actual[i]-Math.round(expected[i]*(PCM_TARGET/sparsePeak)))<=2,
                        "Phase continuity across silence / fractional sample boundaries at "+i);
            System.out.println("ChaconneAudio self-test passed: silence, exact base at zero distance, symmetric detuning, concurrent pitches, linear envelopes, PCM normalization, duration, phase continuity across silence and 44.1kHz fractional sample boundaries.");
        } finally {
            try (var paths = Files.walk(dir)) {
                for (Path path : paths.sorted(java.util.Comparator.reverseOrder()).toList()) Files.deleteIfExists(path);
            }
        }
    }

    private static void writeTestTrack(Path dir,int pitch,int cell,double amplitude,int duration) throws IOException {
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(dir.resolve("pitch-"+pitch+".bin")))) {
            for(int step=1;step<=duration;step++) {
                out.writeInt(step);out.writeInt(1);out.writeInt(cell);out.writeDouble(amplitude);
            }
        }
    }
    private static short[] readPcm(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        check(bytes.length>=44 && bytes[0]=='R' && bytes[8]=='W',"WAV header");
        short[] pcm = new short[(bytes.length-44)/2];
        for(int i=0;i<pcm.length;i++) pcm[i]=(short)((bytes[44+i*2]&255)|((bytes[45+i*2]&255)<<8));
        return pcm;
    }
    private static void compareAnalytic(Path wav,int duration,int sampleRate,double[] frequencies,double[] amplitudes) throws IOException {
        short[] actual=readPcm(wav);
        double[] expected = new double[actual.length];double peak=0;
        for(int i=0;i<expected.length;i++) {
            double ms=i*1000.0/sampleRate;
            double envelope=Math.max(0,Math.min(1,Math.min(ms,duration+1-ms)));
            for(int k=0;k<frequencies.length;k++) expected[i]+=envelope*amplitudes[k]*Math.sin(TWO_PI*frequencies[k]*i/sampleRate);
            expected[i]=(float)expected[i];peak=Math.max(peak,Math.abs(expected[i]));
        }
        for(int i=0;i<actual.length;i++) {
            long target=Math.round(expected[i]*(PCM_TARGET/peak));
            check(Math.abs(actual[i]-target)<=2,"Analytic tone/envelope mismatch at sample "+i);
            check(Math.abs((int)actual[i])<=31130,"PCM normalization headroom");
        }
    }
    private static void check(boolean value,String message) {
        if(!value) throw new AssertionError(message);
    }
}
