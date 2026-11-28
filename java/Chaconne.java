// Offline sonification of a MIDI score with the RINGS FlyWire model (Java 21).
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public final class Chaconne {
    private static final String DEFAULT_INPUT = "720575940624963786";

    record Geometry(double[] distance, int located) {
        static Geometry read(Path file, long[] ids, int stimulus) throws IOException {
            var index = new HashMap<Long, Integer>(ids.length * 2);
            for (int i = 0; i < ids.length; i++) index.put(ids[i], i);
            double[] x = new double[ids.length], y = new double[ids.length], z = new double[ids.length];
            Arrays.fill(x, Double.NaN);
            int located = 0;
            try (var reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                String header = reader.readLine();
                if (!"root_id,x_um,y_um,z_um".equals(header)) throw new IOException("Expected root_id,x_um,y_um,z_um in " + file);
                for (String line; (line = reader.readLine()) != null;) {
                    String[] fields = line.split(",", -1);
                    if (fields.length != 4) throw new IOException("Invalid coordinate row in " + file);
                    Integer i = index.get(Long.parseUnsignedLong(fields[0]));
                    if (i == null) continue;
                    if (!Double.isNaN(x[i])) throw new IOException("Duplicate coordinates for " + fields[0]);
                    x[i] = Double.parseDouble(fields[1]); y[i] = Double.parseDouble(fields[2]); z[i] = Double.parseDouble(fields[3]);
                    if (!Double.isFinite(x[i]) || !Double.isFinite(y[i]) || !Double.isFinite(z[i])) throw new IOException("Nonfinite coordinate for " + fields[0]);
                    located++;
                }
            }
            if (Double.isNaN(x[stimulus])) throw new IOException("No coordinates for the input neuron");
            double[] radius = new double[ids.length];
            for (int i = 0; i < ids.length; i++)
                radius[i] = Math.hypot(Math.hypot(x[i] - x[stimulus], y[i] - y[stimulus]), z[i] - z[stimulus]);
            return new Geometry(radius, located);
        }
    }

    // One disk track per pitch keeps memory independent of the number of voices.
    // Thresholding affects only the sound. All edges and neurons participate in RINGS.
    static final class Track implements AutoCloseable, ChaconneSimulation.SparseStream {
        final DataOutputStream out;
        final double[] radius;
        final double threshold;
        final int[] audible;
        double maxDistance;
        long samples;
        Track(Path path, Geometry geometry, double threshold) throws IOException {
            radius = geometry.distance();
            audible = new int[radius.length];
            this.threshold = threshold;
            out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path), 1 << 16));
        }
        @Override public void sparseFrame(int step, double[] activity, int[] active, int activeCount) throws IOException {
            int count = 0;
            for (int j = 0; j < activeCount; j++) {
                int i = active[j];
                if (activity[i] > threshold && Double.isFinite(radius[i])) audible[count++] = i;
            }
            if (count == 0) return;
            Arrays.sort(audible, 0, count); // Stable mix order across equivalent simulation engines.
            out.writeInt(step); out.writeInt(count);
            for (int j = 0; j < count; j++) {
                int i = audible[j];
                maxDistance = Math.max(maxDistance, radius[i]);
                out.writeInt(i); out.writeDouble(activity[i]);
            }
            samples += count;
        }
        @Override public void close() throws IOException { out.close(); }
    }

    static int positive(String s, String name) {
        int i = Integer.parseInt(s);
        if (i <= 0) throw new IllegalArgumentException(name + " must be positive");
        return i;
    }
    static double nonnegative(String s) {
        double x = Double.parseDouble(s);
        if (!Double.isFinite(x) || x < 0) throw new IllegalArgumentException("Expected a finite nonnegative value: " + s);
        return x;
    }

    static void run(String[] args) throws Exception {
        var options = new HashMap<String,String>();
        Set<String> names = Set.of("--graph","--coords","--midi","--output","--stimulus","--threads","--duration-ms","--tail-ms","--threshold","--spread-cents");
        for (int i=0; i < args.length; i+=2) {
            if (!names.contains(args[i]) || i+1 >= args.length || options.put(args[i],args[i+1]) != null)
                throw new IllegalArgumentException("Unknown, missing or duplicate option: " + args[i]);
        }
        Path midi = Path.of(options.getOrDefault("--midi","music/bach-chaconne-bwv1004.mid"));
        Path graphPath = Path.of(options.getOrDefault("--graph",".runtime/java_graph/graph.bin"));
        Path coords = Path.of(options.getOrDefault("--coords",".runtime/chaconne-coordinates.csv"));
        Path output = Path.of(options.getOrDefault("--output",".runtime/chaconne.wav")).toAbsolutePath();
        if (!output.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".wav"))
            throw new IllegalArgumentException("--output must end in .wav");
        int threads = positive(options.getOrDefault("--threads","4"), "--threads");
        double threshold = nonnegative(options.getOrDefault("--threshold","0.001"));
        double spread = nonnegative(options.getOrDefault("--spread-cents","50"));
        if (spread > 50) throw new IllegalArgumentException("--spread-cents maximum is 50");
        int tail = Integer.parseInt(options.getOrDefault("--tail-ms","250"));
        if (tail < 0) throw new IllegalArgumentException("--tail-ms cannot be negative");
        var score = ChaconneScore.read(midi);
        for (String warning : score.warnings()) System.err.println("MIDI: " + warning);
        long full = Math.addExact(score.durationMillis(),tail);
        long requested = options.containsKey("--duration-ms") ? positive(options.get("--duration-ms"),"--duration-ms") : full;
        int duration = Math.toIntExact(Math.min(requested,full));
        final int sampleRate = 44100;
        if (duration < 1 || (duration + 1L) * sampleRate / 1000 * 2 > 0xfffffff0L - 44)
            throw new IllegalArgumentException("Duration outside PCM WAV limits");
        var graph = RingsBenchmark.loadGraph(graphPath);
        long stimulusId = Long.parseUnsignedLong(options.getOrDefault("--stimulus",DEFAULT_INPUT));
        int stimulus = -1;
        for (int i=0; i<graph.size(); i++) if (graph.ids()[i] == stimulusId) { stimulus=i; break; }
        if (stimulus < 0) throw new IllegalArgumentException("Input neuron missing from graph");
        var geometry = Geometry.read(coords,graph.ids(),stimulus);
        Files.createDirectories(output.getParent());
        Path temp = Files.createTempDirectory(output.getParent(),"rings-audio-");
        double maxDistance = 0;
        long audibleValues = 0;
        int activePitches = 0;
        try {
            for (int pitch = score.minPitch(); pitch <= score.maxPitch(); pitch++) {
                boolean[] down = new boolean[duration];
                boolean used = false;
                for (var note : score.notes()) {
                    if (note.pitch() != pitch) continue;
                    int start = (int)Math.min(duration,note.startMillis());
                    int end = (int)Math.min(duration,note.endMillis());
                    if (start < end) { Arrays.fill(down,start,end,true); used=true; }
                }
                if (!used) continue;
                activePitches++;
                System.err.printf("MIDI %d, %d ms%n", pitch, duration);
                try (var track = new Track(temp.resolve("pitch-"+pitch+".bin"),geometry,threshold)) {
                    ChaconneSimulation.run(graph,stimulus,down,threads,track);
                    maxDistance = Math.max(maxDistance,track.maxDistance);
                    audibleValues += track.samples;
                }
            }
            Path staged = temp.resolve("mix.wav");
            ChaconneAudio.render(temp,staged,duration,score.minPitch(),score.maxPitch(),geometry.distance(),maxDistance,spread,sampleRate);
            try { Files.move(staged,output,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING); }
            catch (AtomicMoveNotSupportedException ex) { Files.move(staged,output,StandardCopyOption.REPLACE_EXISTING); }
            System.out.printf(Locale.ROOT,"{\"score_ms\":%d,\"simulated_ms\":%d,\"min_pitch\":%d,\"max_pitch\":%d,\"rendered_pitches\":%d,\"notes_in_score\":%d,\"nodes\":%d,\"edges\":%d,\"located\":%d,\"audible_values\":%d,\"distance_max_um\":%.9f,\"threshold\":%.9g,\"spread_cents\":%.9g}%n",
                    score.durationMillis(),duration,score.minPitch(),score.maxPitch(),activePitches,score.notes().size(),graph.size(),graph.pre().length,geometry.located(),audibleValues,maxDistance,threshold,spread);
        } finally {
            try (var files=Files.newDirectoryStream(temp)) { for (Path file:files) Files.deleteIfExists(file); }
            Files.deleteIfExists(temp);
        }
    }
    public static void main(String[] args) {
        try { run(args); }
        catch (Exception e) { System.err.println("Chaconne: "+ e); System.exit(1); }
    }
}
