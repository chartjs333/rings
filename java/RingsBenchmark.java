/*
 * Java 21 port of cpp/rings_graph.hpp at 4db57f6654b123fb66c69725b00a2b969f2d5a19.
 * Same float64 arithmetic and ordered CSR reductions. No approximate math.
 * Build: javac -d build/java java/RingsBenchmark.java
 * Run: java -cp build/java RingsBenchmark --graph graph.bin --stimulus ID
 */
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.lang.management.CompilationMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Phaser;

public final class RingsBenchmark {
    static final double PI = 3.141592653589793238462643383279502884;
    static final double TWO_PI = 2.0 * PI;
    static final double INV_TWO_PI = 1.0 / TWO_PI;
    static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    record Config(double dt, double hz, double gain, double coupling, double width) {
        static Config defaults() { return new Config(.001, 10, 1, .4, PI / 2); }
        void validate() {
            for (double value : new double[]{dt, hz, gain, coupling, width})
                if (!Double.isFinite(value) || value < 0)
                    throw new IllegalArgumentException("Invalid config");
            if (dt == 0 || width > PI || !Double.isFinite(2 * PI * hz * dt))
                throw new IllegalArgumentException("Invalid time or width");
        }
    }

    record Graph(long[] ids, long[] row, int[] pre, double[] weight) {
        int size() { return ids.length; }
        void validate() {
            int n = size();
            if (n == 0 || row.length != n + 1 || row[0] != 0
                    || row[n] != pre.length || pre.length != weight.length)
                throw new IllegalArgumentException("Invalid CSR");
            for (int i = 1; i <= n; i++)
                if (row[i] < row[i - 1] || row[i] > pre.length)
                    throw new IllegalArgumentException("Invalid CSR offset");
            for (int p : pre)
                if (p < 0 || p >= n) throw new IllegalArgumentException("Invalid index");
            for (double w : weight)
                if (!Double.isFinite(w)) throw new IllegalArgumentException("Invalid weight");
        }
    }

    static double wrap(double angle) {
        if (!Double.isFinite(angle)) throw new ArithmeticException("Non-finite angle");
        double result = Math.abs(angle) <= 0x1p20
                ? angle - TWO_PI * Math.floor((angle + PI) * INV_TWO_PI)
                : Math.IEEEremainder(angle, TWO_PI);
        if (result < -PI) result += TWO_PI;
        if (result >= PI) result -= TWO_PI;
        return result;
    }

    static double gatedValue(double value, boolean open) {
        long bits = Double.doubleToRawLongBits(value);
        bits &= open ? -1L : 0L;
        return Double.longBitsToDouble(bits);
    }

    static final class Network implements AutoCloseable {
        final Graph graph;
        final Config config;
        final int threadCount;
        final String mode;
        final boolean inline;
        final Phaser barrier;
        final Thread[] workers;
        final boolean[] bad;
        volatile boolean stopping;
        volatile Throwable failure;
        double[] external;
        double[] activity, phase, next, nextPhase;
        byte[] gate, nextGate;
        final byte[] silenced;

        Network(Graph graph, Config config, int threadCount, String mode) {
            graph.validate();
            config.validate();
            if (threadCount < 1 || threadCount > 1024)
                throw new IllegalArgumentException("Invalid thread count");
            if (!mode.equals("platform") && !mode.equals("virtual"))
                throw new IllegalArgumentException("Unknown Java thread mode");
            this.graph = graph;
            this.config = config;
            this.threadCount = threadCount;
            this.mode = mode;
            int n = graph.size();
            activity = new double[n]; phase = new double[n];
            next = new double[n]; nextPhase = new double[n];
            gate = new byte[n]; Arrays.fill(gate, (byte) 1);
            nextGate = new byte[n]; silenced = new byte[n];
            inline = threadCount == 1 && mode.equals("platform");
            barrier = inline ? null : new Phaser(threadCount + 1);
            workers = new Thread[inline ? 0 : threadCount];
            bad = new boolean[inline ? 1 : threadCount];
            try {
                for (int worker = 0; worker < workers.length; worker++) {
                    final int id = worker;
                    int count = n / threadCount;
                    int extra = n % threadCount;
                    int first = id * count + Math.min(id, extra);
                    int stop = first + count + (id < extra ? 1 : 0);
                    Runnable body = () -> runWorker(id, first, stop);
                    workers[id] = mode.equals("virtual")
                            ? Thread.ofVirtual().name("rings-v-" + id).unstarted(body)
                            : Thread.ofPlatform().name("rings-p-" + id).unstarted(body);
                    workers[id].start();
                }
            } catch (RuntimeException | Error error) {
                close();
                throw error;
            }
        }

        void runWorker(int id, int first, int stop) {
            try {
                while (barrier.arriveAndAwaitAdvance() >= 0) {
                    if (stopping) return;
                    bad[id] = compute(first, stop);
                    if (barrier.arriveAndAwaitAdvance() < 0) return;
                }
            } catch (Throwable error) {
                failure = error;
                barrier.forceTermination();
            }
        }

        // Each destination reads the previous step only. No heap allocations here.
        boolean compute(int first, int stop) {
            Graph g = graph;
            Config c = config;
            double[] oldActivity = activity;
            double[] oldPhase = phase;
            double increment = 2 * PI * c.hz() * c.dt();
            boolean nonfinite = false;
            for (int i = first; i < stop; i++) {
                double sum = 0;
                for (int k = (int) g.row()[i]; k < (int) g.row()[i + 1]; k++)
                    sum += g.weight()[k] * oldActivity[g.pre()[k]];
                double u = external[i] + c.coupling() * sum;
                double raw = oldPhase[i] + increment + c.gain() * u * c.dt();
                if (!Double.isFinite(u) || !Double.isFinite(raw)) {
                    nonfinite = true;
                    continue;
                }
                double a = wrap(raw);
                boolean open = Math.abs(a) <= c.width();
                nextPhase[i] = a;
                nextGate[i] = (byte) (open ? 1 : 0);
                // C++ std::max(0.0,u) returns +0 also for u == -0.
                double rectified = u > 0 ? u : 0.0;
                next[i] = gatedValue(rectified, open && silenced[i] == 0);
            }
            return nonfinite;
        }

        void step(double[] input) {
            if (input.length != activity.length)
                throw new IllegalArgumentException("Invalid input size");
            if (stopping || failure != null)
                throw new IllegalStateException("Network worker unavailable", failure);
            external = input;
            if (inline) {
                bad[0] = compute(0, graph.size());
            } else {
                barrier.arriveAndAwaitAdvance();
                barrier.arriveAndAwaitAdvance();
                if (failure != null || barrier.isTerminated())
                    throw new IllegalStateException("Network worker failed", failure);
            }
            for (boolean b : bad)
                if (b) throw new ArithmeticException("Nonfinite step");
            double[] tmp = activity; activity = next; next = tmp;
            tmp = phase; phase = nextPhase; nextPhase = tmp;
            byte[] bytes = gate; gate = nextGate; nextGate = bytes;
        }

        void silence(int i) {
            if (i < 0 || i >= activity.length) throw new IndexOutOfBoundsException(i);
            silenced[i] = 1;
            activity[i] = 0.0;
        }

        @Override public void close() {
            stopping = true;
            if (barrier != null) barrier.forceTermination();
            boolean interrupted = false;
            for (Thread worker : workers) {
                if (worker == null || worker == Thread.currentThread()) continue;
                while (worker.isAlive()) {
                    try { worker.join(); }
                    catch (InterruptedException ex) { interrupted = true; }
                }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    static Graph loadGraph(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long length = channel.size();
            if (length < 24 || length > Integer.MAX_VALUE)
                throw new IOException("Graph file must be between 24 bytes and 2 GiB");
            MappedByteBuffer data = channel.map(FileChannel.MapMode.READ_ONLY, 0, length);
            data.order(LE);
            byte[] magic = new byte[8]; data.get(magic);
            if (!Arrays.equals(magic, "RINGSCSR".getBytes(StandardCharsets.US_ASCII)))
                throw new IOException("Invalid graph magic");
            long nodes = data.getLong(), edges = data.getLong();
            if (nodes < 1 || nodes >= Integer.MAX_VALUE || edges < 0 || edges > Integer.MAX_VALUE)
                throw new IOException("Invalid graph dimensions");
            long expected = 24 + nodes * 8 + (nodes + 1) * 8 + edges * 4 + edges * 8;
            if (length != expected) throw new IOException("Graph file size mismatch");
            int n = Math.toIntExact(nodes), m = Math.toIntExact(edges);
            long[] ids = new long[n], row = new long[n + 1];
            int[] pre = new int[m]; double[] weight = new double[m];
            data.asLongBuffer().get(ids); data.position(data.position() + n * 8);
            data.asLongBuffer().get(row); data.position(data.position() + (n + 1) * 8);
            data.asIntBuffer().get(pre); data.position(data.position() + m * 4);
            data.asDoubleBuffer().get(weight);
            Graph graph = new Graph(ids, row, pre, weight);
            graph.validate();
            return graph;
        }
    }

    static void writeAll(FileChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) channel.write(buffer);
    }

    static void writeDoubles(FileChannel channel, ByteBuffer buffer, double[] values) throws IOException {
        buffer.clear();
        for (double value : values) {
            if (buffer.remaining() < 8) {
                buffer.flip(); writeAll(channel, buffer); buffer.clear();
            }
            buffer.putDouble(value);
        }
        buffer.flip(); writeAll(channel, buffer);
    }

    static void writeState(Network net, Path path) throws IOException {
        if (path == null) return;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            ByteBuffer buffer = ByteBuffer.allocate(1024 * 1024).order(LE);
            buffer.put("RINGSSTA".getBytes(StandardCharsets.US_ASCII));
            buffer.putLong(net.activity.length);
            buffer.flip(); writeAll(channel, buffer);
            writeDoubles(channel, buffer, net.activity);
            writeDoubles(channel, buffer, net.phase);
            writeAll(channel, ByteBuffer.wrap(net.gate));
        }
    }

    record GcStats(long count, long millis) {
        static GcStats capture() {
            long count = 0, millis = 0;
            for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
                if (gc.getCollectionCount() >= 0) count += gc.getCollectionCount();
                if (gc.getCollectionTime() >= 0) millis += gc.getCollectionTime();
            }
            return new GcStats(count, millis);
        }
    }

    static long compilationTime() {
        CompilationMXBean bean = ManagementFactory.getCompilationMXBean();
        return bean != null && bean.isCompilationTimeMonitoringSupported()
                ? bean.getTotalCompilationTime() : -1;
    }

    static String quote(String value) {
        StringBuilder b = new StringBuilder("\"");
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (c == '\\' || c == '"') b.append('\\').append(c);
            else if (c < 32) b.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
            else b.append(c);
        }
        return b.append('"').toString();
    }

    static void benchmark(Graph graph, double[] input, String mode, int threads,
                          int steps, int warmup, Path output) throws IOException {
        double warmupSeconds;
        try (Network warm = new Network(graph, Config.defaults(), threads, mode)) {
            long start = System.nanoTime();
            for (int i = 0; i < warmup; i++) warm.step(input);
            warmupSeconds = (System.nanoTime() - start) * 1e-9;
        }
        try (Network net = new Network(graph, Config.defaults(), threads, mode)) {
            GcStats beforeGc = GcStats.capture();
            long beforeCompilation = compilationTime();
            long start = System.nanoTime();
            for (int i = 0; i < steps; i++) net.step(input);
            double seconds = (System.nanoTime() - start) * 1e-9;
            long afterCompilation = compilationTime();
            GcStats afterGc = GcStats.capture();
            double activitySum = 0, phaseSum = 0;
            int nonzero = 0;
            boolean finite = true;
            for (int i = 0; i < graph.size(); i++) {
                double a = net.activity[i], p = net.phase[i];
                finite &= Double.isFinite(a) && Double.isFinite(p);
                activitySum += a; phaseSum += p;
                if (a != 0) nonzero++;
            }
            if (!finite) throw new ArithmeticException("Nonfinite final state");
            writeState(net, output);
            long compileDelta = beforeCompilation < 0 || afterCompilation < 0
                    ? -1 : afterCompilation - beforeCompilation;
            System.out.printf(Locale.ROOT,
                    "{\"benchmark_version\":1,\"mode\":%s,\"threads\":%d,\"nodes\":%d,\"edges\":%d,"
                    + "\"steps\":%d,\"warmup_steps\":%d,\"warmup_seconds\":%.17g,"
                    + "\"seconds\":%.17g,\"steps_per_second\":%.17g,\"finite\":true,"
                    + "\"final_activity_sum\":%.17g,\"final_phase_sum\":%.17g,"
                    + "\"final_nonzero_activity\":%d,\"compilation_millis_before\":%d,"
                    + "\"compilation_millis_after\":%d,\"compilation_millis_delta\":%d,"
                    + "\"gc_collections_delta\":%d,\"gc_millis_delta\":%d,"
                    + "\"java_version\":%s,\"java_vm_name\":%s}%n",
                    quote(mode), threads, graph.size(), graph.pre().length,
                    steps, warmup, warmupSeconds, seconds, steps / seconds,
                    activitySum, phaseSum, nonzero, beforeCompilation, afterCompilation, compileDelta,
                    afterGc.count() - beforeGc.count(), afterGc.millis() - beforeGc.millis(),
                    quote(System.getProperty("java.version")), quote(System.getProperty("java.vm.name")));
        }
    }

    static boolean validateTrace(Graph graph, double[] input, String mode, int threads,
                                 int steps, Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(
                    java.nio.file.Files.newInputStream(path), 1024 * 1024));
             Network net = new Network(graph, Config.defaults(), threads, mode)) {
            byte[] header = in.readNBytes(24);
            if (header.length != 24) throw new EOFException("Truncated trace header");
            ByteBuffer h = ByteBuffer.wrap(header).order(LE);
            byte[] magic = new byte[8]; h.get(magic);
            if (!Arrays.equals(magic, "RINGSTRC".getBytes(StandardCharsets.US_ASCII))
                    || h.getLong() != graph.size() || h.getLong() != steps)
                throw new IOException("Trace magic, nodes or steps mismatch");
            byte[] bytes = new byte[Math.multiplyExact(graph.size(), 17)];
            ByteBuffer data = ByteBuffer.wrap(bytes).order(LE);
            long activityMismatches = 0, phaseMismatches = 0, gateMismatches = 0;
            double maxActivity = 0, maxPhase = 0;
            for (int step = 0; step < steps; step++) {
                net.step(input);
                in.readFully(bytes);
                data.clear();
                for (int i = 0; i < graph.size(); i++) {
                    double expected = data.getDouble(), actual = net.activity[i];
                    if (!Double.isFinite(expected) || !Double.isFinite(actual))
                        throw new ArithmeticException("Nonfinite activity during validation");
                    maxActivity = Math.max(maxActivity, Math.abs(expected - actual));
                    if (Double.doubleToRawLongBits(expected) != Double.doubleToRawLongBits(actual))
                        activityMismatches++;
                }
                for (int i = 0; i < graph.size(); i++) {
                    double expected = data.getDouble(), actual = net.phase[i];
                    if (!Double.isFinite(expected) || !Double.isFinite(actual))
                        throw new ArithmeticException("Nonfinite phase during validation");
                    maxPhase = Math.max(maxPhase, Math.abs(expected - actual));
                    if (Double.doubleToRawLongBits(expected) != Double.doubleToRawLongBits(actual))
                        phaseMismatches++;
                }
                for (int i = 0; i < graph.size(); i++)
                    if (data.get() != net.gate[i]) gateMismatches++;
            }
            if (in.read() != -1) throw new IOException("Trailing trace bytes");
            boolean passed = activityMismatches == 0 && phaseMismatches == 0 && gateMismatches == 0;
            System.out.printf(Locale.ROOT,
                    "{\"validation\":\"all_steps_vs_cpp\",\"mode\":%s,\"threads\":%d,"
                    + "\"nodes\":%d,\"steps\":%d,\"max_activity_error\":%.17g,"
                    + "\"max_phase_error\":%.17g,\"activity_bit_mismatches\":%d,"
                    + "\"phase_bit_mismatches\":%d,\"gate_mismatches\":%d,\"passed\":%s}%n",
                    quote(mode), threads, graph.size(), steps, maxActivity, maxPhase,
                    activityMismatches, phaseMismatches, gateMismatches, passed);
            return passed;
        }
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static void same(Network expected, Network actual) {
        for (int i = 0; i < expected.activity.length; i++) {
            check(Double.doubleToRawLongBits(expected.activity[i])
                    == Double.doubleToRawLongBits(actual.activity[i]), "Activity mismatch at " + i);
            check(Double.doubleToRawLongBits(expected.phase[i])
                    == Double.doubleToRawLongBits(actual.phase[i]), "Phase mismatch at " + i);
            check(expected.gate[i] == actual.gate[i], "Gate mismatch at " + i);
        }
    }

    static boolean selfTest() {
        check(Double.doubleToRawLongBits(gatedValue(-1.0, false)) == 0L, "Closed +0");
        check(Double.doubleToRawLongBits(gatedValue(-0.0, true))
                == Double.doubleToRawLongBits(-0.0), "Open -0 preserved");
        check(wrap(PI) == -PI && wrap(-PI) == -PI && wrap(0) == 0, "Known wrap values");
        for (double angle : new double[]{Math.nextUp(PI), Math.nextDown(PI),
                Math.nextUp(-PI), Math.nextDown(-PI), 0x1p21, -0x1p21, 1e100, -1e100}) {
            double result = wrap(angle);
            check(result >= -PI && result < PI, "Wrap range");
        }
        Graph isolated = new Graph(new long[]{1}, new long[]{0,0},
                new int[]{}, new double[]{});
        Config boundaryConfig = new Config(.001,0,0,0,.5);
        for (double angle : new double[]{-.5,.5,Math.nextDown(-.5),Math.nextUp(.5)}) {
            try (Network net = new Network(isolated,boundaryConfig,1,"platform")) {
                net.phase[0] = angle;
                net.step(new double[]{1});
                boolean expectedOpen = Math.abs(angle) <= .5;
                check(net.gate[0] == (expectedOpen?1:0)
                        && net.activity[0] == (expectedOpen?1:0),"Inclusive gate boundary");
                if (!expectedOpen)
                    check(Double.doubleToRawLongBits(net.activity[0]) == 0,"Closed gate +0");
            }
        }
        Graph chain = new Graph(new long[]{1,2,3}, new long[]{0,0,1,3},
                new int[]{0,0,1}, new double[]{1,-.5,.5});
        Config noPhase = new Config(.001,0,0,1,PI/2);
        for (String mode : new String[]{"platform","virtual"}) {
            for (int threads : new int[]{1,4}) {
                try (Network net = new Network(chain,noPhase,threads,mode)) {
                    net.step(new double[]{2,0,0});
                    check(Arrays.equals(net.activity,new double[]{2,0,0}),"Step one");
                    net.step(new double[]{0,0,0});
                    check(Arrays.equals(net.activity,new double[]{0,2,0}),"Step two");
                    net.step(new double[]{0,0,0});
                    check(Arrays.equals(net.activity,new double[]{0,0,1}),"Step three");
                    net.silence(0); net.step(new double[]{2,0,0});
                    check(net.activity[0] == 0,"Silencing");
                    double[] beforeActivity = net.activity.clone(), beforePhase = net.phase.clone();
                    byte[] beforeGate = net.gate.clone();
                    boolean rejected = false;
                    try { net.step(new double[]{Double.NaN,0,0}); }
                    catch (ArithmeticException expected) { rejected = true; }
                    check(rejected,"Nonfinite rejection");
                    check(Arrays.equals(beforeActivity,net.activity)
                            && Arrays.equals(beforePhase,net.phase)
                            && Arrays.equals(beforeGate,net.gate),"No partial state commit");
                    net.step(new double[]{0,0,0}); // Recover after a rejected step.
                }
            }
        }
        Config driven = new Config(.1,0,1,0,.1);
        try (Network n = new Network(chain,driven,1,"platform")) {
            n.step(new double[]{2,0,0});
            check(n.gate[0] == 0 && n.activity[0] == 0 && Math.abs(n.phase[0]-.2)<1e-12,
                    "Driven phase closes gate");
        }
        int count = 19;
        long[] ids = new long[count], row = new long[count+1];
        List<Integer> pres = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        for (int i=0;i<count;i++) {
            ids[i]=1000+i; row[i]=pres.size();
            for (int k=0;k<i%5;k++) {
                pres.add((i*7+k*3)%count); weights.add((k%2==0?1.:-1.)/(i%5));
            }
        }
        row[count]=pres.size();
        Graph graph = new Graph(ids,row,pres.stream().mapToInt(Integer::intValue).toArray(),
                weights.stream().mapToDouble(Double::doubleValue).toArray());
        Config config = new Config(.003,13,.73,.61,.42);
        double[] boundaries = {0,PI,-PI,Math.nextDown(PI),Math.nextUp(-PI),
                Math.nextUp(PI),Math.nextDown(-PI),0x1p21,-0x1p21,1e100,-1e100};
        try (Network a = new Network(graph,config,1,"platform");
             Network b = new Network(graph,config,4,"platform");
             Network c = new Network(graph,config,4,"virtual")) {
            for (int i=0;i<count;i++) {
                a.phase[i]=b.phase[i]=c.phase[i]=boundaries[i%boundaries.length];
                a.activity[i]=b.activity[i]=c.activity[i]=(i%4)*.125;
                a.silenced[i]=b.silenced[i]=c.silenced[i]=(byte)(i%7==0?1:0);
            }
            double[] input = new double[count];
            for (int step=0;step<97;step++) {
                for (int i=0;i<count;i++) input[i]=((i+step)%9-4)*.2;
                a.step(input);b.step(input);c.step(input);
                same(a,b);same(a,c);
            }
        }
        boolean invalid = false;
        try { new Graph(new long[]{1},new long[]{0,1},new int[]{1},new double[]{1}).validate(); }
        catch (IllegalArgumentException expected) { invalid = true; }
        check(invalid,"Invalid index rejection");
        System.out.println("{\"self_test\":\"signed_graph_boundaries_large_phases_silencing_nonfinite_recovery_platform_virtual\",\"passed\":true}");
        return true;
    }

    static int natural(String value, boolean allowZero) {
        if (!value.matches("[0-9]+")) throw new IllegalArgumentException("Expected unsigned integer");
        int parsed = Integer.parseInt(value);
        if (parsed < (allowZero?0:1)) throw new IllegalArgumentException("Invalid count");
        return parsed;
    }

    public static void main(String[] args) {
        try {
            Path graphPath = null, output = null, trace = null;
            String mode = "platform";
            int threads = 1, steps = 400, warmup = 600;
            boolean test = false;
            List<Long> stimuli = new ArrayList<>();
            for (int i=0;i<args.length;i++) {
                String arg = args[i];
                if (arg.equals("--self-test")) { test=true; continue; }
                if (++i >= args.length) throw new IllegalArgumentException("Missing argument value");
                String value = args[i];
                switch (arg) {
                    case "--graph" -> graphPath=Path.of(value);
                    case "--output" -> output=Path.of(value);
                    case "--validate-against" -> trace=Path.of(value);
                    case "--mode" -> mode=value;
                    case "--threads" -> threads=natural(value,false);
                    case "--steps" -> steps=natural(value,false);
                    case "--warmup" -> warmup=natural(value,true);
                    case "--stimulus" -> {
                        if (!value.matches("[0-9]+")) throw new IllegalArgumentException("Invalid stimulus ID");
                        stimuli.add(Long.parseUnsignedLong(value));
                    }
                    default -> throw new IllegalArgumentException("Unknown argument: "+arg);
                }
            }
            if (test) { selfTest(); return; }
            if (graphPath == null || stimuli.isEmpty())
                throw new IllegalArgumentException("Required --graph PATH --stimulus ID; optional --threads N --mode platform|virtual --steps N --warmup N --output PATH --validate-against TRACE; or --self-test");
            Graph graph = loadGraph(graphPath);
            double[] input = new double[graph.size()];
            for (long id : stimuli) {
                int found=-1;
                for (int i=0;i<graph.size();i++) if (graph.ids()[i]==id) { found=i;break; }
                if (found<0) throw new IllegalArgumentException("Unknown stimulus ID");
                input[found]=1.;
            }
            if (trace != null) {
                if (!validateTrace(graph,input,mode,threads,steps,trace)) System.exit(1);
            } else benchmark(graph,input,mode,threads,steps,warmup,output);
        } catch (Throwable error) {
            System.err.println(error.getClass().getSimpleName()+": "+error.getMessage());
            System.exit(1);
        }
    }
}
