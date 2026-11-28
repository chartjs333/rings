/*
 * Exact activity simulation for independent pitch channels.
 * Uses the unchanged RingsBenchmark graph, constants and wrap function.
 * Callbacks use 1-based steps and borrowed arrays, valid until the next step.
 */
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Phaser;

public final class ChaconneSimulation {
    private ChaconneSimulation() {}
    public interface Stream {
        void frame(int step, double[] activity) throws IOException;
    }
    /** Optional zero-copy sparse callback; indices enumerate strictly positive values. */
    public interface SparseStream extends Stream {
        void sparseFrame(int step, double[] activity, int[] indices, int count) throws IOException;
        @Override default void frame(int step, double[] activity) throws IOException {
            int[] indices = new int[activity.length];
            int count = 0;
            for (int i = 0; i < activity.length; i++) if (activity[i] > 0) indices[count++] = i;
            sparseFrame(step, activity, indices, count);
        }
    }

    private static final RingsBenchmark.Config CONFIG = RingsBenchmark.Config.defaults();
    private static final double INCREMENT = 2 * RingsBenchmark.PI * CONFIG.hz() * CONFIG.dt();

    /**
     * Exact sparse is the default; -Drings.chaconne.engine=dense selects the reference.
     * Every frame through the last key-down is emitted (including leading silence).
     * After the final key-down, once all activity is exactly zero, later zero frames
     * are omitted. No threshold changes the dynamics.
     */
    public static void run(RingsBenchmark.Graph graph, int stimulusIndex,
                           boolean[] keyDown, int threads, Stream sink) throws Exception {
        if (stimulusIndex < 0 || stimulusIndex >= graph.size() || threads < 1 || threads > 1024)
            throw new IllegalArgumentException("Invalid stimulus or thread count");
        if (keyDown.length == Integer.MAX_VALUE)
            throw new IllegalArgumentException("Timeline too long");
        String engine = System.getProperty("rings.chaconne.engine", "sparse");
        if (engine.equals("dense")) runDense(graph, stimulusIndex, keyDown, threads, sink);
        else if (engine.equals("sparse")) runSparse(graph, stimulusIndex, keyDown, threads, sink);
        else throw new IllegalArgumentException("Unknown engine: " + engine);
    }

    static int lastKey(boolean[] keys) {
        for (int i = keys.length - 1; i >= 0; i--) if (keys[i]) return i + 1;
        return 0;
    }

    static double zeroStep(double phase) {
        return RingsBenchmark.wrap(phase + INCREMENT + CONFIG.gain() * 0.0 * CONFIG.dt());
    }

    static void emit(Stream sink, int step, double[] activity, int[] active, int count)
            throws IOException {
        if (sink instanceof SparseStream sparse) sparse.sparseFrame(step, activity, active, count);
        else sink.frame(step, activity);
    }

    static void runDense(RingsBenchmark.Graph graph, int stimulusIndex, boolean[] keys,
                         int threads, Stream sink) throws Exception {
        int last = lastKey(keys);
        if (last == 0) return;
        int first = 0;
        while (!keys[first]) first++;
        double phase = 0;
        double[] input = new double[graph.size()], zeros = new double[graph.size()];
        int[] active = new int[graph.size()];
        // Exact zero-input pre-roll: every node has the same autonomous state.
        for (int i = 0; i < first; i++) {
            phase = zeroStep(phase);
            emit(sink, i + 1, zeros, active, 0);
        }
        try (RingsBenchmark.Network net = new RingsBenchmark.Network(graph, CONFIG, threads, "platform")) {
            Arrays.fill(net.phase, phase);
            Arrays.fill(net.gate, (byte) (Math.abs(phase) <= CONFIG.width() ? 1 : 0));
            for (int i = first; i < keys.length; i++) {
                input[stimulusIndex] = keys[i] ? 1 : 0;
                net.step(input);
                int count = 0;
                for (int j = 0; j < net.activity.length; j++)
                    if (net.activity[j] > 0) active[count++] = j;
                emit(sink, i + 1, net.activity, active, count);
                if (i + 1 >= last && count == 0) break;
            }
        }
    }

    private static RingsBenchmark.Graph cachedGraph;
    private static Topology cachedTopology;
    static synchronized Topology topology(RingsBenchmark.Graph graph) {
        if (cachedGraph != graph) {
            cachedTopology = new Topology(graph);
            cachedGraph = graph;
        }
        return cachedTopology;
    }

    static final class Topology {
        final RingsBenchmark.Graph graph;
        final int[] outgoingRow, destinations;
        Topology(RingsBenchmark.Graph graph) {
            graph.validate();
            this.graph = graph;
            int n = graph.size();
            outgoingRow = new int[n + 1];
            destinations = new int[graph.pre().length];
            for (int pre : graph.pre()) outgoingRow[pre + 1]++;
            for (int i = 1; i <= n; i++) outgoingRow[i] += outgoingRow[i - 1];
            int[] cursor = outgoingRow.clone();
            for (int post = 0; post < n; post++)
                for (int edge = (int) graph.row()[post]; edge < (int) graph.row()[post + 1]; edge++)
                    destinations[cursor[graph.pre()[edge]]++] = post;
        }
    }

    static final class SparseNetwork implements AutoCloseable {
        final Topology topology;
        final RingsBenchmark.Graph graph;
        final double[] phase, control;
        final int[] lastStep, marked, candidates;
        double[] activity, next;
        int[] active, nextActive;
        int activeCount, candidateCount, step;
        boolean key;
        final int stimulus, threads;
        final Phaser barrier;
        final Thread[] workers;
        volatile Throwable failure;
        volatile boolean stopping;

        SparseNetwork(Topology topology, int stimulus, int length, int threads) {
            this.topology = topology; graph = topology.graph;
            this.stimulus = stimulus; this.threads = threads;
            int n = graph.size();
            phase = new double[n]; lastStep = new int[n]; marked = new int[n];
            candidates = new int[n]; activity = new double[n]; next = new double[n];
            active = new int[n]; nextActive = new int[n];
            control = new double[length + 1];
            for (int t = 1; t < control.length; t++) control[t] = zeroStep(control[t - 1]);
            barrier = threads == 1 ? null : new Phaser(threads + 1);
            workers = new Thread[threads == 1 ? 0 : threads];
            try {
                for (int i = 0; i < workers.length; i++) {
                    final int id = i;
                    workers[i] = Thread.ofPlatform().name("chaconne-sparse-" + i)
                            .unstarted(() -> worker(id));
                    workers[i].start();
                }
            } catch (RuntimeException | Error error) { close(); throw error; }
        }

        void worker(int id) {
            try {
                while (barrier.arriveAndAwaitAdvance() >= 0) {
                    if (stopping) return;
                    int first = (int) ((long) candidateCount * id / threads);
                    int stop = (int) ((long) candidateCount * (id + 1) / threads);
                    compute(first, stop);
                    if (barrier.arriveAndAwaitAdvance() < 0) return;
                }
            } catch (Throwable error) { failure = error; barrier.forceTermination(); }
        }

        void add(int node) {
            if (marked[node] != step) {
                marked[node] = step;
                candidates[candidateCount++] = node;
            }
        }

        void step(boolean down) {
            if (failure != null || stopping) throw new IllegalStateException("Worker unavailable", failure);
            step++;
            if (step >= control.length) throw new IllegalStateException("Timeline exhausted");
            key = down;
            candidateCount = 0;
            if (down) add(stimulus);
            // Even subnormal positive activity contributes; no magnitude cutoff.
            for (int a = 0; a < activeCount; a++) {
                int pre = active[a];
                for (int edge = topology.outgoingRow[pre]; edge < topology.outgoingRow[pre + 1]; edge++)
                    add(topology.destinations[edge]);
            }
            // Large frontiers use CSR row order for locality, with linear work.
            // This changes only independent destination processing order.
            if (candidateCount > graph.size() / 16) {
                int count = 0;
                for (int i = 0; i < marked.length; i++)
                    if (marked[i] == step) candidates[count++] = i;
            }
            if (candidateCount > 0) {
                if (workers.length == 0 || candidateCount < 256) compute(0, candidateCount);
                else {
                    barrier.arriveAndAwaitAdvance();
                    barrier.arriveAndAwaitAdvance();
                    if (failure != null || barrier.isTerminated())
                        throw new IllegalStateException("Worker failed", failure);
                }
            }
            int newCount = 0;
            for (int k = 0; k < candidateCount; k++) {
                int i = candidates[k];
                if (next[i] > 0) nextActive[newCount++] = i;
            }
            // All row reductions finished before any old values are cleared.
            for (int k = 0; k < activeCount; k++) activity[active[k]] = 0.0;
            double[] old = activity; activity = next; next = old;
            int[] oldActive = active; active = nextActive; nextActive = oldActive;
            activeCount = newCount;
        }

        void compute(int first, int stop) {
            for (int k = first; k < stop; k++) {
                int i = candidates[k];
                // Restore EVERY skipped zero-input update in original arithmetic.
                boolean onControl = lastStep[i] == 0 || Double.doubleToRawLongBits(phase[i])
                        == Double.doubleToRawLongBits(control[lastStep[i]]);
                double p = onControl ? control[step - 1] : phase[i];
                if (!onControl)
                    for (int t = lastStep[i] + 1; t < step; t++) p = zeroStep(p);
                double sum = 0;
                for (int edge = (int) graph.row()[i]; edge < (int) graph.row()[i + 1]; edge++)
                    sum += graph.weight()[edge] * activity[graph.pre()[edge]];
                double u = (key && i == stimulus ? 1.0 : 0.0) + CONFIG.coupling() * sum;
                double raw = p + INCREMENT + CONFIG.gain() * u * CONFIG.dt();
                if (!Double.isFinite(u) || !Double.isFinite(raw))
                    throw new ArithmeticException("Nonfinite sparse step");
                double a = RingsBenchmark.wrap(raw);
                phase[i] = a; lastStep[i] = step;
                next[i] = RingsBenchmark.gatedValue(u > 0 ? u : 0.0, Math.abs(a) <= CONFIG.width());
            }
        }

        // Inspection never modifies lazy state, so tests exercise actual skipped gaps.
        double phaseAtCurrentStep(int node) {
            if (lastStep[node] == 0 || Double.doubleToRawLongBits(phase[node])
                    == Double.doubleToRawLongBits(control[lastStep[node]])) return control[step];
            double value = phase[node];
            for (int t = lastStep[node] + 1; t <= step; t++) value = zeroStep(value);
            return value;
        }

        @Override public void close() {
            stopping = true;
            if (barrier != null) barrier.forceTermination();
            boolean interrupted = false;
            for (Thread thread : workers) {
                if (thread == null || thread == Thread.currentThread()) continue;
                while (thread.isAlive()) try { thread.join(); }
                catch (InterruptedException error) { interrupted = true; }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    static void runSparse(RingsBenchmark.Graph graph, int stimulus, boolean[] keys,
                          int threads, Stream sink) throws Exception {
        int last = lastKey(keys);
        if (last == 0) return;
        try (SparseNetwork net = new SparseNetwork(topology(graph), stimulus, keys.length, threads)) {
            for (int i = 0; i < keys.length; i++) {
                net.step(keys[i]);
                emit(sink, i + 1, net.activity, net.active, net.activeCount);
                if (i + 1 >= last && net.activeCount == 0) break;
            }
        }
    }

    static void require(boolean ok, String message) { if (!ok) throw new AssertionError(message); }
    static void compare(RingsBenchmark.Graph graph, int stimulus, boolean[] keys, int threads,
                        boolean phasesEveryStep) throws Exception {
        double[] input = new double[graph.size()];
        try (RingsBenchmark.Network dense = new RingsBenchmark.Network(graph, CONFIG, threads, "platform");
             SparseNetwork sparse = new SparseNetwork(new Topology(graph), stimulus, keys.length, threads)) {
            for (int t = 0; t < keys.length; t++) {
                input[stimulus] = keys[t] ? 1 : 0;
                dense.step(input); sparse.step(keys[t]);
                for (int i = 0; i < graph.size(); i++) {
                    if (Double.doubleToRawLongBits(dense.activity[i])
                            != Double.doubleToRawLongBits(sparse.activity[i]))
                        throw new AssertionError("Activity mismatch step=" + (t + 1) + " node=" + i);
                    if (phasesEveryStep || t == keys.length - 1 || t % 97 == 0) {
                        double phase = sparse.phaseAtCurrentStep(i);
                        if (Double.doubleToRawLongBits(dense.phase[i]) != Double.doubleToRawLongBits(phase))
                            throw new AssertionError("Phase mismatch step=" + (t + 1) + " node=" + i);
                        if (dense.gate[i] != (Math.abs(phase) <= CONFIG.width() ? 1 : 0))
                            throw new AssertionError("Gate mismatch step=" + (t + 1) + " node=" + i);
                    }
                }
            }
        }
    }

    static boolean[] fixture(int steps) {
        boolean[] keys = new boolean[steps];
        for (int t = 0; t < steps; t++)
            keys[t] = t >= 137 && t < 238 || t >= 401 && t < 474 || t >= 612 && t < 751;
        return keys;
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args[0].equals("--self-test")) {
            int n = 17;
            long[] ids = new long[n], row = new long[n + 1];
            int[] pre = new int[n * 7]; double[] weight = new double[n * 7];
            Random random = new Random(971);
            for (int i = 0; i < n; i++) {
                ids[i] = i; row[i] = i * 7;
                for (int j = 0; j < 7; j++) {
                    pre[i * 7 + j] = random.nextInt(n);
                    weight[i * 7 + j] = j == 6 ? Double.MIN_VALUE : (random.nextDouble() - .5) * .3;
                }
            }
            row[n] = pre.length;
            RingsBenchmark.Graph graph = new RingsBenchmark.Graph(ids, row, pre, weight);
            for (int threads : new int[]{1, 4}) compare(graph, 0, fixture(1001), threads, true);
            RingsBenchmark.Graph cancellation = new RingsBenchmark.Graph(new long[]{1,2,3,4},
                    new long[]{0,0,2,3,5}, new int[]{0,0,1,0,2},
                    new double[]{1,-1,-.5,Double.MIN_VALUE,1});
            compare(cancellation, 0, fixture(1001), 4, true);
            System.out.println("{\"sparse_self_test\":true,\"cases\":\"signed_subnormal_cancellation_silence_resumption\",\"threads\":[1,4]}");
        } else if (args[0].equals("--validate-graph") && args.length >= 3) {
            RingsBenchmark.Graph graph = RingsBenchmark.loadGraph(Path.of(args[1]));
            long id = Long.parseUnsignedLong(args[2]);
            int stimulus = -1;
            for (int i = 0; i < graph.size(); i++) if (graph.ids()[i] == id) stimulus = i;
            require(stimulus >= 0, "Unknown stimulus");
            int steps = args.length >= 4 ? Integer.parseInt(args[3]) : 1001;
            long started = System.nanoTime();
            compare(graph, stimulus, fixture(steps), 4, false);
            System.out.printf(java.util.Locale.ROOT,
                    "{\"full_graph_sparse_validation\":true,\"nodes\":%d,\"steps\":%d,\"seconds\":%.6f}%n",
                    graph.size(), steps, (System.nanoTime() - started) * 1e-9);
        } else throw new IllegalArgumentException("--self-test | --validate-graph GRAPH STIMULUS [STEPS]");
    }
}

