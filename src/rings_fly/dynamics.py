"""Experimental phase-gated dynamics on a sparse directed graph.

This is a new graph architecture inspired by RINGS, not a port of the original
image-grid recurrence and not a biophysical neuron or EEG model. Activity and
drive are dimensionless. ``carrier_hz`` supplies a time scale to the gate; it is
not a claim that an anatomical neuron oscillates at that frequency.

Weights have orientation ``W[postsynaptic, presynaptic]``. Each synchronous step
uses only the previous activity:

    drive = external + coupling * W @ old_activity
    phase = wrap(old_phase + (2*pi*carrier_hz + phase_sensitivity*drive)*dt_s)
    gate = abs(phase) <= half_width
    activity = gate * max(0, drive)

The signed incoming term can inhibit activity; outgoing activity is always
nonnegative. All graph edges have an implicit one-step delay. Changing ``dt_s``
therefore changes both integration resolution and that delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
CountArray = NDArray[np.int64]


def _scalar(value: object, name: str, *, positive: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _nodes(value: ArrayLike, n: int, name: str, *, lower: float | None = None,
           upper: float | None = None) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    try:
        result = np.array(np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)),
                          dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar or broadcast to ({n},)") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if lower is not None and np.any(result < lower):
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and np.any(result > upper):
        raise ValueError(f"{name} must be at most {upper}")
    return result


def _snapshot(array: NDArray) -> NDArray:
    result = array.copy()
    result.flags.writeable = False
    return result


def _wrap(phase: FloatArray) -> FloatArray:
    """Wrap finite angles to [-pi, pi), without adding pi to huge inputs."""
    result = np.remainder(phase, 2.0 * np.pi)
    result[result >= np.pi] -= 2.0 * np.pi
    return result


@dataclass(frozen=True)
class DynamicsConfig:
    """Illustrative defaults; numerical values are validated by the network.

    ``carrier_hz``, ``phase_sensitivity`` and ``half_width`` may be scalars or
    arrays broadcastable to one value per node. The other fields are scalars.
    ``phase_sensitivity=0`` gives the fixed-gate baseline with identical weights
    and recurrence. Sensitivity has units of radians per dimensionless drive
    per second; ``dt_s`` is in seconds.
    """

    dt_s: float = 0.001
    carrier_hz: ArrayLike = 10.0
    phase_sensitivity: ArrayLike = 1.0
    coupling: float = 0.4
    half_width: ArrayLike = np.pi / 2.0


@dataclass(frozen=True)
class StepResult:
    """Independent, read-only state snapshots after one completed step.

    Gates describe the phase condition, including when drive is zero, negative,
    or a node is silenced. They cannot be recovered from ``activity != 0``.
    """

    activity: FloatArray
    gates: BoolArray
    phase: FloatArray


class RingFlyDynamics:
    """One phase-gated rate unit per node, with O(N + nnz) state and storage.

    Only SciPy sparse matrices/arrays are accepted; dense N-by-N conversions are
    never performed. The constructor takes its own CSR copy of the weights and
    parameter arrays. It does not normalize or otherwise rescale weights.

    ``set_silenced(indices)`` replaces the silenced set and immediately clears
    those nodes' stored activity, preventing a one-tick outgoing leak. Silencing
    clamps activity, not phase or the reported phase gate. It persists on reset.
    """

    def __init__(self, weights: sparse.spmatrix, config: DynamicsConfig | None = None,
                 initial_phase: ArrayLike = 0.0):
        if not sparse.issparse(weights):
            raise TypeError("weights must be a SciPy sparse matrix or sparse array")
        if len(weights.shape) != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] < 1:
            raise ValueError("weights must be a nonempty square sparse matrix")
        if np.issubdtype(weights.dtype, np.complexfloating):
            raise ValueError("weights must be real")
        self.n_nodes = int(weights.shape[0])
        self.config = config if config is not None else DynamicsConfig()
        if not isinstance(self.config, DynamicsConfig):
            raise TypeError("config must be a DynamicsConfig")
        self._dt_s = _scalar(self.config.dt_s, "dt_s", positive=True)
        self._coupling = _scalar(self.config.coupling, "coupling")
        self._carrier = _nodes(self.config.carrier_hz, self.n_nodes, "carrier_hz", lower=0.0)
        self._sensitivity = _nodes(self.config.phase_sensitivity, self.n_nodes,
                                   "phase_sensitivity", lower=0.0)
        self._half_width = _nodes(self.config.half_width, self.n_nodes, "half_width",
                                  lower=0.0, upper=np.pi)
        self._initial_phase = _wrap(_nodes(initial_phase, self.n_nodes, "initial_phase"))
        try:
            with np.errstate(over="raise", invalid="raise"):
                self._phase_increment = (2.0 * np.pi * self._carrier) * self._dt_s
                self._weights = sparse.csr_matrix(weights, dtype=np.float64, copy=True)
                self._weights.sum_duplicates()
                self._weights.sort_indices()
                self._weights.eliminate_zeros()
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("weights or phase increment overflowed") from exc
        if not np.all(np.isfinite(self._weights.data)) or not np.all(np.isfinite(self._phase_increment)):
            raise ValueError("weights and phase increment must be finite")
        # Protect the internal copy; callers still own their original matrix.
        for array in (self._weights.data, self._weights.indices, self._weights.indptr):
            array.flags.writeable = False
        self._silenced = np.zeros(self.n_nodes, dtype=bool)
        self.reset()

    @property
    def activity(self) -> FloatArray:
        return _snapshot(self._activity)

    @property
    def phase(self) -> FloatArray:
        return _snapshot(self._phase)

    @property
    def gates(self) -> BoolArray:
        return _snapshot(self._gates)

    @property
    def silenced(self) -> BoolArray:
        return _snapshot(self._silenced)

    def reset(self, initial_phase: ArrayLike | None = None) -> None:
        """Clear activity; restore configured phases, or override for this reset.

        A supplied override does not change the constructor's initial phases.
        The silenced set persists. No transition counts are maintained here.
        """
        phase = self._initial_phase.copy() if initial_phase is None else _wrap(
            _nodes(initial_phase, self.n_nodes, "initial_phase"))
        self._phase = phase
        self._activity = np.zeros(self.n_nodes, dtype=np.float64)
        self._gates = np.abs(phase) <= self._half_width

    def set_silenced(self, indices: Iterable[int] | int) -> None:
        """Replace the silenced indices; pass an empty iterable to clear them."""
        if np.isscalar(indices):
            values = np.asarray([indices])
        elif isinstance(indices, np.ndarray):
            values = indices
        else:
            values = np.asarray(list(indices))
        if values.ndim != 1:
            raise ValueError("silenced indices must be a one-dimensional sequence")
        if values.size and (not np.issubdtype(values.dtype, np.integer)
                            or np.issubdtype(values.dtype, np.bool_)):
            raise ValueError("silenced indices must be integers, not booleans or floats")
        if values.size and (np.any(values < 0) or np.any(values >= self.n_nodes)):
            raise ValueError("silenced index is outside the network")
        new_mask = np.zeros(self.n_nodes, dtype=bool)
        new_mask[values.astype(np.intp, copy=False)] = True
        self._silenced = new_mask
        self._activity[new_mask] = 0.0

    def step(self, external: ArrayLike = 0.0) -> StepResult:
        """Advance one synchronous tick; reject invalid inputs before committing.

        ``external`` is a dimensionless scalar or per-node drive. Signed external
        input is allowed. Overflow raises ``ValueError`` and preserves old state.
        """
        external_array = _nodes(external, self.n_nodes, "external")
        try:
            with np.errstate(over="raise", invalid="raise"):
                drive = external_array + self._coupling * (self._weights @ self._activity)
                phase_raw = self._phase + self._phase_increment + self._sensitivity * drive * self._dt_s
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("non-finite drive or phase during step") from exc
        if not np.all(np.isfinite(drive)) or not np.all(np.isfinite(phase_raw)):
            raise ValueError("non-finite drive or phase during step")
        phase = _wrap(phase_raw)
        gates = np.abs(phase) <= self._half_width
        activity = np.maximum(0.0, drive)
        activity[~gates | self._silenced] = 0.0
        self._phase, self._gates, self._activity = phase, gates, activity
        return StepResult(_snapshot(activity), _snapshot(gates), _snapshot(phase))


@dataclass(frozen=True)
class GateStatistics:
    """Pooled online statistics; probabilities without support are NaN.

    ``steps`` counts samples and ``transitions`` counts adjacent time pairs
    within episodes (the same denominator of opportunities for every node).
    ``valid01`` and ``valid10`` identify probabilities with nonzero denominators.
    Means and duty cycles are NaN when no samples have been observed.
    """

    mean_activity: FloatArray
    duty_cycle: FloatArray
    n00: CountArray
    n01: CountArray
    n10: CountArray
    n11: CountArray
    p01: FloatArray
    p10: FloatArray
    valid01: BoolArray
    valid10: BoolArray
    steps: int
    transitions: int


class GateSummary:
    """O(N) summaries without storing a time-by-node trace.

    Call ``end_episode()`` between independent episodes to prevent counting a
    transition across their boundary. It keeps pooled counts; ``reset()`` clears
    all counts and the pending previous sample. Probabilities are calculated
    from pooled counts, never from an average of per-episode probabilities.
    """

    def __init__(self, n_nodes: int):
        if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
            raise ValueError("n_nodes must be a positive integer")
        self.n_nodes = int(n_nodes)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.transitions = 0
        self._activity_sum = np.zeros(self.n_nodes, dtype=np.float64)
        self._open_count = np.zeros(self.n_nodes, dtype=np.int64)
        self._counts = np.zeros((4, self.n_nodes), dtype=np.int64)
        self._previous: BoolArray | None = None

    def end_episode(self) -> None:
        self._previous = None

    def update(self, result: StepResult) -> None:
        gates = np.asarray(result.gates)
        if gates.shape != (self.n_nodes,) or gates.dtype != np.dtype(bool):
            raise ValueError("gates must be a boolean vector with one value per node")
        activity = _nodes(result.activity, self.n_nodes, "activity", lower=0.0)
        try:
            with np.errstate(over="raise", invalid="raise"):
                new_sum = self._activity_sum + activity
        except FloatingPointError as exc:
            raise ValueError("activity summary overflowed") from exc
        if not np.all(np.isfinite(new_sum)):
            raise ValueError("activity summary overflowed")
        if self._previous is not None:
            prev = self._previous
            self._counts[0] += ~prev & ~gates
            self._counts[1] += ~prev & gates
            self._counts[2] += prev & ~gates
            self._counts[3] += prev & gates
            self.transitions += 1
        self._activity_sum = new_sum
        self._open_count += gates
        self.steps += 1
        self._previous = gates.copy()

    def result(self) -> GateStatistics:
        den01 = self._counts[0] + self._counts[1]
        den10 = self._counts[2] + self._counts[3]
        valid01, valid10 = den01 > 0, den10 > 0
        p01 = np.full(self.n_nodes, np.nan)
        p10 = np.full(self.n_nodes, np.nan)
        np.divide(self._counts[1], den01, out=p01, where=valid01)
        np.divide(self._counts[2], den10, out=p10, where=valid10)
        mean = self._activity_sum / self.steps if self.steps else np.full(self.n_nodes, np.nan)
        duty = self._open_count / self.steps if self.steps else np.full(self.n_nodes, np.nan)
        return GateStatistics(
            _snapshot(mean), _snapshot(duty), *(_snapshot(x) for x in self._counts),
            _snapshot(p01), _snapshot(p10), _snapshot(valid01), _snapshot(valid10),
            self.steps, self.transitions,
        )
