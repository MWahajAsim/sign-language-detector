"""Unit tests for temporal smoothing, majority voting, and duplicate suppression."""

from pathlib import Path
import sys
import time
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import PredictionResult
from src.prediction_smoothing import (
    ConfidenceFilter,
    DuplicateSuppressor,
    PredictionSmoothingEngine,
    TemporalFilter,
)


def test_confidence_filter():
    """Verify thresholding marks low-confidence predictions as UNCERTAIN."""
    filter_engine = ConfidenceFilter(threshold=0.75)

    high_conf_pred = PredictionResult(
        label="B", confidence=0.88, probabilities={"B": 0.88, "A": 0.12}, is_valid=True
    )
    label, conf, valid = filter_engine.check(high_conf_pred)
    assert valid is True
    assert label == "B"
    assert conf == 0.88

    low_conf_pred = PredictionResult(
        label="B", confidence=0.62, probabilities={"B": 0.62, "A": 0.38}, is_valid=False
    )
    label, conf, valid = filter_engine.check(low_conf_pred)
    assert valid is False
    assert label == "UNCERTAIN"
    assert conf == 0.62


def test_temporal_filter_majority_voting():
    """Verify sliding window stabilizes predictions and rejects intermittent noise.

    With a padded-window denominator, consensus requires ceil(window*ratio)
    occurrences of the same label within the fixed-size window.
    """
    temp_filter = TemporalFilter(window_size=5, majority_ratio=0.60)

    # Required consensus count: ceil(5 * 0.60) = 3
    # Window partially filled with only 2 'A's -> below threshold, must return None
    assert temp_filter.add("A") is None
    assert temp_filter.add("A") is None

    # Noise frame occurs (still only 2 'A's)
    assert temp_filter.add("B") is None

    # Third 'A' reaches the required count of 3 out of the padded window of 5
    assert temp_filter.add("A") == "A"

    # Sustained consensus keeps returning 'A'
    assert temp_filter.add("A") == "A"


def test_temporal_filter_noise_rejection():
    """Intermittent single-frame noise must never produce a consensus."""
    temp_filter = TemporalFilter(window_size=5, majority_ratio=0.80)  # need 4/5

    seq = ["A", "B", "A", "B", "A", "B", "A", "B"]
    for label in seq:
        assert temp_filter.add(label) is None


def test_temporal_filter_uncertain_erodes_consensus():
    """UNCERTAIN / None frames inject neutral markers that erode consensus."""
    tf = TemporalFilter(window_size=5, majority_ratio=0.60)  # need 3/5

    # Build consensus on 'A'
    assert tf.add("A") is None
    assert tf.add("A") is None
    assert tf.add("A") == "A"   # 3 A's -> consensus

    # One UNCERTAIN frame: buffer [A,A,A,N] (padded view) -> '__NONE__' ties with
    # 'A' at 3 and becomes most_common, so consensus drops immediately.
    assert tf.add("UNCERTAIN") is None

    # More neutral frames keep the window eroded; 'A' falls below the required count
    assert tf.add(None) is None      # buffer [A,A,A,N,N] -> A=3 but NONE wins tie
    assert tf.add(None) is None      # buffer [A,A,N,N,N] -> no label consensus


def test_temporal_filter_reset_clears_state():
    """After reset(), the window must behave like a fresh filter."""
    tf = TemporalFilter(window_size=5, majority_ratio=0.60)
    tf.add("A")
    tf.add("A")
    tf.add("A")
    tf.reset()
    assert len(tf.buffer) == 0
    assert tf.add("B") is None  # cannot inherit prior consensus


class FakeClock:
    """Deterministic injectable clock for debounce tests (no wall-clock sleeps)."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_duplicate_suppressor_debounce():
    """Verify repeating signs are suppressed until cooldown period passes."""
    clock = FakeClock()
    suppressor = DuplicateSuppressor(cooldown_seconds=0.30, clock=clock)

    # First emission of 'H' accepted immediately
    assert suppressor.process("H") == "H"

    # Immediate second emission of 'H' suppressed
    assert suppressor.process("H") is None

    # Different character 'E' accepted immediately without waiting
    assert suppressor.process("E") == "E"

    # Immediate second emission of 'E' suppressed
    assert suppressor.process("E") is None

    # Advance the virtual clock past the cooldown (deterministic, no sleep)
    clock.advance(0.35)
    # Now second emission of 'E' is accepted
    assert suppressor.process("E") == "E"

    # Just under the cooldown again -> suppressed
    clock.advance(0.25)
    assert suppressor.process("E") is None

    # Past the cooldown -> accepted
    clock.advance(0.10)
    assert suppressor.process("E") == "E"


def test_smoothing_engine_end_to_end():
    """Verify integrated smoothing engine handles hand absence and stabilization."""
    engine = PredictionSmoothingEngine(
        confidence_threshold=0.70,
        window_size=5,
        majority_ratio=0.60,
        debounce_cooldown=0.50,
    )

    # 1. No hand
    disp, conf, char, status = engine.process(None)
    assert disp == "NO_HAND"
    assert char is None

    # 2. Feeding consistent 'C' predictions
    valid_pred = PredictionResult("C", 0.90, {"C": 0.90}, True)

    emitted = []
    for _ in range(5):
        _, _, char, _ = engine.process(valid_pred)
        if char:
            emitted.append(char)

    # Exactly one 'C' should be emitted
    assert emitted == ["C"]
