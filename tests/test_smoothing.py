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
    """Verify sliding window stabilizes predictions and rejects intermittent noise."""
    temp_filter = TemporalFilter(window_size=5, majority_ratio=0.60)

    # Window partially filled: must return None
    assert temp_filter.add("A") is None
    assert temp_filter.add("A") is None

    # Noise frame occurs
    assert temp_filter.add("B") is None

    # Consensus reached ('A' appears 4 out of 5 times)
    assert temp_filter.add("A") is None
    result = temp_filter.add("A")
    assert result == "A"


def test_duplicate_suppressor_debounce():
    """Verify repeating signs are suppressed until cooldown period passes."""
    suppressor = DuplicateSuppressor(cooldown_seconds=0.30)

    # First emission of 'H' accepted immediately
    assert suppressor.process("H") == "H"

    # Immediate second emission of 'H' suppressed
    assert suppressor.process("H") is None

    # Different character 'E' accepted immediately without waiting
    assert suppressor.process("E") == "E"

    # Immediate second emission of 'E' suppressed
    assert suppressor.process("E") is None

    # Wait for cooldown to elapse
    time.sleep(0.35)
    # Now second emission of 'E' is accepted
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
