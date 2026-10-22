"""Temporal smoothing, confidence filtering, and duplicate suppression engine.

Implements sliding window majority voting and debounce cooldowns to eliminate
per-frame prediction jitter and prevent runaway duplicate letter generation.
"""

from collections import Counter, deque
import time
from typing import Deque, Optional, Tuple

from src.models import PredictionResult


class ConfidenceFilter:
    """Filters out predictions that do not meet the minimum confidence threshold."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def check(self, prediction: PredictionResult) -> Tuple[Optional[str], float, bool]:
        """Evaluate if prediction passes threshold.

        Returns:
            Tuple of (label or 'UNCERTAIN', confidence, is_valid)
        """
        if prediction.confidence >= self.threshold:
            return prediction.label, prediction.confidence, True
        return "UNCERTAIN", prediction.confidence, False


class TemporalFilter:
    """Sliding window majority voting buffer."""

    def __init__(self, window_size: int = 7, majority_ratio: float = 0.70):
        self.window_size = window_size
        self.majority_ratio = majority_ratio
        self.buffer: Deque[str] = deque(maxlen=window_size)

    def add(self, label: Optional[str]) -> Optional[str]:
        """Push label to sliding window and return stable label if consensus is reached.

        Args:
            label: Detected label string, 'UNCERTAIN', or None (hand absent).

        Returns:
            Stable label string if agreement >= majority_ratio, else None.
        """
        if label is None or label == "UNCERTAIN":
            # If hand is absent or uncertain, clear or insert marker
            self.buffer.append("__NONE__")
            return None

        self.buffer.append(label)

        # Check if window is full enough
        if len(self.buffer) < self.window_size:
            return None

        # Count frequencies
        counts = Counter(self.buffer)
        most_common_label, count = counts.most_common(1)[0]

        if most_common_label == "__NONE__":
            return None

        # Check consensus ratio
        required_count = int(self.window_size * self.majority_ratio)
        if count >= required_count:
            return most_common_label

        return None

    def reset(self) -> None:
        """Clear the sliding window buffer."""
        self.buffer.clear()


class DuplicateSuppressor:
    """Debounces consecutively emitted characters using a cooldown interval."""

    def __init__(self, cooldown_seconds: float = 0.80):
        self.cooldown_seconds = cooldown_seconds
        self.last_emitted_label: Optional[str] = None
        self.last_emitted_time: float = 0.0

    def process(self, stable_label: Optional[str]) -> Optional[str]:
        """Check if stable label can be emitted as new text.

        Args:
            stable_label: Consolidated label from temporal filter.

        Returns:
            Label to append to text, or None if suppressed by cooldown.
        """
        if stable_label is None:
            return None

        current_time = time.time()
        elapsed = current_time - self.last_emitted_time

        # If it's a different letter than the last one, accept immediately
        if stable_label != self.last_emitted_label:
            self.last_emitted_label = stable_label
            self.last_emitted_time = current_time
            return stable_label

        # If it's the same letter, require debounce cooldown to elapse
        if elapsed >= self.cooldown_seconds:
            self.last_emitted_time = current_time
            return stable_label

        # Suppressed due to cooldown
        return None

    def reset(self) -> None:
        """Reset emission history."""
        self.last_emitted_label = None
        self.last_emitted_time = 0.0


class PredictionSmoothingEngine:
    """Unified smoothing engine orchestrating confidence, temporal, and debounce logic."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        window_size: int = 7,
        majority_ratio: float = 0.70,
        debounce_cooldown: float = 0.80,
    ):
        self.confidence_filter = ConfidenceFilter(confidence_threshold)
        self.temporal_filter = TemporalFilter(window_size, majority_ratio)
        self.duplicate_suppressor = DuplicateSuppressor(debounce_cooldown)

    def process(
        self, prediction: Optional[PredictionResult]
    ) -> Tuple[str, float, Optional[str], str]:
        """Process a raw prediction through the smoothing pipeline.

        Args:
            prediction: Raw model prediction, or None if no hand detected.

        Returns:
            Tuple of:
                - display_label: Label to show on HUD ('A', 'UNCERTAIN', 'NO_HAND').
                - display_confidence: Confidence score for HUD.
                - committed_character: Character to append to text box (or None).
                - system_status: High-level status string ('DETECTING', 'STABLE', etc.).
        """
        if prediction is None:
            self.temporal_filter.reset()
            return "NO_HAND", 0.0, None, "No Hand Detected"

        filtered_label, conf, is_valid = self.confidence_filter.check(prediction)

        if not is_valid:
            self.temporal_filter.add("UNCERTAIN")
            return "UNCERTAIN", conf, None, "Uncertain Prediction"

        # Pass through temporal sliding window
        stable_label = self.temporal_filter.add(filtered_label)

        if stable_label is None:
            return filtered_label, conf, None, "Stabilizing..."

        # Pass through duplicate debounce suppressor
        emitted_char = self.duplicate_suppressor.process(stable_label)

        status = f"Accepted '{stable_label}'" if emitted_char else f"Holding '{stable_label}'"
        return stable_label, conf, emitted_char, status

    def reset(self) -> None:
        """Clear all temporal smoothing states."""
        self.temporal_filter.reset()
        self.duplicate_suppressor.reset()
