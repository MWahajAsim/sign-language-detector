"""Integration tests for the unified inference engine."""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceEngine, InferenceOutput


def test_inference_engine_no_hand_frame():
    """Verify inference engine handles blank frames without crashing."""
    engine = InferenceEngine()

    # Create 640x480 black image
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output: InferenceOutput = engine.process_frame(blank_frame)

    assert isinstance(output, InferenceOutput)
    assert output.hand_detected is False
    assert output.predicted_label == "None"
    assert output.confidence == 0.0
    assert output.annotated_frame.shape == (480, 640, 3)

    engine.close()


def test_inference_engine_text_operations():
    """Verify space, backspace, and clear operations through inference engine."""
    engine = InferenceEngine()

    engine.text_accumulator.append("TEST")
    assert engine.text_accumulator.get_text() == "TEST"

    engine.add_space()
    assert engine.text_accumulator.get_text() == "TEST "

    engine.backspace()
    assert engine.text_accumulator.get_text() == "TEST"

    engine.clear_text()
    assert engine.text_accumulator.get_text() == ""

    engine.close()
