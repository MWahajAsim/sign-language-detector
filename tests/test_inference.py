"""Integration tests for the unified inference engine."""

from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceEngine, InferenceOutput
from src.preprocessing import LandmarkNormalizer
from src.synthetic_data import get_base_hand_pose


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


def test_model_confidence_above_threshold():
    """Verify that recognized hand signs achieve high confidence (70-95%+)."""
    engine = InferenceEngine()

    test_letters = ["A", "B", "C", "D", "L", "O", "W", "Y"]
    for letter in test_letters:
        pose = get_base_hand_pose(letter)
        features = LandmarkNormalizer.normalize(pose)
        pred = engine.classifier.predict(features, confidence_threshold=0.65)

        assert pred.label == letter
        assert pred.confidence >= 0.70, f"Letter {letter} confidence {pred.confidence} was below 70%"
        assert pred.is_valid is True

    engine.close()


def test_handedness_reflection_invariance():
    """Verify left-handed (mirrored) input is correctly recognized."""
    engine = InferenceEngine()

    for letter in ["A", "B", "L", "Y"]:
        # Right hand pose
        pose_right = get_base_hand_pose(letter)
        # Mirrored (left hand) pose
        pose_left = pose_right.copy()
        pose_left[:, 0] = -pose_left[:, 0]

        feat_right = LandmarkNormalizer.normalize(pose_right)
        feat_left = LandmarkNormalizer.normalize(pose_left)

        pred_right = engine.classifier.predict(feat_right)
        pred_left = engine.classifier.predict(feat_left)

        assert pred_right.label == letter
        assert pred_left.label == letter
        assert pred_left.confidence >= 0.70

    engine.close()


def test_inference_engine_video_file_processing():
    """Verify video file processing produces output video and transcribes text."""
    engine = InferenceEngine()

    # Create a small dummy video file
    temp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_in, fourcc, 10.0, (640, 480))
    for i in range(15):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        out_video_path, sentence, count = engine.process_video_file(
            video_path=temp_in,
            output_path=temp_out,
            draw_landmarks=True,
            mirror=False,
        )
        assert Path(out_video_path).exists()
        assert Path(out_video_path).stat().st_size > 0
        assert count == 15
        assert isinstance(sentence, str)
    finally:
        Path(temp_in).unlink(missing_ok=True)
        Path(temp_out).unlink(missing_ok=True)
        engine.close()
