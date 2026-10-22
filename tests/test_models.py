"""Unit tests for model loading and classification inference."""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.models import SignClassifier


def test_sign_classifier_load_and_predict():
    """Verify serialized model loads correctly and outputs calibrated probabilities."""
    model_dir = DEFAULT_CONFIG.paths.models_dir
    assert (model_dir / "classifier.joblib").exists(), "Model artifact must exist before running test."

    classifier = SignClassifier.load(model_dir)
    assert len(classifier.class_names) == 24
    assert classifier.estimator is not None

    # Dummy feature vector (63 zeros)
    dummy_feat = np.zeros(63, dtype=np.float32)
    result = classifier.predict(dummy_feat, confidence_threshold=0.70)

    assert result.label in classifier.class_names
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 24

    # Probabilities should sum to approximately 1.0
    total_prob = sum(result.probabilities.values())
    np.testing.assert_allclose(total_prob, 1.0, atol=1e-3)
