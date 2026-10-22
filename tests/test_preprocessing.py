"""Unit tests for landmark normalization and feature preprocessing."""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import LandmarkNormalizer


def test_landmark_normalization_shape_and_bounds():
    """Verify output shape is (63,) and coordinates lie within unit sphere."""
    # Create arbitrary 21 3D coordinates
    rng = np.random.RandomState(42)
    raw_landmarks = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    normalized = LandmarkNormalizer.normalize(raw_landmarks)

    assert normalized.shape == (63,)
    assert normalized.dtype == np.float32

    # Reshape back to 21x3 to verify properties
    coords = normalized.reshape(21, 3)

    # Wrist (point 0) must be exactly at origin
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0], atol=1e-6)

    # Maximum distance from wrist must be <= 1.0 (within unit sphere)
    distances = np.linalg.norm(coords, axis=1)
    assert np.max(distances) <= 1.0 + 1e-5


def test_landmark_translation_invariance():
    """Verify shifting hand across camera frame produces identical features."""
    rng = np.random.RandomState(101)
    base_landmarks = rng.uniform(0.2, 0.6, size=(21, 3)).astype(np.float32)

    # Shift hand arbitrarily in (x, y, z)
    translation_vector = np.array([5.5, -3.2, 8.0], dtype=np.float32)
    shifted_landmarks = base_landmarks + translation_vector

    norm_base = LandmarkNormalizer.normalize(base_landmarks)
    norm_shifted = LandmarkNormalizer.normalize(shifted_landmarks)

    # Normalized vectors must be identical despite translation
    np.testing.assert_allclose(norm_base, norm_shifted, atol=1e-5)


def test_landmark_scale_invariance():
    """Verify hand moving closer/further from camera produces identical features."""
    rng = np.random.RandomState(202)
    base_landmarks = rng.uniform(0.1, 0.5, size=(21, 3)).astype(np.float32)

    # Scale hand by 3.5x
    scaled_landmarks = base_landmarks * 3.5

    norm_base = LandmarkNormalizer.normalize(base_landmarks)
    norm_scaled = LandmarkNormalizer.normalize(scaled_landmarks)

    # Normalized vectors must be identical despite scaling
    np.testing.assert_allclose(norm_base, norm_scaled, atol=1e-5)


def test_landmark_validation():
    """Verify validator catches NaNs, Infs, and incorrect dimensions."""
    valid_features = np.zeros(63, dtype=np.float32)
    assert LandmarkNormalizer.validate(valid_features) is True

    # Bad shape
    assert LandmarkNormalizer.validate(np.zeros(62)) is False
    assert LandmarkNormalizer.validate(np.zeros((21, 3))) is False

    # NaNs and Infs
    nan_features = valid_features.copy()
    nan_features[5] = np.nan
    assert LandmarkNormalizer.validate(nan_features) is False

    inf_features = valid_features.copy()
    inf_features[12] = np.inf
    assert LandmarkNormalizer.validate(inf_features) is False


def test_landmark_augmentation():
    """Verify augmentation preserves output shape and keeps wrist at origin."""
    rng = np.random.RandomState(303)
    base_features = np.zeros(63, dtype=np.float32)
    base_features[3:] = rng.uniform(-0.5, 0.5, size=60)

    augmented = LandmarkNormalizer.augment(base_features, rng=rng)

    assert augmented.shape == (63,)
    coords = augmented.reshape(21, 3)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0], atol=1e-5)
