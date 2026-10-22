"""Preprocessing and coordinate normalization module.

Transforms raw MediaPipe 3D hand landmarks into translation-invariant,
scale-invariant 63-dimensional feature vectors suitable for machine learning.
"""

from typing import Optional, Tuple
import numpy as np


class LandmarkNormalizer:
    """Mathematical coordinate normalizer for 21 3D hand landmarks."""

    NUM_LANDMARKS = 21
    NUM_COORDINATES = 3  # (x, y, z)
    FEATURE_DIM = 63     # 21 * 3

    @classmethod
    def normalize(cls, landmarks: np.ndarray) -> np.ndarray:
        """Translate wrist to origin and scale by maximum Euclidean distance.

        Args:
            landmarks: Array of shape (21, 3) or (63,) containing raw coordinates.

        Returns:
            Normalized 1D numpy array of shape (63,) bounded within unit sphere.

        Raises:
            ValueError: If input shape does not match (21, 3) or (63,).
        """
        if landmarks is None:
            raise ValueError("Input landmarks array cannot be None.")

        # Ensure correct shape (21, 3)
        if landmarks.ndim == 1 and landmarks.shape[0] == cls.FEATURE_DIM:
            coords = landmarks.reshape(cls.NUM_LANDMARKS, cls.NUM_COORDINATES).copy()
        elif landmarks.ndim == 2 and landmarks.shape == (cls.NUM_LANDMARKS, cls.NUM_COORDINATES):
            coords = landmarks.copy()
        else:
            raise ValueError(
                f"Expected landmarks of shape (21, 3) or (63,), received {landmarks.shape}"
            )

        # Step 1: Translation Invariance (Wrist Centering)
        # Landmark 0 is the Wrist. Subtract wrist coordinates from all 21 points.
        wrist = coords[0, :].copy()
        centered = coords - wrist  # Wrist becomes (0, 0, 0)

        # Step 2: Scale Invariance (Divide by maximum Euclidean distance from wrist)
        distances = np.linalg.norm(centered, axis=1)  # shape (21,)
        max_dist = np.max(distances)

        epsilon = 1e-7
        if max_dist > epsilon:
            normalized = centered / (max_dist + epsilon)
        else:
            normalized = centered

        # Step 3: Flatten to 63-element feature vector
        return normalized.flatten().astype(np.float32)

    @classmethod
    def validate(cls, features: np.ndarray) -> bool:
        """Verify that a feature vector is valid, finite, and well-formed.

        Args:
            features: 1D feature array.

        Returns:
            True if valid, False otherwise.
        """
        if features is None or not isinstance(features, np.ndarray):
            return False
        if features.shape != (cls.FEATURE_DIM,):
            return False
        if np.isnan(features).any() or np.isinf(features).any():
            return False
        return True

    @classmethod
    def augment(
        cls,
        features_63: np.ndarray,
        jitter_std: float = 0.015,
        scale_range: Tuple[float, float] = (0.92, 1.08),
        rotation_angle_deg: float = 12.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Apply physiologically plausible augmentation to a normalized feature vector.

        Includes small Gaussian jitter, uniform scaling, and small 2D in-plane rotation.

        Args:
            features_63: 63-element normalized array.
            jitter_std: Standard deviation of Gaussian noise added to landmarks.
            scale_range: Min and max scale multiplier.
            rotation_angle_deg: Max degrees of random in-plane rotation.
            rng: Optional numpy RandomState for reproducible randomness.

        Returns:
            Augmented 63-element normalized vector.
        """
        if rng is None:
            rng = np.random.RandomState()

        coords = features_63.reshape(cls.NUM_LANDMARKS, cls.NUM_COORDINATES).copy()

        # 1. Random Gaussian jitter (simulates tremor and sensor noise)
        noise = rng.normal(0.0, jitter_std, size=coords.shape).astype(np.float32)
        # Keep wrist at origin
        noise[0] = 0.0
        coords += noise

        # 2. Random scaling
        scale = rng.uniform(scale_range[0], scale_range[1])
        coords *= scale

        # 3. Random 2D in-plane rotation around Z-axis (roll angle)
        theta = np.radians(rng.uniform(-rotation_angle_deg, rotation_angle_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_matrix = np.array(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=np.float32
        )
        coords = np.dot(coords, rot_matrix.T)

        # Re-normalize to guarantee invariant bounds
        return cls.normalize(coords)


class FrameProcessor:
    """Preprocesses raw camera frames for MediaPipe ingestion."""

    def __init__(self, target_width: int = 640, target_height: int = 480):
        self.target_width = target_width
        self.target_height = target_height

    def process(
        self,
        frame: np.ndarray,
        mirror: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert BGR OpenCV frame to RGB and apply horizontal mirroring.

        Args:
            frame: Raw BGR frame from webcam or file.
            mirror: If True, horizontally flips frame for natural webcam mirror view.

        Returns:
            Tuple of (rgb_frame for MediaPipe, display_frame for visualization).
        """
        import cv2

        if mirror:
            frame = cv2.flip(frame, 1)

        # Resize if dimensions differ
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )

        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, frame
