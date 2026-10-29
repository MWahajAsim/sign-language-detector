"""Feature extraction module supporting modern and legacy MediaPipe Hand detection.

Extracts 21 3D anatomical hand landmarks from RGB image frames and provides
drawing utilities for visual skeleton diagnostics.
"""

from pathlib import Path
import sys
from typing import Any, List, Optional, Tuple
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG

# Canonical hand skeleton connection pairs (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


class HandLandmarkDetector:
    """Robust 21-landmark hand keypoint extractor compatible with MediaPipe 0.10+ and 1.x."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        model_path: Optional[Path] = None,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.model_path = model_path or (DEFAULT_CONFIG.paths.models_dir / "hand_landmarker.task")
        self._mode = "tasks" if hasattr(mp, "tasks") else "solutions"
        self._init_detector()

    def _ensure_model_asset(self) -> Path:
        """Download official MediaPipe hand_landmarker.task asset if missing."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            print(f"[Detector] Downloading hand_landmarker.task model asset to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, str(self.model_path))
            print("[Detector] Model asset downloaded successfully.")
        return self.model_path

    def _init_detector(self) -> None:
        """Initialize appropriate MediaPipe detector backend."""
        try:
            from mediapipe.tasks.python import vision, BaseOptions
            self._mode = "tasks"

            model_file = self._ensure_model_asset()
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_file)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            return
        except (ImportError, AttributeError):
            pass

        try:
            import mediapipe.solutions.hands as mp_hands
            self._mode = "solutions"
            self.detector = mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except (ImportError, AttributeError) as err:
            raise RuntimeError(f"No compatible MediaPipe backend found: {err}")


    def extract(
        self,
        rgb_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """Process RGB frame and extract 21 3D landmarks for the primary hand.

        Args:
            rgb_frame: RGB image array of shape (H, W, 3) and uint8 dtype.

        Returns:
            Tuple of:
                - landmarks: (21, 3) numpy float32 array, or None if no hand detected.
                - raw_landmarks_data: Data object for visual drawing.
        """
        if self._mode == "tasks":
            frame_data = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_data)
            result = self.detector.detect(mp_image)

            if not result.hand_landmarks or len(result.hand_landmarks) == 0:
                return None, None

            primary_landmarks = result.hand_landmarks[0]
            coords = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(primary_landmarks):
                coords[i, 0] = lm.x
                coords[i, 1] = lm.y
                coords[i, 2] = lm.z

            return coords, coords

        elif self._mode == "solutions":
            results = self.detector.process(rgb_frame)
            if not results.multi_hand_landmarks:
                return None, None

            primary_hand = results.multi_hand_landmarks[0]
            coords = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(primary_hand.landmark):
                coords[i, 0] = lm.x
                coords[i, 1] = lm.y
                coords[i, 2] = lm.z

            return coords, primary_hand

        return None, None

    def draw_landmarks(
        self,
        bgr_image: np.ndarray,
        landmarks_data: Any
    ) -> np.ndarray:
        """Render 21 hand landmarks and connection lines onto BGR image.

        Args:
            bgr_image: OpenCV BGR image array.
            landmarks_data: Array of shape (21, 3) or MediaPipe proto.

        Returns:
            Image array with high-contrast skeleton drawn.
        """
        if landmarks_data is None:
            return bgr_image

        annotated = bgr_image.copy()
        h, w = annotated.shape[:2]

        # Extract (x, y) coordinates into pixel space
        if isinstance(landmarks_data, np.ndarray) and landmarks_data.shape == (21, 3):
            points = [
                (int(landmarks_data[i, 0] * w), int(landmarks_data[i, 1] * h))
                for i in range(21)
            ]
        elif hasattr(landmarks_data, "landmark"):
            points = [
                (int(lm.x * w), int(lm.y * h))
                for lm in landmarks_data.landmark
            ]
        else:
            return annotated

        # Draw bone connection lines
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = points[p1_idx]
            pt2 = points[p2_idx]
            cv2.line(annotated, pt1, pt2, (0, 220, 255), 2, cv2.LINE_AA)

        # Draw landmark joints (red fill with white border)
        for pt in points:
            cv2.circle(annotated, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

        return annotated

    def close(self) -> None:
        """Release detector resources."""
        if hasattr(self, "detector") and self.detector is not None:
            if hasattr(self.detector, "close"):
                self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
