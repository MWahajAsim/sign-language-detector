"""Unified real-time inference engine.

Integrates frame preprocessing, MediaPipe hand landmark extraction,
coordinate normalization, model inference, temporal smoothing,
and text accumulation into a clean, reusable API.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG, AppConfig
from src.feature_extraction import HandLandmarkDetector
from src.models import PredictionResult, SignClassifier
from src.prediction_smoothing import PredictionSmoothingEngine
from src.preprocessing import FrameProcessor, LandmarkNormalizer
from src.text_builder import TextAccumulator


@dataclass
class InferenceOutput:
    """Comprehensive output payload for every processed frame or image."""

    annotated_frame: np.ndarray
    predicted_label: str
    confidence: float
    accumulated_text: str
    status_message: str
    hand_detected: bool
    probabilities: Dict[str, float]


class InferenceEngine:
    """High-level inference orchestrator for webcam streams and static images."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.frame_processor = FrameProcessor(
            target_width=self.config.inference.frame_width,
            target_height=self.config.inference.frame_height,
        )
        self.detector = HandLandmarkDetector(
            static_image_mode=False,
            max_num_hands=self.config.inference.max_num_hands,
            min_detection_confidence=self.config.inference.min_detection_confidence,
            min_tracking_confidence=self.config.inference.min_tracking_confidence,
        )
        self.normalizer = LandmarkNormalizer()
        self.classifier = SignClassifier.load(self.config.paths.models_dir)
        self.smoothing_engine = PredictionSmoothingEngine(
            confidence_threshold=self.config.inference.confidence_threshold,
            window_size=self.config.inference.smoothing_window_size,
            majority_ratio=self.config.inference.majority_threshold_ratio,
            debounce_cooldown=self.config.inference.debounce_cooldown_seconds,
        )
        self.text_accumulator = TextAccumulator()

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        draw_landmarks: bool = True,
        mirror: bool = True,
    ) -> InferenceOutput:
        """Process a live video frame and update inference and text states.

        Args:
            bgr_frame: Input frame from OpenCV VideoCapture.
            draw_landmarks: Whether to render landmark skeleton onto output image.
            mirror: Whether to horizontally flip the input frame for natural mirror view.

        Returns:
            InferenceOutput containing annotated frame, current prediction, confidence,
            accumulated text, and system status.
        """
        if bgr_frame is None:
            raise ValueError("Input frame is None.")

        # 1. Preprocess Frame (resize and RGB conversion)
        rgb_frame, display_frame = self.frame_processor.process(bgr_frame, mirror=mirror)

        # 2. Extract Landmarks via MediaPipe
        landmarks, landmark_proto = self.detector.extract(rgb_frame)

        if landmarks is None:
            # No hand in frame
            disp_label, conf, _, status = self.smoothing_engine.process(None)
            annotated = self._render_hud(
                display_frame,
                label=disp_label,
                confidence=conf,
                status=status,
                hand_detected=False
            )
            return InferenceOutput(
                annotated_frame=annotated,
                predicted_label="None",
                confidence=0.0,
                accumulated_text=self.text_accumulator.get_text(),
                status_message=status,
                hand_detected=False,
                probabilities={},
            )

        # 3. Coordinate Normalization (Wrist-centering and scale invariance)
        features_63 = self.normalizer.normalize(landmarks)

        # 4. Model Prediction
        prediction: PredictionResult = self.classifier.predict(
            features_63,
            confidence_threshold=self.config.inference.confidence_threshold
        )

        # 5. Temporal Smoothing & Duplicate Suppression
        disp_label, conf, emitted_char, status = self.smoothing_engine.process(prediction)

        # 6. Append Emitted Character to Sentence Buffer
        if emitted_char is not None:
            self.text_accumulator.append(emitted_char)

        # 7. Render Visual Annotations
        annotated = display_frame.copy()
        if draw_landmarks and landmark_proto is not None:
            annotated = self.detector.draw_landmarks(annotated, landmark_proto)

        annotated = self._render_hud(
            annotated,
            label=disp_label,
            confidence=conf,
            status=status,
            hand_detected=True
        )

        return InferenceOutput(
            annotated_frame=annotated,
            predicted_label=disp_label,
            confidence=conf,
            accumulated_text=self.text_accumulator.get_text(),
            status_message=status,
            hand_detected=True,
            probabilities=prediction.probabilities,
        )

    def process_image(
        self,
        bgr_image: np.ndarray,
        draw_landmarks: bool = True
    ) -> InferenceOutput:
        """Process a static image (without temporal sliding window).

        Args:
            bgr_image: Input image array.
            draw_landmarks: Whether to render landmark skeleton onto image.

        Returns:
            InferenceOutput for the single image.
        """
        if bgr_image is None:
            raise ValueError("Input image is None.")

        rgb_frame, display_frame = self.frame_processor.process(bgr_image, mirror=False)
        landmarks, landmark_proto = self.detector.extract(rgb_frame)

        if landmarks is None:
            annotated = self._render_hud(
                display_frame,
                label="NO_HAND",
                confidence=0.0,
                status="No hand detected in image",
                hand_detected=False
            )
            return InferenceOutput(
                annotated_frame=annotated,
                predicted_label="NO_HAND",
                confidence=0.0,
                accumulated_text=self.text_accumulator.get_text(),
                status_message="No hand detected in image",
                hand_detected=False,
                probabilities={},
            )

        features_63 = self.normalizer.normalize(landmarks)
        pred = self.classifier.predict(
            features_63,
            confidence_threshold=self.config.inference.confidence_threshold
        )

        annotated = display_frame.copy()
        if draw_landmarks and landmark_proto is not None:
            annotated = self.detector.draw_landmarks(annotated, landmark_proto)

        status = f"Detected '{pred.label}'" if pred.is_valid else "Uncertain Prediction"
        annotated = self._render_hud(
            annotated,
            label=pred.label if pred.is_valid else "UNCERTAIN",
            confidence=pred.confidence,
            status=status,
            hand_detected=True
        )

        return InferenceOutput(
            annotated_frame=annotated,
            predicted_label=pred.label if pred.is_valid else "UNCERTAIN",
            confidence=pred.confidence,
            accumulated_text=self.text_accumulator.get_text(),
            status_message=status,
            hand_detected=True,
            probabilities=pred.probabilities,
        )

    def process_video_file(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        draw_landmarks: bool = True,
        mirror: bool = False,
    ) -> Tuple[str, str, int]:
        """Process an entire video file, annotating frames and accumulating text.

        Args:
            video_path: Path to source video file (MP4, AVI, MOV, etc.).
            output_path: Optional path for annotated output video. If None, saves to temp file.
            draw_landmarks: Whether to render landmark skeleton.
            mirror: Whether to horizontally flip video frames.

        Returns:
            Tuple of (output_video_path, accumulated_text, total_frames_processed).
        """
        import tempfile

        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found at: {v_path}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {v_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_w = self.config.inference.frame_width
        target_h = self.config.inference.frame_height

        if output_path is None:
            out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        else:
            out_file = str(output_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_file, fourcc, fps, (target_w, target_h))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                output: InferenceOutput = self.process_frame(
                    frame,
                    draw_landmarks=draw_landmarks,
                    mirror=mirror,
                )
                writer.write(output.annotated_frame)
                frame_count += 1
        finally:
            cap.release()
            writer.release()

        return out_file, self.text_accumulator.get_text(), frame_count

    def _render_hud(
        self,
        image: np.ndarray,
        label: str,
        confidence: float,
        status: str,
        hand_detected: bool,
    ) -> np.ndarray:
        """Render high-contrast Heads-Up Display (HUD) overlay on the image."""
        img = image.copy()
        h, w = img.shape[:2]

        # Semi-transparent top banner
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

        if not hand_detected:
            color = (80, 80, 240)  # Red/Amber
            text = "NO HAND DETECTED"
        elif label == "UNCERTAIN":
            color = (0, 215, 255)  # Yellow
            text = f"UNCERTAIN ({confidence * 100:.1f}%)"
        else:
            color = (50, 220, 50)  # Green
            text = f"SIGN: {label} ({confidence * 100:.1f}%)"

        cv2.putText(
            img, text, (15, 32),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA
        )

        # Status subtitle
        cv2.putText(
            img, f"Status: {status}", (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        return img

    def add_space(self) -> str:
        """Append a whitespace delimiter to the accumulated text."""
        return self.text_accumulator.add_space()

    def backspace(self) -> str:
        """Delete the last character from the accumulated text."""
        return self.text_accumulator.backspace()

    def clear_text(self) -> str:
        """Clear the accumulated text and reset temporal smoothing."""
        self.smoothing_engine.reset()
        return self.text_accumulator.clear()

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update runtime confidence threshold."""
        self.config.inference.confidence_threshold = threshold
        self.smoothing_engine.confidence_filter.threshold = threshold

    def set_smoothing_window(self, window_size: int) -> None:
        """Update runtime temporal smoothing window size."""
        self.config.inference.smoothing_window_size = window_size
        self.smoothing_engine.temporal_filter.window_size = window_size

    def close(self) -> None:
        """Release vision and detector resources."""
        self.detector.close()
