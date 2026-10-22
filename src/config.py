"""Configuration module for Sign Language Detection System.

Defines all runtime parameters, dataset paths, model paths, thresholds,
and supported class mappings using strongly typed dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """Project directory and file paths."""

    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = PROJECT_ROOT / "models" / "classifier.joblib"
    class_names_path: Path = PROJECT_ROOT / "models" / "class_names.json"
    metadata_path: Path = PROJECT_ROOT / "models" / "metadata.json"
    docs_dir: Path = PROJECT_ROOT / "docs"


@dataclass
class ModelConfig:
    """Machine learning model hyperparameters and configuration."""

    model_type: str = "random_forest"  # Options: 'random_forest', 'svm'
    n_estimators: int = 150
    max_depth: int = 16
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    svm_c: float = 10.0
    svm_gamma: str = "scale"
    random_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15


@dataclass
class InferenceConfig:
    """Real-time webcam inference and smoothing settings."""

    # Hand detection parameters
    max_num_hands: int = 1
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.50

    # Decision and hysteresis thresholds
    confidence_threshold: float = 0.70  # Predictions with lower prob marked UNCERTAIN
    smoothing_window_size: int = 7       # Frames in sliding window
    majority_threshold_ratio: float = 0.70  # Fraction of window that must agree
    debounce_cooldown_seconds: float = 0.80  # Cooldown before repeating same letter

    # Camera capture settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps_target: int = 30


@dataclass
class SignLanguageConfig:
    """Target sign language definition and vocabulary."""

    sign_language: str = "American Sign Language (ASL)"
    modality: str = "Static Fingerspelling Alphabet"

    # 24 static letters of ASL alphabet (J and Z require motion and are excluded from static MVP)
    classes: List[str] = field(
        default_factory=lambda: [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
            "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y"
        ]
    )

    # Special virtual actions
    special_actions: List[str] = field(
        default_factory=lambda: ["SPACE", "BACKSPACE", "CLEAR"]
    )


@dataclass
class AppConfig:
    """Unified application configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    signs: SignLanguageConfig = field(default_factory=SignLanguageConfig)


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
