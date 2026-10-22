"""Machine Learning model wrapper and serialization module.

Provides unified interface for training, probability prediction, and serialization
of Random Forest and Support Vector Machine classifiers.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.config import DEFAULT_CONFIG, ModelConfig


@dataclass
class PredictionResult:
    """Standardized prediction output dataclass."""

    label: str
    confidence: float
    probabilities: Dict[str, float]
    is_valid: bool


class SignClassifier:
    """Machine learning classifier for normalized 63-dim hand landmark vectors."""

    def __init__(
        self,
        model_type: str = "random_forest",
        config: Optional[ModelConfig] = None,
    ):
        self.model_type = model_type.lower()
        self.config = config or DEFAULT_CONFIG.model
        self.class_names: List[str] = []
        self.estimator: Optional[Union[RandomForestClassifier, SVC]] = None
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Instantiate the underlying scikit-learn estimator."""
        if self.model_type == "random_forest":
            self.estimator = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_seed,
                n_jobs=-1,
            )
        elif self.model_type == "svm":
            self.estimator = SVC(
                C=self.config.svm_c,
                kernel="rbf",
                gamma=self.config.svm_gamma,
                probability=True,  # Enables Platt scaling for predict_proba
                random_state=self.config.random_seed,
            )
        else:
            raise ValueError(
                f"Unsupported model_type: '{self.model_type}'. Choose 'random_forest' or 'svm'."
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: List[str]
    ) -> "SignClassifier":
        """Train the classifier on normalized feature vectors.

        Args:
            X: 2D array of shape (N_samples, 63).
            y: 1D array of integer class indices or string labels.
            class_names: Ordered list of string class names.

        Returns:
            self: Fitted instance.
        """
        if X.ndim != 2 or X.shape[1] != 63:
            raise ValueError(f"Expected X with shape (N, 63), received {X.shape}")

        self.class_names = list(class_names)
        self.estimator.fit(X, y)
        return self

    def predict(
        self,
        features_63: np.ndarray,
        confidence_threshold: float = 0.70
    ) -> PredictionResult:
        """Predict sign label and class probabilities for a single feature vector.

        Args:
            features_63: Normalized 1D array of shape (63,).
            confidence_threshold: Minimum probability to consider prediction valid.

        Returns:
            PredictionResult with top label, confidence score, and all probabilities.
        """
        if self.estimator is None or not self.class_names:
            raise RuntimeError("Model is not trained or loaded. Call fit() or load() first.")

        # Ensure 2D input (1, 63)
        x_in = features_63.reshape(1, -1)

        # Get calibrated probability distribution
        probs = self.estimator.predict_proba(x_in)[0]  # shape (num_classes,)

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top_label = self.class_names[top_idx]

        is_valid = top_conf >= confidence_threshold

        prob_dict = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        return PredictionResult(
            label=top_label,
            confidence=top_conf,
            probabilities=prob_dict,
            is_valid=is_valid,
        )

    def save(self, directory: Union[str, Path]) -> None:
        """Serialize trained model, class list, and metadata to directory.

        Args:
            directory: Target directory path.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        model_file = dir_path / "classifier.joblib"
        class_file = dir_path / "class_names.json"
        meta_file = dir_path / "metadata.json"

        # 1. Save estimator
        joblib.dump(self.estimator, model_file)

        # 2. Save class names
        with open(class_file, "w", encoding="utf-8") as f:
            json.dump(self.class_names, f, indent=2)

        # 3. Save metadata
        metadata = {
            "model_type": self.model_type,
            "feature_dim": 63,
            "num_classes": len(self.class_names),
            "classes": self.class_names,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scikit_learn_version": joblib.__version__,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SignClassifier":
        """Load serialized model, class names, and metadata from directory.

        Args:
            directory: Directory containing classifier.joblib and class_names.json.

        Returns:
            Loaded SignClassifier instance.
        """
        dir_path = Path(directory)
        model_file = dir_path / "classifier.joblib"
        class_file = dir_path / "class_names.json"
        meta_file = dir_path / "metadata.json"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_file}. Run training first."
            )
        if not class_file.exists():
            raise FileNotFoundError(
                f"Class names mapping not found at {class_file}."
            )

        with open(class_file, "r", encoding="utf-8") as f:
            class_names = json.load(f)

        model_type = "random_forest"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                model_type = metadata.get("model_type", "random_forest")

        instance = cls(model_type=model_type)
        instance.estimator = joblib.load(model_file)
        instance.class_names = class_names
        return instance
