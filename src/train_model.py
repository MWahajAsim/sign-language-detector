"""Model training pipeline.

Loads normalized landmark datasets, enforces signer-aware or stratified splits,
fits baseline Random Forest / SVM classifiers, evaluates on validation/test sets,
and serializes model artifacts with comprehensive metadata.
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_CONFIG, AppConfig
from src.models import SignClassifier
from src.synthetic_data import generate_synthetic_dataset



def load_or_generate_dataset(
    config: AppConfig,
    force_generate: bool = False
) -> pd.DataFrame:
    """Load existing landmark CSV or generate a synthetic baseline dataset.

    Args:
        config: Application configuration.
        force_generate: If True, regenerates dataset even if file exists.

    Returns:
        Pandas DataFrame of landmark samples.
    """
    dataset_file = config.paths.processed_data_dir / "landmarks_dataset.csv"

    if force_generate or not dataset_file.exists():
        print(f"[Dataset] Generating baseline landmark dataset at {dataset_file}...")
        df = generate_synthetic_dataset(
            output_csv_path=dataset_file,
            classes=config.signs.classes,
            num_signers=4,
            samples_per_signer_class=60,
            random_seed=config.model.random_seed,
        )
        print(f"[Dataset] Generated {len(df)} samples across {df['label'].nunique()} classes.")
        return df

    print(f"[Dataset] Loading existing dataset from {dataset_file}...")
    df = pd.read_csv(dataset_file)
    print(f"[Dataset] Loaded {len(df)} samples across {df['label'].nunique()} classes.")
    return df


def prepare_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Prepare train, validation, and test splits with stratification.

    If multiple signers exist, holds out an entire signer for testing to evaluate
    unseen-signer generalization.
    """
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if len(feature_cols) != 63:
        raise ValueError(f"Expected 63 feature columns, found {len(feature_cols)}.")

    classes = sorted(df["label"].unique().tolist())
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["label"].values

    # Check if multiple signers exist for signer-independent split
    signers = df["signer_id"].unique().tolist() if "signer_id" in df.columns else []

    if len(signers) >= 4:
        # Signer-independent splitting: hold out one signer completely for testing
        test_signer = signers[-1]
        val_signer = signers[-2]
        train_signers = signers[:-2]

        print(f"[Splits] Signer-Independent Split: Train={train_signers}, Val={val_signer}, Test={test_signer}")

        train_mask = df["signer_id"].isin(train_signers)
        val_mask = df["signer_id"] == val_signer
        test_mask = df["signer_id"] == test_signer

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
    else:
        # Stratified random split
        print("[Splits] Performing stratified random train/val/test split...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_all, y_all,
            test_size=test_size,
            stratify=y_all,
            random_state=random_seed
        )

        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=relative_val_size,
            stratify=y_train_val,
            random_state=random_seed
        )

    print(f"[Splits] Train: {len(X_train)} samples | Val: {len(X_val)} samples | Test: {len(X_test)} samples")
    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def train(
    config: AppConfig = DEFAULT_CONFIG,
    model_type: str = "random_forest",
    force_data_gen: bool = False
) -> Dict[str, float]:
    """Execute complete training pipeline and export serialized model artifact.

    Args:
        config: Application configuration instance.
        model_type: 'random_forest' or 'svm'.
        force_data_gen: If True, regenerates synthetic dataset.

    Returns:
        Dictionary of validation and test metrics.
    """
    df = load_or_generate_dataset(config, force_generate=force_data_gen)

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = prepare_splits(
        df,
        test_size=config.model.test_size,
        val_size=config.model.val_size,
        random_seed=config.model.random_seed
    )

    print(f"\n[Training] Fitting {model_type} classifier on {len(X_train)} samples...")
    classifier = SignClassifier(model_type=model_type, config=config.model)
    classifier.fit(X_train, y_train, class_names=class_names)

    # 1. Validation Set Evaluation
    val_preds = classifier.estimator.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds, average="macro"))

    print(f"\n[Validation Results]")
    print(f"Validation Top-1 Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Macro-F1 Score: {val_f1:.4f}")

    # 2. Test Set Evaluation (Held-Out Final Benchmark)
    test_preds = classifier.estimator.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    test_f1 = float(f1_score(y_test, test_preds, average="macro"))

    print(f"\n[Held-Out Test Results]")
    print(f"Test Top-1 Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Macro-F1 Score: {test_f1:.4f}")

    # 3. Save Model Artifact
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(config.paths.models_dir)
    print(f"\n[Artifacts] Serialized model saved to {config.paths.models_dir}")

    # Update metadata with benchmark metrics
    meta_path = config.paths.metadata_path
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata.update({
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "num_train_samples": len(X_train),
            "num_val_samples": len(X_val),
            "num_test_samples": len(X_test),
        })
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Detection Model")
    parser.add_argument(
        "--model", type=str, default="random_forest", choices=["random_forest", "svm"],
        help="Classifier model architecture"
    )
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Force regeneration of the landmark dataset"
    )
    args = parser.parse_args()

    train(model_type=args.model, force_data_gen=args.regenerate_data)
