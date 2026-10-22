"""Model evaluation and benchmarking suite.

Generates comprehensive classification metrics, confusion matrices, latency
benchmarks, and false-positive rejection tests on held-out test data.
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.config import DEFAULT_CONFIG, AppConfig
from src.models import SignClassifier
from src.train_model import prepare_splits


def evaluate(
    config: AppConfig = DEFAULT_CONFIG,
    output_dir: Path = PROJECT_ROOT / "reports"
) -> dict:
    """Evaluate trained model on held-out test set and export reports.

    Args:
        config: Application configuration.
        output_dir: Directory to save evaluation reports and plots.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print(f"[Evaluation] Loading model from {config.paths.models_dir}...")
    classifier = SignClassifier.load(config.paths.models_dir)

    # 2. Load Dataset and Test Split
    data_file = config.paths.processed_data_dir / "landmarks_dataset.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}. Run training first.")

    df = pd.read_csv(data_file)
    _, _, _, _, X_test, y_test, class_names = prepare_splits(
        df,
        test_size=config.model.test_size,
        val_size=config.model.val_size,
        random_seed=config.model.random_seed
    )

    print(f"[Evaluation] Evaluating on {len(X_test)} held-out test samples across {len(class_names)} classes...")

    # 3. Model Predictions
    y_pred = classifier.estimator.predict(X_test)
    y_proba = classifier.estimator.predict_proba(X_test)

    # Metrics
    test_acc = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    clf_report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"\n=======================================================")
    print(f"             HELD-OUT TEST EVALUATION SUMMARY          ")
    print(f"=======================================================")
    print(f"Top-1 Accuracy:  {test_acc * 100:.2f}%")
    print(f"Macro-F1 Score:  {test_macro_f1:.4f}")
    print(f"=======================================================\n")

    # 4. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(f"Normalized Confusion Matrix ({classifier.model_type.upper()})\nTest Accuracy: {test_acc*100:.1f}% | Macro-F1: {test_macro_f1:.3f}")
    plt.xlabel("Predicted Sign")
    plt.ylabel("Ground Truth Sign")
    plt.tight_layout()

    cm_plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix visualization to {cm_plot_path}")

    # 5. Latency & Real-Time Throughput Benchmark
    print("[Benchmark] Benchmarking CPU inference latency over 500 passes...")
    warmup_n = 50
    bench_n = 500

    # Warmup
    for _ in range(warmup_n):
        _ = classifier.predict(X_test[0], confidence_threshold=0.70)

    latencies = []
    for i in range(bench_n):
        sample = X_test[i % len(X_test)]
        t0 = time.perf_counter()
        _ = classifier.predict(sample, confidence_threshold=0.70)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    equiv_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    print(f"Mean Latency:    {mean_lat:.3f} ms")
    print(f"P50 Latency:     {p50_lat:.3f} ms")
    print(f"P95 Latency:     {p95_lat:.3f} ms")
    print(f"Equivalent FPS:  {equiv_fps:.1f} FPS (model inference only)")

    # 6. Negative / Out-of-Distribution Rejection Test
    print("[Rejection Test] Testing rejection on 500 random noise vectors...")
    rng = np.random.RandomState(42)
    noise_samples = rng.uniform(-1.0, 1.0, size=(500, 63)).astype(np.float32)
    # Normalize noise samples as standard landmarks
    noise_samples = np.array([s / (np.max(np.abs(s)) + 1e-7) for s in noise_samples])

    rejected_count = 0
    for sample in noise_samples:
        pred = classifier.predict(sample, confidence_threshold=0.70)
        if not pred.is_valid:
            rejected_count += 1

    rejection_rate = float(rejected_count / len(noise_samples))
    print(f"Uncertainty Rejection Rate (tau=0.70): {rejection_rate * 100:.2f}% (Target: >= 80%)")

    # 7. Export Full Structured Report
    report = {
        "model_type": classifier.model_type,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "num_test_samples": len(X_test),
        "latency_benchmark_ms": {
            "mean": mean_lat,
            "p50": p50_lat,
            "p95": p95_lat,
            "p99": p99_lat,
            "equivalent_fps": equiv_fps,
        },
        "negative_rejection_test": {
            "num_noise_samples": len(noise_samples),
            "rejection_rate": rejection_rate,
            "threshold_used": 0.70,
        },
        "per_class_metrics": clf_report_dict,
    }

    report_json_path = output_dir / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Evaluation] Saved structured metrics to {report_json_path}\n")

    return report


if __name__ == "__main__":
    evaluate()
