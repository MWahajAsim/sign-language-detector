"""Data collection utility for custom sign language landmark samples.

Provides interactive webcam capture, on-screen guidance, and automatic
landmark extraction to collect high-quality real-world training datasets.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.feature_extraction import HandLandmarkDetector
from src.preprocessing import FrameProcessor, LandmarkNormalizer


def collect_samples(
    letter: str,
    signer_id: str = "signer_01",
    num_samples: int = 50,
    camera_index: int = 0,
    output_csv: Path = PROJECT_ROOT / "data" / "raw" / "custom_landmarks.csv"
) -> None:
    """Capture hand landmarks from webcam for a specific sign label.

    Args:
        letter: Character label to collect (e.g., 'A', 'B', etc.).
        signer_id: Identifier of the person signing.
        num_samples: Target count of stable frames to collect.
        camera_index: OpenCV camera device index.
        output_csv: File path to append captured samples.
    """
    letter = letter.upper()
    print(f"\n[Data Collection] Initializing camera {camera_index} for Sign '{letter}'...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_index}. Please check device connection.")
        return

    detector = HandLandmarkDetector(static_image_mode=False, max_num_hands=1)
    processor = FrameProcessor()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    collected_records = []
    is_recording = False
    samples_captured = 0

    print("=============================================================")
    print(f" Ready to collect {num_samples} samples for letter '{letter}'.")
    print(" Controls:")
    print("   Press 'SPACE' or 'C' : Start/Pause automatic capture")
    print("   Press 'Q' or 'ESC'   : Finish and save session")
    print("=============================================================\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Failed to grab frame from camera.")
                break

            rgb_frame, display_frame = processor.process(frame, mirror=True)
            landmarks, landmark_proto = detector.extract(rgb_frame)

            annotated = display_frame.copy()
            if landmark_proto is not None:
                annotated = detector.draw_landmarks(annotated, landmark_proto)

            # Check recording logic
            if is_recording and landmarks is not None:
                norm_feat = LandmarkNormalizer.normalize(landmarks)
                row = {"label": letter, "signer_id": signer_id}
                for i in range(63):
                    row[f"feat_{i}"] = float(norm_feat[i])
                collected_records.append(row)
                samples_captured += 1

                if samples_captured >= num_samples:
                    is_recording = False
                    print(f"\n[Success] Captured all {num_samples} samples for '{letter}'!")

            # HUD Display
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 60), (30, 30, 30), -1)

            rec_status = "RECORDING" if is_recording else "PAUSED (Press C/Space)"
            rec_color = (0, 0, 255) if is_recording else (0, 215, 255)
            cv2.putText(
                annotated,
                f"Sign: '{letter}' | Signer: {signer_id} | [{rec_status}]",
                (15, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, rec_color, 1, cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                f"Samples: {samples_captured}/{num_samples} | Hand: {'DETECTED' if landmarks is not None else 'NOT FOUND'}",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
            )

            cv2.imshow("Sign Language Data Collector", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('c'), ord('C'), 32]:  # 'C' or Spacebar
                is_recording = not is_recording
                print(f"[Status] Recording toggled: {'ON' if is_recording else 'PAUSED'}")
            elif key in [ord('q'), ord('Q'), 27]:  # 'Q' or ESC
                print("[Status] Ending collection session.")
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    if collected_records:
        df_new = pd.DataFrame(collected_records)
        if output_csv.exists():
            df_existing = pd.read_csv(output_csv)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(output_csv, index=False)
            print(f"[Data Collection] Appended {len(df_new)} rows to {output_csv} (Total: {len(df_combined)} rows).")
        else:
            df_new.to_csv(output_csv, index=False)
            print(f"[Data Collection] Saved {len(df_new)} rows to new file at {output_csv}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Hand Landmarks from Webcam")
    parser.add_argument("--letter", type=str, required=True, help="Letter to collect (e.g. A, B, C)")
    parser.add_argument("--signer", type=str, default="signer_01", help="Signer ID string")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()

    collect_samples(
        letter=args.letter,
        signer_id=args.signer,
        num_samples=args.samples,
        camera_index=args.camera
    )
