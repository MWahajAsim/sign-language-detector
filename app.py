"""Main application entry point for Sign Language Detection System.

Provides command-line switches to launch the Gradio web interface, run a native
OpenCV desktop diagnostic window, execute model training, or perform benchmarks.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.evaluate_model import evaluate
from src.inference import InferenceEngine, InferenceOutput
from src.train_model import train
from ui.gradio_app import launch as launch_gradio


def run_opencv_desktop(camera_index: int = 0) -> None:
    """Run real-time inference loop in an ultra-low latency native OpenCV desktop window."""
    print(f"\n[Desktop App] Initializing OpenCV window on camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_index}. Please check connection.")
        return

    engine = InferenceEngine(DEFAULT_CONFIG)
    window_name = "Sign Language Detection System (Native OpenCV Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("=============================================================")
    print(" Controls:")
    print("   'SPACE' : Insert space into sentence")
    print("   'B'     : Backspace / delete last character")
    print("   'C'     : Clear accumulated text")
    print("   'L'     : Toggle landmark skeleton overlay")
    print("   'Q'/ESC : Quit application")
    print("=============================================================\n")

    draw_landmarks = True
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Failed to grab frame from camera.")
                break

            output: InferenceOutput = engine.process_frame(
                frame,
                draw_landmarks=draw_landmarks,
                mirror=True
            )

            # Compute actual runtime FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-7)
            prev_time = curr_time

            display_img = output.annotated_frame

            # Render bottom sentence bar
            h, w = display_img.shape[:2]
            cv2.rectangle(display_img, (0, h - 55), (w, h), (15, 15, 15), -1)
            cv2.putText(
                display_img,
                f"Sentence: {output.accumulated_text}",
                (15, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Render FPS in top-right corner
            cv2.putText(
                display_img,
                f"{fps:.1f} FPS",
                (w - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == 32:  # Spacebar
                engine.add_space()
            elif key in [ord('b'), ord('B')]:
                engine.backspace()
            elif key in [ord('c'), ord('C')]:
                engine.clear_text()
            elif key in [ord('l'), ord('L')]:
                draw_landmarks = not draw_landmarks
                print(f"[Toggle] Landmark overlay: {'ON' if draw_landmarks else 'OFF'}")

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Parse CLI commands and launch selected mode."""
    parser = argparse.ArgumentParser(
        description="Sign Language Detection and Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="web",
        choices=["web", "cv2"],
        help="Interface mode: 'web' (Gradio browser UI) or 'cv2' (native OpenCV window)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio web server port (default: 7860)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run model training pipeline and export serialized artifacts",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run comprehensive test evaluation and export reports",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm"],
        help="Model architecture for training",
    )

    args = parser.parse_args()

    # Model training mode
    if args.train:
        print("[Launcher] Starting model training pipeline...")
        train(model_type=args.model_type)
        return

    # Evaluation mode
    if args.evaluate:
        print("[Launcher] Starting evaluation and benchmark suite...")
        evaluate()
        return

    # Launch requested interface
    if args.mode == "web":
        print(f"[Launcher] Launching Gradio Web Application on http://127.0.0.1:{args.port}...")
        launch_gradio(server_port=args.port)
    elif args.mode == "cv2":
        run_opencv_desktop(camera_index=args.camera)


if __name__ == "__main__":
    main()
