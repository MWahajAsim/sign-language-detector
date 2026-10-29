"""Interactive Gradio Web Application for Sign Language Recognition.

Provides real-time webcam inference, recorded video file transcription, static photo analysis,
landmark visualization, confidence filtering, temporal smoothing controls, and sentence accumulation.
"""

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import cv2
import gradio as gr
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.inference import InferenceEngine, InferenceOutput

# Initialize singleton inference engine
engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Lazy initialize and return the global InferenceEngine instance."""
    global engine
    if engine is None:
        print("[Gradio] Initializing InferenceEngine...")
        engine = InferenceEngine(DEFAULT_CONFIG)
    return engine


def process_webcam_frame(
    frame: np.ndarray,
    draw_landmarks: bool,
    confidence_threshold: float,
    window_size: int,
) -> Tuple[Optional[np.ndarray], str, str, str, str]:
    """Process a live frame from Gradio webcam stream.

    Returns:
        Tuple of (annotated_frame, predicted_label, confidence_str, status_str, accumulated_text)
    """
    if frame is None:
        return None, "None", "0.0%", "Awaiting webcam feed...", ""

    eng = get_inference_engine()
    eng.set_confidence_threshold(confidence_threshold)
    eng.set_smoothing_window(int(window_size))

    # Gradio provides RGB frame directly; convert to BGR for OpenCV-based engine
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output: InferenceOutput = eng.process_frame(bgr_frame, draw_landmarks=draw_landmarks, mirror=True)

    # Convert annotated BGR frame back to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(output.annotated_frame, cv2.COLOR_BGR2RGB)

    conf_display = f"{output.confidence * 100:.1f}%" if output.hand_detected else "N/A"

    return (
        annotated_rgb,
        output.predicted_label,
        conf_display,
        output.status_message,
        output.accumulated_text,
    )


def process_uploaded_video(
    video_path: Optional[str],
    draw_landmarks: bool,
    confidence_threshold: float,
    window_size: int,
) -> Tuple[Optional[str], str, str]:
    """Process an uploaded video file frame-by-frame and transcribe into a sentence.

    Returns:
        Tuple of (annotated_video_path, accumulated_sentence, status_message)
    """
    if not video_path:
        return None, "", "Please upload a video file first."

    eng = get_inference_engine()
    eng.set_confidence_threshold(confidence_threshold)
    eng.set_smoothing_window(int(window_size))

    out_file, sentence, frame_count = eng.process_video_file(
        video_path=video_path,
        draw_landmarks=draw_landmarks,
        mirror=False,
    )
    status_summary = f"✅ Analyzed {frame_count} video frames. Generated Sentence: '{sentence}'"
    return out_file, sentence, status_summary


def process_static_image(
    image: np.ndarray,
    draw_landmarks: bool,
    confidence_threshold: float,
) -> Tuple[Optional[np.ndarray], str, str, str, Dict[str, float], str]:
    """Process an uploaded static image.

    Returns:
        Tuple of (annotated_image, predicted_label, confidence_str, status_str, top_probabilities, sentence)
    """
    if image is None:
        return None, "None", "0.0%", "Please upload an image.", {}, ""

    eng = get_inference_engine()
    eng.set_confidence_threshold(confidence_threshold)

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output: InferenceOutput = eng.process_image(bgr_image, draw_landmarks=draw_landmarks)

    annotated_rgb = cv2.cvtColor(output.annotated_frame, cv2.COLOR_BGR2RGB)
    conf_display = f"{output.confidence * 100:.1f}%" if output.hand_detected else "N/A"

    # Sort top 5 probabilities for clean bar chart display
    top_probs = dict(sorted(output.probabilities.items(), key=lambda item: item[1], reverse=True)[:5])

    return (
        annotated_rgb,
        output.predicted_label,
        conf_display,
        output.status_message,
        top_probs,
        eng.text_accumulator.get_text(),
    )


def handle_add_space() -> str:
    """Append a space to the text buffer."""
    eng = get_inference_engine()
    return eng.add_space()


def handle_backspace() -> str:
    """Delete last character from the text buffer."""
    eng = get_inference_engine()
    return eng.backspace()


def handle_clear() -> str:
    """Clear all accumulated text and reset temporal smoothing."""
    eng = get_inference_engine()
    return eng.clear_text()


def handle_append_from_image(current_pred: str) -> str:
    """Append currently detected static image sign into text buffer."""
    eng = get_inference_engine()
    if current_pred and current_pred not in ["None", "NO_HAND", "UNCERTAIN"]:
        eng.text_accumulator.append(current_pred)
    return eng.text_accumulator.get_text()


def build_gradio_app() -> gr.Blocks:
    """Construct the complete Gradio interface layout and event bindings."""
    custom_css = """
    .banner-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 0.2rem; }
    .banner-subtitle { font-size: 1.1rem; text-align: center; color: #6b7280; margin-bottom: 1.2rem; }
    .disclaimer-box { background-color: #fef3c7; border-left: 5px solid #f59e0b; padding: 12px 16px; border-radius: 6px; color: #92400e; font-size: 0.92rem; margin-bottom: 1.5rem; }
    .text-display-box textarea { font-family: 'Courier New', monospace; font-size: 1.5rem !important; font-weight: bold; }
    """

    with gr.Blocks(title="Sign Language Detection System", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            <div class="banner-title">🖐️ Sign Language Detection & Recognition System</div>
            <div class="banner-subtitle">Real-Time ASL Fingerspelling Recognition & Sentence Transcription using MediaPipe Hands & Machine Learning</div>
            <div class="disclaimer-box">
                <strong>⚠️ ETHICAL & SCOPE NOTICE:</strong> This application is an educational and assistive prototype strictly designed for isolated static American Sign Language (ASL) fingerspelling (letters A-Y, excluding motion letters J & Z). It does <em>not</em> translate natural continuous sign language or grammar, and must <strong>never</strong> replace qualified human sign language interpreters in healthcare, legal, or emergency situations.
            </div>
            """
        )

        with gr.Tabs():
            # ======================== TAB 1: LIVE WEBCAM ========================
            with gr.TabItem("📹 Live Webcam Stream"):
                with gr.Row():
                    with gr.Column(scale=3):
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Live Camera Stream",
                            type="numpy",
                        )
                        webcam_output = gr.Image(
                            label="Real-Time Detection Feed (Skeleton & HUD)",
                            type="numpy",
                            interactive=False,
                        )
                        with gr.Row():
                            draw_landmarks_cb = gr.Checkbox(
                                value=True,
                                label="Show 21-Point Hand Skeleton Overlay",
                            )

                        with gr.Accordion("⚙️ Advanced Detection Settings", open=False):
                            conf_slider = gr.Slider(
                                minimum=0.40,
                                maximum=0.95,
                                value=0.65,
                                step=0.05,
                                label="Confidence Rejection Threshold (tau)",
                                info="Predictions below this score are marked UNCERTAIN."
                            )
                            smooth_slider = gr.Slider(
                                minimum=3,
                                maximum=15,
                                value=7,
                                step=2,
                                label="Sliding Window Size (Frames)",
                                info="Number of consecutive frames used for majority voting."
                            )

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 🔍 Real-Time Sign Diagnostics")
                            with gr.Row():
                                live_prediction = gr.Textbox(
                                    label="Detected Sign",
                                    value="None",
                                    interactive=False,
                                    scale=1,
                                )
                                live_confidence = gr.Textbox(
                                    label="Confidence",
                                    value="0.0%",
                                    interactive=False,
                                    scale=1,
                                )
                            live_status = gr.Textbox(
                                label="Pipeline Status",
                                value="Ready",
                                interactive=False,
                            )

                        with gr.Group():
                            gr.Markdown("### 📝 Accumulated Text Buffer")
                            accumulated_text_box = gr.Textbox(
                                label="Sentence Output",
                                value="",
                                lines=3,
                                elem_classes=["text-display-box"],
                                interactive=True,
                            )
                            with gr.Row():
                                btn_space = gr.Button("␣ Space", variant="secondary")
                                btn_backspace = gr.Button("⌫ Backspace", variant="secondary")
                                btn_clear = gr.Button("🗑️ Clear", variant="stop")

                # Wire real-time webcam frame processing
                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_input, draw_landmarks_cb, conf_slider, smooth_slider],
                    outputs=[
                        webcam_output,
                        live_prediction,
                        live_confidence,
                        live_status,
                        accumulated_text_box,
                    ],
                )

                btn_space.click(fn=handle_add_space, inputs=[], outputs=[accumulated_text_box])
                btn_backspace.click(fn=handle_backspace, inputs=[], outputs=[accumulated_text_box])
                btn_clear.click(fn=handle_clear, inputs=[], outputs=[accumulated_text_box])

            # ======================== TAB 2: VIDEO FILE ANALYSIS ========================
            with gr.TabItem("🎬 Video File Analysis"):
                with gr.Row():
                    with gr.Column(scale=3):
                        video_file_input = gr.Video(
                            sources=["upload"],
                            label="Upload Sign Language Video (MP4 / AVI / MOV / WebM)",
                        )
                        btn_process_video = gr.Button("🚀 Process Video & Generate Sentence", variant="primary")
                        annotated_video_output = gr.Video(
                            label="Annotated Video Output (with Hand Skeleton & Tracking HUD)",
                            interactive=False,
                        )

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📝 Transcribed Sentence from Video")
                            video_sentence_box = gr.Textbox(
                                label="Sentence Output",
                                value="",
                                lines=3,
                                elem_classes=["text-display-box"],
                                interactive=True,
                            )
                            with gr.Row():
                                btn_vid_space = gr.Button("␣ Space", variant="secondary")
                                btn_vid_backspace = gr.Button("⌫ Backspace", variant="secondary")
                                btn_vid_clear = gr.Button("🗑️ Clear", variant="stop")

                            video_status_box = gr.Textbox(
                                label="Video Processing Status",
                                value="Upload a video and click 'Process Video'.",
                                interactive=False,
                            )

                btn_process_video.click(
                    fn=process_uploaded_video,
                    inputs=[video_file_input, draw_landmarks_cb, conf_slider, smooth_slider],
                    outputs=[
                        annotated_video_output,
                        video_sentence_box,
                        video_status_box,
                    ],
                )

                btn_vid_space.click(fn=handle_add_space, inputs=[], outputs=[video_sentence_box])
                btn_vid_backspace.click(fn=handle_backspace, inputs=[], outputs=[video_sentence_box])
                btn_vid_clear.click(fn=handle_clear, inputs=[], outputs=[video_sentence_box])

            # ======================== TAB 3: STATIC IMAGE UPLOAD ========================
            with gr.TabItem("🖼️ Static Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(
                            sources=["upload"],
                            type="numpy",
                            label="Upload Hand Sign Photo (JPG / PNG)",
                        )
                        btn_analyze = gr.Button("🔍 Analyze Hand Sign", variant="primary")
                        image_output = gr.Image(
                            label="Annotated Hand (Skeleton & Diagnostics)",
                            interactive=False,
                            type="numpy",
                        )

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 🔍 Image Prediction Results")
                            with gr.Row():
                                static_prediction = gr.Textbox(label="Predicted Letter", interactive=False)
                                static_confidence = gr.Textbox(label="Confidence Score", interactive=False)
                            static_status = gr.Textbox(label="Status", interactive=False)
                            prob_chart = gr.Label(label="Top Class Probabilities", num_top_classes=5)

                        with gr.Group():
                            gr.Markdown("### 📝 Accumulated Sentence Buffer")
                            photo_sentence_box = gr.Textbox(
                                label="Current Sentence",
                                value="",
                                lines=2,
                                elem_classes=["text-display-box"],
                                interactive=True,
                            )
                            with gr.Row():
                                btn_append = gr.Button("➕ Append Sign to Sentence", variant="primary")
                                btn_photo_space = gr.Button("␣ Space", variant="secondary")
                                btn_photo_backspace = gr.Button("⌫ Backspace", variant="secondary")
                                btn_photo_clear = gr.Button("🗑️ Clear", variant="stop")

                btn_analyze.click(
                    fn=process_static_image,
                    inputs=[image_input, draw_landmarks_cb, conf_slider],
                    outputs=[
                        image_output,
                        static_prediction,
                        static_confidence,
                        static_status,
                        prob_chart,
                        photo_sentence_box,
                    ],
                )

                btn_append.click(
                    fn=handle_append_from_image,
                    inputs=[static_prediction],
                    outputs=[photo_sentence_box],
                )

                btn_photo_space.click(fn=handle_add_space, inputs=[], outputs=[photo_sentence_box])
                btn_photo_backspace.click(fn=handle_backspace, inputs=[], outputs=[photo_sentence_box])
                btn_photo_clear.click(fn=handle_clear, inputs=[], outputs=[photo_sentence_box])

            # ======================== TAB 4: SYSTEM SPECS & VOCABULARY ========================
            with gr.TabItem("ℹ️ Supported Signs & Documentation"):
                gr.Markdown(
                    r"""
                    ### 📚 Supported ASL Fingerspelling Vocabulary (24 Static Letters)
                    The system recognizes the 24 static handshapes of the American Sign Language manual alphabet:
                    
                    | Letters Supported | Description |
                    | :--- | :--- |
                    | **A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y** | Static single-hand fingerspelling handshapes. |
                    
                    > **Note on J and Z:** In standard ASL, **J** requires tracing a 'J' curve in the air with the pinky, and **Z** requires tracing a 'Z' zigzag in the air with the index finger. Because these letters require spatiotemporal motion trajectories, they are excluded from the static MVP and are planned for Phase 2 dynamic sequence modeling.
                    
                    ### 🔬 Pipeline Specifications
                    - **Vision Backbone:** MediaPipe Hands (21 3D anatomical landmarks).
                    - **Normalization:** Invariant wrist-centering ($P'_i = P_i - P_0$), automatic handedness reflection ($P'_x = -P'_x$ when left-handed), and Euclidean distance scaling ($P''_i = P'_i / d_{\max}$).
                    - **Model Engine:** 150-Tree Random Forest Classifier with Platt-calibrated probability distributions.
                    - **Temporal Hysteresis:** 7-frame sliding window with majority consensus filtering and 0.8s duplicate debounce cooldown.
                    """
                )

    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False):
    """Launch the Gradio web application."""
    app = build_gradio_app()
    app.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
