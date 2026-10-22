# Product Requirements Document (PRD)

**Project Name:** Sign Language Detection and Recognition System  
**Document Type:** Product Requirements Document (PRD)  
**Version:** 1.0  
**Status:** Approved for MVP Implementation  
**Target Sign Language:** American Sign Language (ASL) Fingerspelling (Static Alphabet)  
**Authors:** Senior Machine Learning Engineer, Computer Vision Engineer, Software Architect, Technical Product Manager  

---

## A. Executive Summary

### 1. Problem Statement
Over 70 million deaf and hard-of-hearing individuals worldwide use sign language as their primary mode of communication. However, because the overwhelming majority of hearing society does not understand sign language, deaf individuals face persistent communication barriers in administrative, medical, educational, and commercial environments. While certified human sign language interpreters provide essential communication access, severe interpreter shortages and high costs make them unavailable for spontaneous, routine, or educational interactions.

Automated gesture recognition systems offer a promising assistive capability. However, many naive systems oversimplify sign language into static 2D image matching, fail under diverse lighting or skin tones, or claim full translation capabilities without addressing the complex linguistic syntax, facial grammar, and temporal dynamics of natural sign language.

### 2. Proposed Solution
This project delivers a robust, modular, accessible computer vision application in Python that recognizes sign language gestures from standard RGB webcam streams or static image uploads and converts them into readable text in real time. 

The **Minimum Viable Product (MVP)** deliberately focuses on **isolated static fingerspelling classification** (the static letters of the ASL alphabet) paired with an intelligent temporal hysteresis smoothing buffer. This architecture delivers high accuracy, ultra-low latency, and predictable performance on standard commodity laptops without requiring expensive discrete GPUs.

### 3. Intended Users and Beneficiaries
- **Deaf and Hard-of-Hearing Individuals:** Users seeking a quick, accessible text-entry tool for fingerspelling names, numbers, or specific terms in one-on-one scenarios with non-signers.
- **Sign Language Students & Learners:** Beginners seeking real-time interactive feedback on their handshapes, joint positioning, and fingerspelling clarity.
- **Special Education Instructors:** Educators requiring transparent, reliable demonstration software to teach accessible technology and computer vision.
- **Accessibility Researchers & Developers:** Engineers seeking an extensible, decoupled reference architecture for hand-landmark extraction, normalization, and temporal filtering.

### 4. Expected Outcome & Project Boundaries
- **Expected Outcome:** A functional, locally executed Python application featuring webcam processing at $\ge 15$ FPS, hand-landmark extraction, coordinate normalization, machine learning classification ($\ge 85\%$ accuracy on held-out test data), confidence thresholding, text buffering, and a user-friendly Gradio web GUI.
- **Explicit Project Boundaries:**
  - The MVP is **not** a full natural sign language translator.
  - The MVP does **not** recognize continuous fluent signing or complex sentence syntax.
  - The MVP does **not** interpret facial expressions, body posture, or non-manual grammatical markers.
  - The MVP is an assistive educational prototype and **must never** be deployed as a substitute for certified human interpreters in medical, legal, or safety-critical contexts.

---

## B. Problem Statement & Technical Taxonomy

### 1. Communication Barriers in Detail
Sign language users frequently experience friction when attempting to convey proper nouns (names, addresses, technical terms) to non-signing interlocutors. In natural sign language, such information is conveyed through **fingerspelling** (spelling out words letter-by-letter using standardized handshapes). Developing an automated tool that reliably transcribes fingerspelled letters into words bridges this immediate gap.

### 2. Classification of Sign Language Processing Problems
To maintain scientific rigor and prevent scope creep, gesture recognition problems must be delineated into four distinct tiers:

| Tier | Category | Input Modality | Core Technical Complexity | Feasibility for MVP |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Static Hand-Sign Classification** | Single frame / static 3D landmark vector | Differentiating subtle joint angles and finger curls without temporal cues. | **Selected for MVP** (High feasibility, low compute, deterministic). |
| **Tier 2** | **Dynamic Gesture Recognition** | Sequence of frames ($T \in [15, 60]$) | Modeling spatial trajectories, velocity, and stroke phases over time. | Planned Phase 2 (Requires sequence models such as LSTM/GRU/Transformer). |
| **Tier 3** | **Continuous Sign Language Recognition (CSLR)** | Continuous, unsegmented video stream | Co-articulation, transition movement removal, dynamic sign boundary alignment. | Long-term research (Requires CTC loss or temporal action localization). |
| **Tier 4** | **Natural Sign Language Translation (SLT)** | Multi-channel video (Hands + Face + Torso) | Translating visual gloss sequences into grammatical spoken language text. | Out of scope (Requires large parallel multi-modal corpora and translation models). |

### 3. Scope Justification for Version 1.0
The initial release focuses strictly on **Tier 1 (Static ASL Fingerspelling)**. This choice ensures:
- Robust mathematical normalization against camera distance, hand size, and image resolution.
- Millisecond-level inference on CPU, eliminating thermal throttling on student laptops.
- Direct alignment between training feature distributions and real-time webcam extraction.
- Immediate utility for fingerspelling practice and letter-by-letter word construction.

---

## C. Target Users & Needs Analysis

| User Group | Core Needs | System Feature Mapping | Explicit Limitations |
| :--- | :--- | :--- | :--- |
| **Deaf / Hard-of-Hearing** | Reliable text transcription, local privacy, zero latency lag. | High-confidence predictions, instant visual feedback, text accumulation with space/backspace. | Understands isolated fingerspelling only, not fluent continuous conversational signing. |
| **Sign Language Learners** | Accuracy validation, confidence metrics, visual diagnostic overlays. | Confidence score gauge, hand-landmark skeleton overlay toggle, uncertain-sign warning. | Cannot critique subtle joint rotations or teach cultural/facial grammar. |
| **Educators & Trainers** | Simple setup, deterministic behavior, interactive controls. | One-click launch, reset/clear controls, image upload mode for offline assignments. | Intended for educational demonstration only; not a full curriculum tool. |
| **Developers & Researchers** | Modular code, reproducible evaluation, clear API contracts. | Decoupled modules (`preprocessing`, `models`, `smoothing`), standard serialization, pytest suite. | Baseline ML model; requires sequence extensions for dynamic vocabulary. |

---

## D. Product Goals and Success Metrics

All metrics represent provisional engineering targets to be validated on held-out test splits and live user trials.

| Goal | Metric | Target | Validation Protocol |
| :--- | :--- | :--- | :--- |
| **Classification Accuracy** | Top-1 Accuracy on controlled test split | $\ge 88.0\%$ | Evaluated on stratified held-out test set across 24 classes. |
| **Class Balance** | Macro-Averaged F1 Score | $\ge 0.85$ | Unweighted average of per-class F1 scores to prevent majority class bias. |
| **Unseen Signer Robustness** | Signer-Independent Top-1 Accuracy | $\ge 72.0\%$ | Evaluated strictly on signers omitted from the training split. |
| **Inference Latency** | End-to-end latency per frame | $\le 45\text{ ms}$ | Measured from frame grab to UI text update on an Intel Core i5/Ryzen 5 CPU. |
| **Real-Time Throughput** | Effective Processing Frame Rate | $\ge 15\text{ FPS}$ | Monitored during live webcam execution at $640 \times 480$ resolution. |
| **False Acceptance Rate** | False positive rate on resting/empty hands | $\le 5.0\%$ | Evaluated against a negative test dataset containing resting hands and random motions. |
| **Uncertainty Rejection** | Rejection rate of invalid/unsupported signs | $\ge 80.0\%$ | Rejection triggered whenever softmax probability or tree voting is $< \tau_{\text{conf}}$ (default 0.70). |
| **Usability** | Task completion rate | $\ge 90.0\%$ | Measured in user testing (spelling a 5-letter word without system crash). |
| **Ethical Transparency** | Visible disclaimer compliance | $100\%$ | Persistent notice in the UI indicating the system is an educational prototype. |

---

## E. Functional Requirements

### Priority Schema
- **Must Have (P0):** Critical for MVP operation.
- **Should Have (P1):** Important capability, necessary for good user experience.
- **Could Have (P2):** Optional enhancement for subsequent releases.

| ID | Title | Priority | Description & Expected Behavior | Acceptance Criteria |
| :--- | :--- | :---: | :--- | :--- |
| **FR-01** | Webcam Video Input | **P0** | Stream real-time frames from local default camera ($640 \times 480$ at $\ge 15$ FPS). | User can start/stop stream; app displays live feed; missing camera triggers graceful warning. |
| **FR-02** | Static Image Upload | **P0** | Allow user to upload PNG/JPG/JPEG images for offline landmark analysis and prediction. | Uploaded image is processed; hand detected; predicted label and confidence displayed. |
| **FR-03** | Hand Detection & Tracking | **P0** | Detect visible hands using MediaPipe Hands and extract 21 3D anatomical landmarks. | Detects hand in standard lighting; reports `NO_HAND_DETECTED` state when hand is absent without crashing. |
| **FR-04** | Landmark Normalization | **P0** | Translate landmarks relative to wrist (origin) and scale by maximum Euclidean distance. | Coordinates are invariant to hand position in frame and distance from camera. |
| **FR-05** | Sign Classification | **P0** | Classify normalized 63-dimensional landmark vectors into one of 24 static ASL letters. | Model outputs predicted class label and probability distribution across all classes. |
| **FR-06** | Confidence Filtering | **P0** | Filter predictions against configurable threshold $\tau_{\text{conf}}$ (default $0.70$). | Predictions with confidence $< \tau_{\text{conf}}$ flagged as `UNCERTAIN` and suppressed from text. |
| **FR-07** | Temporal Smoothing | **P0** | Buffer last $N$ predictions (sliding window) and enforce majority voting stability. | A letter is accepted only if present in $\ge 70\%$ of frames across a 7-frame window. |
| **FR-08** | Duplicate Suppression | **P0** | Enforce a debounce cooldown ($0.8\text{ s}$) before committing identical consecutive letters. | Holding a sign steadily does not produce a continuous runaway stream of duplicate characters. |
| **FR-09** | Text Accumulator | **P0** | Append verified stable characters into an editable sentence/word buffer. | Characters append sequentially in real time to the text field. |
| **FR-10** | Reset & Clear Controls | **P0** | Provide dedicated UI controls to clear accumulated text and reset temporal buffers. | Clicking 'Clear' clears text display and resets sliding window history instantly. |
| **FR-11** | Model Validation on Startup | **P0** | Validate model file existence, checksum, feature dimension, and class label mapping. | Informative error displayed if `models/classifier.joblib` or `class_names.json` is missing. |
| **FR-12** | Skeleton Overlay Toggle | **P1** | Toggle drawing of 21 landmarks and skeletal connection lines on the video stream. | Landmarks render when enabled and disappear cleanly when disabled. |
| **FR-13** | Word Spacing & Backspace | **P1** | Provide UI buttons and timed-pause logic to insert spaces and delete last character. | Clicking 'Space' inserts `" "`; clicking 'Backspace' removes the last character. |
| **FR-14** | Dynamic Threshold Tuning | **P1** | Interactive UI slider allowing user to adjust confidence threshold ($0.50 - 0.95$). | Slider adjusts runtime threshold dynamically without restarting the application. |
| **FR-15** | Text-to-Speech (TTS) | **P2** | Optional synthesizer to vocalize accumulated text using local `pyttsx3` or browser speech. | Audio plays on button click; failures do not disrupt visual inference loop. |
| **FR-16** | Video File Processing | **P2** | Batch processor to ingest pre-recorded MP4/AVI videos and extract continuous predictions. | Processes video frame-by-frame and outputs timestamped log of recognized signs. |

---

## F. Non-Functional Requirements

### 1. Performance
- **Latency:** Maximum 45 ms end-to-end processing latency per frame on standard quad-core CPU.
- **Throughput:** Maintain $\ge 15$ FPS on $640 \times 480$ RGB stream under standard lighting.
- **Memory Footprint:** Peak RAM consumption under 650 MB during continuous operation.

### 2. Reliability & Robustness
- **Error Recovery:** Zero unhandled crashes when camera disconnects, lighting drops to zero, or non-hand objects enter the frame.
- **Memory Stability:** Zero memory leaks over continuous 60-minute execution sessions.

### 3. Maintainability & Code Quality
- **Modularity:** Strict separation of concerns between I/O, feature extraction, normalization, modeling, smoothing, and UI.
- **Typing & Documentation:** 100% type hints (`typing.Optional`, `np.ndarray`, etc.) and Google-style docstrings across all modules.
- **Testing:** Minimum 85% code test coverage across core processing, normalization, and smoothing modules using `pytest`.

### 4. Privacy & Data Security
- **Local-First Execution:** 100% of video processing, landmark extraction, and model inference occurs on the local machine.
- **Zero Video Retention:** Raw webcam frames are processed in volatile memory and immediately discarded. No video is recorded or transmitted to any external server.

### 5. Accessibility & Inclusivity
- **Visual Design:** High-contrast color palette conforming to WCAG 2.1 AA standards.
- **Multi-Modal Feedback:** Both visual text and color-coded status indicators (with textual labels, not color alone).

---

## G. Scope, Boundaries & Limitations

### 1. Supported Scope (MVP)
- **Target Language:** American Sign Language (ASL) Fingerspelling.
- **Vocabulary:** 24 static letters: `A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y`.
- **Special Controls:** Virtual gesture or UI actions for `SPACE`, `BACKSPACE`, and `CLEAR`.
- **Input Types:** Single-hand RGB webcam stream ($640 \times 480$) and uploaded still images.
- **Operating Systems:** Windows 10/11, macOS 12+, Ubuntu 20.04+.

### 2. Explicitly Excluded (Out of Scope for MVP)
- **Motion-Dependent Letters:** Letters `J` and `Z` (which require dynamic tracing trajectories in ASL) are classified as unsupported or handled via future sequence models.
- **Continuous Conversational Signing:** No sentence parsing, idiom recognition, or non-manual marker translation.
- **Two-Handed Signs:** Signs requiring dual-hand interaction are deferred to future iterations.
- **Other National Sign Languages:** Pakistan Sign Language (PSL), British Sign Language (BSL), etc., are structurally distinct and not covered by the ASL model.

---

## H. User Stories & Use Cases

### User Stories
- **US-01 (Deaf User):** *As a deaf individual*, I want to show a fingerspelled handshape to my webcam and see the corresponding letter transcribed instantly into text, so that I can communicate specific names and codes to a hearing person.
- **US-02 (Sign Language Student):** *As a student learning ASL*, I want to see a confidence score and a hand landmark skeleton overlay, so that I can verify whether my finger positioning is anatomically correct.
- **US-03 (Classroom Instructor):** *As an accessibility instructor*, I want to clear the accumulated text buffer and adjust the confidence threshold via the UI, so that I can demonstrate how model sensitivity affects classification errors.
- **US-04 (Privacy-Conscious User):** *As a user concerned about biometric privacy*, I want all image processing to run locally on my machine without internet transmission, so that my facial and hand video remains completely private.

### Detailed Use Cases

#### Use Case UC-01: Real-Time Fingerspelling to Text
- **Primary Actor:** End User
- **Preconditions:** Camera connected, application running, model loaded successfully.
- **Main Success Scenario:**
  1. User clicks "Start Camera" on the web interface.
  2. Webcam stream displays live video with optional landmark skeleton overlay.
  3. User raises hand and holds the ASL sign for 'B'.
  4. System detects hand landmarks, normalizes coordinates, and classifies frame as 'B' with 94% confidence.
  5. The prediction remains consistent across 5 consecutive frames.
  6. The letter 'B' is committed to the accumulated text box.
  7. A cooldown timer prevents duplicate 'B' characters while the hand remains still.
  8. User lowers hand or changes handshape to 'A'.
  9. System accepts 'A' and appends it: `"BA"`.
- **Extensions / Failure Scenarios:**
  - *Low Confidence:* Prediction confidence is 58% ($< 70\%$). System displays `"UNCERTAIN"` in yellow and does not append text.
  - *Hand Missing / Dropped:* User lowers hand. System immediately displays `"NO HAND DETECTED"`, resets the temporal buffer, and preserves previously committed text.

---

## I. User Interface Requirements

### 1. Visual Layout Specification
The interface follows a clean, accessible two-column responsive layout:
- **Left Column (Visual Stream & Diagnostics):**
  - Live video feed showing input frame and landmark skeleton overlay.
  - Real-time prediction banner (e.g., `Predicted: B (92.4%)`).
  - System status pill (`DETECTING`, `UNCERTAIN`, `NO HAND`, `COOLDOWN`).
  - Toggle switch: `Show Landmark Skeleton`.
- **Right Column (Text Output & Controls):**
  - Large accumulated text output box with high-contrast monospace typography.
  - Action buttons: `[Space]`, `[Backspace]`, `[Clear All]`, `[Copy Text]`.
  - Settings Accordion: `Confidence Threshold Slider (0.50 - 0.95)`, `Smoothing Window Size (3 - 15 frames)`.
  - Image Upload Tab: File dropper for static image inference.
  - Prominent ethical disclaimer banner at bottom of page.

### 2. UI Framework Decision Matrix

| Framework | Latency & Performance | Ease of Local Deployment | Browser Accessibility | Final Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Gradio** | Excellent for local client streaming; native webcam and image blocks. | Very High (pure Python, one command launch). | High (standard browser UI, screen-reader friendly). | **Recommended Primary Framework** |
| **Streamlit** | Good, but WebRTC streaming components require complex third-party wrappers (`streamlit-webrtc`). | Medium (state management re-executes whole script). | High. | Secondary Alternative |
| **OpenCV GUI (`cv2.imshow`)** | Ultra-low latency ($< 5\text{ ms}$). | Requires local desktop X11/Win32 display server; no web access. | Poor (no accessible DOM or screen reader integration). | Diagnostic / Developer Mode Only |

---

## J. Testing and Acceptance Criteria

### 1. Traceability Matrix

| Requirement | Test Category | Specific Test Case | Expected Result | Pass/Fail Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **FR-01** | Functional | `test_webcam_stream_init` | Frame generator yields valid $640 \times 480 \times 3$ uint8 arrays. | No camera crash; frames valid. |
| **FR-03** | Integration | `test_hand_detection_present` | Pass synthetic/real hand image to detector. | Returns 21 landmarks with $(x, y, z)$. |
| **FR-03** | Negative | `test_hand_detection_absent` | Pass blank/black image to detector. | Returns `None`; status is `NO_HAND`. |
| **FR-04** | Unit | `test_landmark_normalization` | Pass translated and scaled landmarks. | Output feature vector is centered at origin and scaled to unit bounds. |
| **FR-06** | Unit | `test_confidence_filtering` | Feed probability $0.62$ with threshold $0.70$. | Returns `is_valid=False`, label=`UNCERTAIN`. |
| **FR-07** | Unit | `test_temporal_smoothing` | Feed sequence `['A', 'A', 'B', 'A', 'A']`. | Majority vote yields `'A'`. |
| **FR-08** | Unit | `test_duplicate_suppression` | Feed stable `'A'` across 30 frames with no gap. | Appends `'A'` exactly once. |
| **FR-10** | Functional | `test_text_clear_and_reset` | Call `reset()` on TextAccumulator. | Buffer length becomes 0; history purged. |
| **FR-11** | System | `test_missing_model_startup` | Launch with missing model file. | Clean `FileNotFoundError` with user instructions. |

---

## K. Definition of Done (DoD)
The MVP implementation is formally complete when:
1. All Must-Have functional requirements (FR-01 to FR-11) pass verified unit and integration tests.
2. The model achieves $\ge 88\%$ top-1 accuracy on the held-out validation set and runs at $\ge 15$ FPS.
3. Live webcam text entry can spell a test word (e.g., `"HELLO"`) with stable duplicate suppression.
4. The Gradio web interface launches locally with zero unhandled exceptions.
5. Codebase contains 100% typed functions, Google-style docstrings, and comprehensive documentation.
