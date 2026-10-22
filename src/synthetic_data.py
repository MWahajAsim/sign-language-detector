"""Synthetic realistic hand landmark dataset generator.

Constructs anatomically grounded 21-landmark poses for all 24 static ASL fingerspelling
letters, simulating variations across multiple signers, bone lengths, noise, and tilts.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.preprocessing import LandmarkNormalizer


def get_base_hand_pose(letter: str) -> np.ndarray:
    """Generate canonical 3D coordinates (21, 3) for a given ASL letter.

    Coordinates are roughly modeled in a right-hand coordinate system:
    Wrist at (0, 0, 0), Y pointing up towards fingertips, X pointing towards pinky,
    Z pointing out from palm.
    """
    letter = letter.upper()
    coords = np.zeros((21, 3), dtype=np.float32)

    # 0: Wrist
    coords[0] = [0.0, 0.0, 0.0]

    # Knuckle base positions (MCPs) relative to wrist
    coords[1] = [-0.18, 0.15, 0.05]   # Thumb CMC
    coords[2] = [-0.28, 0.32, 0.08]   # Thumb MCP
    coords[5] = [-0.14, 0.55, 0.00]   # Index MCP
    coords[9] = [ 0.00, 0.58, 0.00]   # Middle MCP
    coords[13] = [ 0.13, 0.54, 0.00]  # Ring MCP
    coords[17] = [ 0.24, 0.48, 0.00]  # Pinky MCP

    # Helper function to generate finger joints
    def build_finger(base_idx: int, mcp_pos: np.ndarray, direction: np.ndarray, seg_len: float):
        # Generates PIP (base+1), DIP (base+2), TIP (base+3)
        dir_norm = direction / np.linalg.norm(direction)
        coords[base_idx + 1] = mcp_pos + dir_norm * seg_len * 1.0  # PIP
        coords[base_idx + 2] = coords[base_idx + 1] + dir_norm * seg_len * 0.8  # DIP
        coords[base_idx + 3] = coords[base_idx + 2] + dir_norm * seg_len * 0.7  # TIP

    def build_curled_finger(base_idx: int, mcp_pos: np.ndarray, curl_inwards: float = 0.22):
        # A finger curled in towards the palm (fist)
        coords[base_idx + 1] = mcp_pos + np.array([0.0, 0.15, 0.10])
        coords[base_idx + 2] = mcp_pos + np.array([0.0, 0.08, 0.20])
        coords[base_idx + 3] = mcp_pos + np.array([0.0, -curl_inwards, 0.15])

    # Default directions
    up = np.array([0.0, 1.0, 0.0])
    thumb_up = np.array([-0.05, 1.0, 0.05])
    thumb_side = np.array([-1.0, 0.1, 0.0])
    thumb_across = np.array([0.4, 0.2, 0.12])

    if letter == "A":
        # Fist with thumb upright resting alongside index
        build_curled_finger(5, coords[5])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        # Thumb straight along index
        coords[3] = coords[2] + thumb_up * 0.15
        coords[4] = coords[3] + thumb_up * 0.14

    elif letter == "B":
        # 4 fingers straight up, thumb tucked across palm
        build_finger(5, coords[5], up, 0.18)
        build_finger(9, coords[9], up, 0.19)
        build_finger(13, coords[13], up, 0.18)
        build_finger(17, coords[17], up, 0.15)
        # Thumb across palm
        coords[3] = coords[2] + thumb_across * 0.16
        coords[4] = coords[3] + thumb_across * 0.14

    elif letter == "C":
        # Curved fingers and thumb forming C shape
        curve_dir = np.array([0.0, 0.7, 0.7])
        build_finger(5, coords[5], curve_dir, 0.15)
        build_finger(9, coords[9], curve_dir, 0.15)
        build_finger(13, coords[13], curve_dir, 0.14)
        build_finger(17, coords[17], curve_dir, 0.12)
        thumb_c = np.array([0.1, 0.5, 0.5])
        coords[3] = coords[2] + thumb_c * 0.15
        coords[4] = coords[3] + thumb_c * 0.14

    elif letter == "D":
        # Index straight up, other fingers curled touching thumb tip
        build_finger(5, coords[5], up, 0.18)
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + np.array([0.3, 0.3, 0.15]) * 0.16
        coords[4] = coords[9 + 3]  # touches middle tip

    elif letter == "E":
        # All fingertips curled down tightly, thumb tucked below fingertips
        build_curled_finger(5, coords[5], 0.28)
        build_curled_finger(9, coords[9], 0.28)
        build_curled_finger(13, coords[13], 0.28)
        build_curled_finger(17, coords[17], 0.28)
        coords[3] = coords[2] + np.array([0.3, 0.05, 0.15]) * 0.15
        coords[4] = coords[3] + np.array([0.3, -0.05, 0.15]) * 0.12

    elif letter == "F":
        # Index and thumb form circle, middle, ring, pinky straight up
        coords[3] = coords[2] + np.array([0.2, 0.3, 0.15]) * 0.15
        coords[4] = coords[3] + np.array([0.2, 0.2, 0.10]) * 0.14
        # Index curls to meet thumb
        coords[6] = coords[5] + np.array([0.0, 0.2, 0.1])
        coords[7] = coords[6] + np.array([-0.05, 0.1, 0.05])
        coords[8] = coords[4]  # Touch thumb tip
        build_finger(9, coords[9], up, 0.19)
        build_finger(13, coords[13], up, 0.18)
        build_finger(17, coords[17], up, 0.15)

    elif letter == "G":
        # Index pointing horizontally, thumb parallel, other 3 curled
        h_dir = np.array([0.8, 0.4, 0.0])
        build_finger(5, coords[5], h_dir, 0.17)
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + h_dir * 0.15
        coords[4] = coords[3] + h_dir * 0.13

    elif letter == "H":
        # Index and middle pointing horizontally together, others curled
        h_dir = np.array([0.8, 0.4, 0.0])
        build_finger(5, coords[5], h_dir, 0.17)
        build_finger(9, coords[9], h_dir, 0.17)
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + np.array([0.2, 0.1, 0.1]) * 0.14
        coords[4] = coords[3] + np.array([0.2, 0.1, 0.1]) * 0.12

    elif letter == "I":
        # Pinky straight up, other fingers curled in fist
        build_curled_finger(5, coords[5])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_finger(17, coords[17], up, 0.15)
        coords[3] = coords[2] + thumb_across * 0.15
        coords[4] = coords[3] + thumb_across * 0.12

    elif letter == "K":
        # Index straight up, middle points forward/up at angle, thumb rests between them
        build_finger(5, coords[5], up, 0.18)
        k_dir = np.array([0.1, 0.8, 0.4])
        build_finger(9, coords[9], k_dir, 0.18)
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + np.array([0.2, 0.3, 0.2]) * 0.16
        coords[4] = coords[3] + np.array([0.1, 0.2, 0.1]) * 0.13

    elif letter == "L":
        # L-shape: thumb out to side, index straight up, others curled
        build_finger(5, coords[5], up, 0.18)
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + thumb_side * 0.18
        coords[4] = coords[3] + thumb_side * 0.16

    elif letter == "M":
        # Fist with thumb tucked beneath index, middle, and ring
        build_curled_finger(5, coords[5])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        # Thumb reaches under pinky/ring
        coords[3] = coords[2] + np.array([0.5, 0.2, -0.05]) * 0.16
        coords[4] = coords[3] + np.array([0.3, 0.1, -0.05]) * 0.14

    elif letter == "N":
        # Fist with thumb tucked beneath index and middle only
        build_curled_finger(5, coords[5])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + np.array([0.35, 0.2, -0.05]) * 0.15
        coords[4] = coords[3] + np.array([0.25, 0.1, -0.05]) * 0.13

    elif letter == "O":
        # O-shape: all fingers curved to meet thumb
        o_dir = np.array([0.1, 0.6, 0.6])
        build_finger(5, coords[5], o_dir, 0.14)
        build_finger(9, coords[9], o_dir, 0.14)
        build_finger(13, coords[13], o_dir, 0.13)
        build_finger(17, coords[17], o_dir, 0.12)
        coords[3] = coords[2] + np.array([0.2, 0.4, 0.4]) * 0.15
        coords[4] = coords[8]  # Meets index tip

    elif letter == "P":
        # Like K, but pointed downwards
        down = np.array([0.2, -0.8, 0.2])
        p_mid = np.array([0.4, -0.7, 0.4])
        build_finger(5, coords[5], down, 0.17)
        build_finger(9, coords[9], p_mid, 0.17)
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + np.array([0.2, -0.2, 0.1]) * 0.15
        coords[4] = coords[3] + np.array([0.1, -0.1, 0.1]) * 0.12

    elif letter == "Q":
        # Like G, but pointed downwards
        down = np.array([0.3, -0.9, 0.0])
        build_finger(5, coords[5], down, 0.17)
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + down * 0.15
        coords[4] = coords[3] + down * 0.12

    elif letter == "R":
        # Index and middle crossed and straight up, others curled
        r_index = np.array([0.08, 1.0, -0.05])
        r_middle = np.array([-0.08, 1.0, 0.05])
        build_finger(5, coords[5], r_index, 0.18)
        build_finger(9, coords[9], r_middle, 0.18)
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + thumb_across * 0.15
        coords[4] = coords[3] + thumb_across * 0.12

    elif letter == "S":
        # Tight fist with thumb crossed in front of fingers
        build_curled_finger(5, coords[5], 0.25)
        build_curled_finger(9, coords[9], 0.25)
        build_curled_finger(13, coords[13], 0.25)
        build_curled_finger(17, coords[17], 0.25)
        s_thumb = np.array([0.4, 0.3, 0.22])
        coords[3] = coords[2] + s_thumb * 0.16
        coords[4] = coords[3] + s_thumb * 0.14

    elif letter == "T":
        # Fist with thumb tucked between index and middle
        build_curled_finger(5, coords[5])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + np.array([0.15, 0.25, 0.15]) * 0.15
        coords[4] = coords[5] + np.array([0.05, 0.20, 0.18])

    elif letter == "U":
        # Index and middle straight up and pressed together
        u_dir = np.array([0.0, 1.0, 0.0])
        build_finger(5, coords[5], u_dir, 0.18)
        build_finger(9, coords[9], u_dir, 0.18)
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + thumb_across * 0.15
        coords[4] = coords[3] + thumb_across * 0.12

    elif letter == "V":
        # Peace sign: index and middle straight up, spread apart
        v_idx = np.array([-0.25, 0.95, 0.0])
        v_mid = np.array([ 0.25, 0.95, 0.0])
        build_finger(5, coords[5], v_idx, 0.18)
        build_finger(9, coords[9], v_mid, 0.18)
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + thumb_across * 0.15
        coords[4] = coords[3] + thumb_across * 0.12

    elif letter == "W":
        # W sign: index, middle, ring straight up and spread apart
        w_idx = np.array([-0.30, 0.95, 0.0])
        w_mid = np.array([ 0.00, 1.00, 0.0])
        w_ring = np.array([ 0.30, 0.95, 0.0])
        build_finger(5, coords[5], w_idx, 0.18)
        build_finger(9, coords[9], w_mid, 0.18)
        build_finger(13, coords[13], w_ring, 0.18)
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + thumb_across * 0.15
        coords[4] = coords[17 + 3]  # touches pinky tip

    elif letter == "X":
        # Index curled like a hook, other fingers in fist
        coords[6] = coords[5] + np.array([0.0, 0.15, 0.05])
        coords[7] = coords[6] + np.array([0.0, 0.08, 0.15])
        coords[8] = coords[7] + np.array([0.0, -0.05, 0.10])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_curled_finger(17, coords[17])
        coords[3] = coords[2] + thumb_across * 0.15
        coords[4] = coords[3] + thumb_across * 0.12

    elif letter == "Y":
        # Hang loose: thumb and pinky extended out, middle 3 curled
        y_thumb = np.array([-0.8, 0.5, 0.1])
        y_pinky = np.array([ 0.7, 0.7, 0.1])
        coords[3] = coords[2] + y_thumb * 0.18
        coords[4] = coords[3] + y_thumb * 0.16
        build_curled_finger(5, coords[5])
        build_curled_finger(9, coords[9])
        build_curled_finger(13, coords[13])
        build_finger(17, coords[17], y_pinky, 0.16)

    else:
        # Fallback open palm
        build_finger(5, coords[5], up, 0.18)
        build_finger(9, coords[9], up, 0.19)
        build_finger(13, coords[13], up, 0.18)
        build_finger(17, coords[17], up, 0.15)
        coords[3] = coords[2] + thumb_up * 0.16
        coords[4] = coords[3] + thumb_up * 0.14

    return coords


def generate_synthetic_dataset(
    output_csv_path: Path,
    classes: List[str] = DEFAULT_CONFIG.signs.classes,
    num_signers: int = 4,
    samples_per_signer_class: int = 60,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate high-fidelity landmark dataset across signers with augmentation.

    Args:
        output_csv_path: Destination CSV path.
        classes: List of class labels to generate.
        num_signers: Number of distinct simulated signers.
        samples_per_signer_class: Number of samples per class per signer.
        random_seed: Reproducibility seed.

    Returns:
        Pandas DataFrame containing normalized features, labels, and signer IDs.
    """
    rng = np.random.RandomState(random_seed)
    records = []

    # Assign each signer unique anatomical proportions (bone scaling variance)
    signer_profiles = []
    for s_id in range(1, num_signers + 1):
        signer_profiles.append({
            "signer_id": f"signer_{s_id:02d}",
            "hand_scale": rng.uniform(0.88, 1.12),
            "finger_length_bias": rng.uniform(-0.03, 0.03),
            "tilt_bias": rng.uniform(-5.0, 5.0),
        })

    for profile in signer_profiles:
        signer_id = profile["signer_id"]

        for label in classes:
            base_pose = get_base_hand_pose(label)

            # Apply signer hand scale and proportions
            signer_base = base_pose * profile["hand_scale"]

            for _ in range(samples_per_signer_class):
                # Apply base normalization
                norm_feat = LandmarkNormalizer.normalize(signer_base)

                # Apply realistic random augmentation
                aug_feat = LandmarkNormalizer.augment(
                    norm_feat,
                    jitter_std=0.015,
                    scale_range=(0.94, 1.06),
                    rotation_angle_deg=10.0,
                    rng=rng,
                )

                row_dict = {"label": label, "signer_id": signer_id}
                for i in range(LandmarkNormalizer.FEATURE_DIM):
                    row_dict[f"feat_{i}"] = float(aug_feat[i])

                records.append(row_dict)

    df = pd.DataFrame(records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df
