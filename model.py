from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# Add current directory to sys.path to ensure we can import sam2_train
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from sam2_train.build_sam import build_sam2_video_predictor

RESOURCE_PATH = Path("resources")

# ============================================================
# Checkpoint path: Grand Challenge uploads model weights to
# /opt/ml/model/. Fall back to local checkpoints/ for testing.
# ============================================================
GC_MODEL_PATH = Path("/opt/ml/model")
LOCAL_CHECKPOINT_PATH = current_dir / "checkpoints" / "best_dice.pth"

def _find_checkpoint() -> str:
    """Locate the model checkpoint, preferring Grand Challenge path."""
    # Grand Challenge: weights uploaded via Algorithm Models
    gc_ckpt = GC_MODEL_PATH / "best_dice.pth"
    if gc_ckpt.exists():
        print(f"Using Grand Challenge checkpoint: {gc_ckpt}")
        return str(gc_ckpt)
    
    # Also check if any .pth file exists under /opt/ml/model/
    if GC_MODEL_PATH.exists():
        pth_files = list(GC_MODEL_PATH.glob("*.pth"))
        if pth_files:
            print(f"Using Grand Challenge checkpoint: {pth_files[0]}")
            return str(pth_files[0])
    
    # Local fallback for testing
    if LOCAL_CHECKPOINT_PATH.exists():
        print(f"Using local checkpoint: {LOCAL_CHECKPOINT_PATH}")
        return str(LOCAL_CHECKPOINT_PATH)
    
    raise FileNotFoundError(
        f"No checkpoint found. Searched:\n"
        f"  - {gc_ckpt}\n"
        f"  - {GC_MODEL_PATH}/*.pth\n"
        f"  - {LOCAL_CHECKPOINT_PATH}"
    )


CONFIG_NAME = "sam2_hiera_s.yaml"


def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:
    """
    MedSAM-2 based tumor tracking algorithm.

    Dimension convention (from official baseline documentation):
        frames.shape == (W, H, T)   — but via SimpleITK GetArrayFromImage
                                       the numpy array is actually (T, H, W)
        target.shape == (W, H, 1)   — actually (1, H, W) in numpy

    Since inference.py uses SimpleITK for both reading and writing, the
    actual numpy convention is (T, H, W) / (1, H, W) throughout. We
    process in this convention and return the same shape.

    IMPORTANT: The output shape must match the input frames shape.
    """

    # ------------------------------------------------------------------
    # 1. Initialize Hydra and build model
    # ------------------------------------------------------------------
    # Use absolute path for config_dir to avoid working-directory issues
    config_abs_path = str(current_dir / "sam2_train")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize_config_dir(
        version_base="1.2",
        config_dir=config_abs_path,
        job_name="trackrad_inference",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = _find_checkpoint()

    predictor = build_sam2_video_predictor(
        config_file=CONFIG_NAME,
        ckpt_path=ckpt_path,
        device=device,
    )

    # ------------------------------------------------------------------
    # 2. Determine dimensions
    # ------------------------------------------------------------------
    # Numpy array from SimpleITK: shape is (T, H, W)
    # (Official docs label this as (W, H, T) but axis order is the same)
    ndim = frames.ndim
    assert ndim == 3, f"Expected 3D frames array, got {ndim}D with shape {frames.shape}"

    # We treat axis-0 as T (time), axis-1 as H, axis-2 as W
    # This is consistent with SimpleITK's GetArrayFromImage output
    T, H, W = frames.shape
    print(f"Frames shape (T, H, W): ({T}, {H}, {W})")
    print(f"Target shape: {target.shape}")

    # ------------------------------------------------------------------
    # 3. Preprocess frames
    # ------------------------------------------------------------------
    # Normalize frames to 0-255 float
    frames_float = frames.astype(np.float64)
    fmin, fmax = frames_float.min(), frames_float.max()
    if fmax > fmin:
        frames_norm = (frames_float - fmin) / (fmax - fmin) * 255.0
    else:
        frames_norm = np.zeros_like(frames_float)

    frames_tensor = torch.from_numpy(frames_norm).float()  # (T, H, W)

    # Expand to 3 channels: (T, 3, H, W)
    frames_tensor = frames_tensor.unsqueeze(1).repeat(1, 3, 1, 1)

    # Resize to model input size
    target_size = predictor.image_size  # e.g. 1024
    frames_resized = F.interpolate(
        frames_tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    # ------------------------------------------------------------------
    # 4. Initialize inference state
    # ------------------------------------------------------------------
    inference_state = predictor.val_init_state(
        frames_resized,
        video_height=H,
        video_width=W,
    )

    # ------------------------------------------------------------------
    # 5. Prepare and add initial mask (target on frame 0)
    # ------------------------------------------------------------------
    # target shape from SimpleITK: (1, H, W)
    # Could also be (H, W) if squeezed somewhere
    target_tensor = torch.from_numpy(target.astype(np.float32))

    if target_tensor.ndim == 3:
        # (1, H, W) -> take first slice to get (H, W)
        mask = target_tensor[0]
    elif target_tensor.ndim == 2:
        mask = target_tensor
    else:
        raise ValueError(f"Unexpected target shape: {target.shape}")

    # Binarize
    mask = (mask > 0)

    print(f"Initial mask shape: {mask.shape}, nonzero pixels: {mask.sum().item()}")

    # Add mask on frame 0, object ID 1
    predictor.add_new_mask(
        inference_state,
        frame_idx=0,
        obj_id=1,
        mask=mask,
    )

    # ------------------------------------------------------------------
    # 6. Propagate through video (causal, forward only)
    # ------------------------------------------------------------------
    video_segments = {}  # frame_idx -> binary mask (H, W)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        # out_mask_logits: (N_obj, H, W)
        if 1 in out_obj_ids:
            idx = out_obj_ids.index(1)
            mask_logits = out_mask_logits[idx]  # (H, W)
            mask_pred = (mask_logits > 0.0).cpu().numpy().astype(np.uint8)
            video_segments[out_frame_idx] = mask_pred

    # ------------------------------------------------------------------
    # 7. Construct output array — same shape as input frames
    # ------------------------------------------------------------------
    output = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        if t in video_segments:
            output[t] = video_segments[t]
        elif t == 0:
            # Fallback: use the input mask for frame 0 if propagation
            # did not return it (it normally should)
            output[t] = mask.cpu().numpy().astype(np.uint8)

    print(f"Output shape: {output.shape}, unique values: {np.unique(output)}")
    return output
