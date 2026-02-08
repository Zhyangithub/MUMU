from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to sys.path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

# ============================================================
# Checkpoint: Grand Challenge -> /opt/ml/model/
# ============================================================
GC_MODEL_PATH = Path("/opt/ml/model")
LOCAL_CHECKPOINT_PATH = current_dir / "checkpoints" / "best_dice.pth"


def _find_checkpoint() -> str:
    """Locate the model checkpoint, preferring Grand Challenge path."""
    # 1. Exact match
    gc_ckpt = GC_MODEL_PATH / "best_dice.pth"
    if gc_ckpt.exists():
        print(f"[CKPT] Found: {gc_ckpt}")
        return str(gc_ckpt)

    # 2. Any .pth at top level
    if GC_MODEL_PATH.exists():
        for p in GC_MODEL_PATH.glob("*.pth"):
            print(f"[CKPT] Found: {p}")
            return str(p)

        # 3. Recursive search
        for p in GC_MODEL_PATH.rglob("*.pth"):
            print(f"[CKPT] Found (recursive): {p}")
            return str(p)

        # Debug: list contents
        print(f"[CKPT] Contents of {GC_MODEL_PATH}:")
        for p in GC_MODEL_PATH.rglob("*"):
            print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)" if p.is_file() else f"  {p}/")

    # 4. Local fallback
    if LOCAL_CHECKPOINT_PATH.exists():
        print(f"[CKPT] Local fallback: {LOCAL_CHECKPOINT_PATH}")
        return str(LOCAL_CHECKPOINT_PATH)

    raise FileNotFoundError(
        f"No checkpoint found at {GC_MODEL_PATH} or {LOCAL_CHECKPOINT_PATH}"
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

    Args:
        frames: (W, H, T) — official convention from inference.py
        target: (W, H, 1) — official convention from inference.py
    Returns:
        output: (W, H, T) — must match input frames shape
    """
    print(f"[INPUT] frames.shape = {frames.shape}, target.shape = {target.shape}")

    # ==================================================================
    # CRITICAL FIX 1: Transpose input (W, H, T) -> (T, H, W) for processing
    # ==================================================================
    frames = frames.transpose(2, 1, 0)   # (W, H, T) -> (T, H, W)
    target = target.transpose(2, 1, 0)   # (W, H, 1) -> (1, H, W)

    T, H, W = frames.shape
    print(f"[PROC] After transpose: frames ({T}, {H}, {W}), target {target.shape}")

    # Extract 2D mask from target
    if target.ndim == 3:
        mask_np = target[0]  # (1, H, W) -> (H, W)
    else:
        mask_np = target
    mask_np = (mask_np > 0).astype(np.uint8)

    # ==================================================================
    # Initialize Hydra + Model (with error handling)
    # ==================================================================
    try:
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_abs_path = str(current_dir / "sam2_train")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        initialize_config_dir(
            version_base="1.2",
            config_dir=config_abs_path,
            job_name="trackrad_inference",
        )
        print(f"[HYDRA] Initialized with config_dir={config_abs_path}")
    except Exception as e:
        print(f"[HYDRA ERROR] {e}")
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    ckpt_path = _find_checkpoint()

    try:
        from sam2_train.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            config_file=CONFIG_NAME,
            ckpt_path=ckpt_path,
            device=device,
        )
        print(f"[MODEL] Loaded, image_size={predictor.image_size}")
    except Exception as e:
        print(f"[MODEL ERROR] {e}")
        raise

    # ==================================================================
    # Preprocess frames
    # ==================================================================
    frames_float = frames.astype(np.float64)
    fmin, fmax = frames_float.min(), frames_float.max()
    if fmax > fmin:
        frames_norm = (frames_float - fmin) / (fmax - fmin) * 255.0
    else:
        frames_norm = np.zeros_like(frames_float)

    frames_tensor = torch.from_numpy(frames_norm).float()  # (T, H, W)
    frames_tensor = frames_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # (T, 3, H, W)

    img_size = predictor.image_size
    frames_resized = F.interpolate(
        frames_tensor,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    print(f"[PREPROC] frames_resized: {frames_resized.shape}")

    # ==================================================================
    # Initialize inference state + add mask
    # ==================================================================
    try:
        inference_state = predictor.val_init_state(
            frames_resized,
            video_height=H,
            video_width=W,
        )

        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)) > 0
        print(f"[MASK] shape={mask_tensor.shape}, nonzero={mask_tensor.sum().item()}")

        predictor.add_new_mask(
            inference_state,
            frame_idx=0,
            obj_id=1,
            mask=mask_tensor,
        )
    except Exception as e:
        print(f"[INIT ERROR] {e}")
        raise

    # ==================================================================
    # Propagate (causal, forward only)
    # ==================================================================
    video_segments = {}

    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            if 1 in out_obj_ids:
                idx = out_obj_ids.index(1)
                logits = out_mask_logits[idx]
                pred = (logits > 0.0).cpu().numpy().astype(np.uint8)
                video_segments[out_frame_idx] = pred
    except Exception as e:
        print(f"[PROPAGATE ERROR] {e}")
        raise

    print(f"[PROPAGATE] Got masks for {len(video_segments)}/{T} frames")

    # ==================================================================
    # Construct output in (T, H, W)
    # ==================================================================
    output = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        if t in video_segments:
            output[t] = video_segments[t]
        elif t == 0:
            output[t] = mask_np

    # ==================================================================
    # CRITICAL FIX 2: Transpose output (T, H, W) -> (W, H, T) to match official format
    # ==================================================================
    output = output.transpose(2, 1, 0)  # (T, H, W) -> (W, H, T)

    print(f"[OUTPUT] shape={output.shape}, unique={np.unique(output)}")
    return output.astype(np.uint8)
