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
    # 1) Exact match
    gc_ckpt = GC_MODEL_PATH / "best_dice.pth"
    if gc_ckpt.exists():
        print(f"[CKPT] Found: {gc_ckpt}")
        return str(gc_ckpt)

    # 2) Any .pth at top level
    if GC_MODEL_PATH.exists():
        for p in GC_MODEL_PATH.glob("*.pth"):
            print(f"[CKPT] Found: {p}")
            return str(p)

        # 3) Recursive search
        for p in GC_MODEL_PATH.rglob("*.pth"):
            print(f"[CKPT] Found (recursive): {p}")
            return str(p)

        # Debug: list contents
        print(f"[CKPT] Contents of {GC_MODEL_PATH}:")
        for p in GC_MODEL_PATH.rglob("*"):
            print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)" if p.is_file() else f"  {p}/")

    # 4) Local fallback
    if LOCAL_CHECKPOINT_PATH.exists():
        print(f"[CKPT] Local fallback: {LOCAL_CHECKPOINT_PATH}")
        return str(LOCAL_CHECKPOINT_PATH)

    raise FileNotFoundError(
        f"No checkpoint found at {GC_MODEL_PATH} or {LOCAL_CHECKPOINT_PATH}"
    )


DEFAULT_CONFIG_CANDIDATES = [
    "sam2.1_hiera_t512.yaml",  # 与v3训练/推理默认配置保持一致
    "sam2_hiera_t.yaml",
    "sam2_hiera_s.yaml",
]


def _get_config_candidates() -> list[str]:
    """
    Resolve config candidates from env and defaults.
    You can force a specific config via SAM2_CONFIG_NAME.
    """
    forced = os.getenv("SAM2_CONFIG_NAME", "").strip()
    if forced:
        return [forced]
    return DEFAULT_CONFIG_CANDIDATES.copy()



def _to_thw(arr: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    """Convert a 2D/3D array to (T, H, W) and return original layout tag."""
    if arr.ndim == 2:
        return arr[None, ...], "THW"
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got shape={arr.shape}")

    d0, d1, d2 = arr.shape

    # Common GC wrapper layout for this challenge: (W, H, T) with W ~= H.
    if d0 == d1 and d2 != d0:
        return arr.transpose(2, 1, 0), "HWT"

    # Already (T, H, W) when H ~= W.
    if d1 == d2 and d0 != d1:
        return arr, "THW"

    # Fallback: infer shortest axis as time.
    t_axis = int(np.argmin(arr.shape))
    if t_axis == 0:
        return arr, "THW"
    if t_axis == 2:
        return arr.transpose(2, 1, 0), "HWT"
    # Rare case: (H, T, W) -> (T, H, W)
    return arr.transpose(1, 0, 2), "HTW"



def _from_thw(arr_thw: np.ndarray, layout: str) -> np.ndarray:
    """Restore (T, H, W) array back to original layout."""
    if layout == "THW":
        return arr_thw
    if layout == "HWT":
        return arr_thw.transpose(2, 1, 0)
    if layout == "HTW":
        return arr_thw.transpose(1, 0, 2)
    raise ValueError(f"Unknown layout tag: {layout}")



def _pick_prompt_mask(target_thw: np.ndarray) -> tuple[np.ndarray, int]:
    """Pick first non-empty target frame as 2D prompt mask and frame index."""
    if target_thw.ndim == 2:
        mask = target_thw
        frame_idx = 0
    elif target_thw.ndim == 3:
        flat_sum = target_thw.reshape(target_thw.shape[0], -1).sum(axis=1)
        nz = np.where(flat_sum > 0)[0]
        frame_idx = int(nz[0]) if nz.size > 0 else 0
        print(f"[PROMPT] Using target frame index={frame_idx}")
        mask = target_thw[frame_idx]
    else:
        raise ValueError(f"target must be 2D/3D, got shape={target_thw.shape}")
    return (mask > 0).astype(np.uint8), frame_idx



def _coerce_pred_hw(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert predictor output to binary (H, W)."""
    out = np.asarray(pred)
    while out.ndim > 2:
        out = out[0]
    if out.shape != (h, w):
        out_t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out_t = F.interpolate(out_t, size=(h, w), mode="nearest")
        out = out_t[0, 0].numpy()
    return (out > 0).astype(np.uint8)



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
        frames: challenge input series array
        target: challenge input first-frame mask (or a full mask sequence)
    Returns:
        output: same layout as input `frames`
    """
    print(f"[INPUT] frames.shape = {frames.shape}, target.shape = {target.shape}")

    # Normalize inputs to canonical (T, H, W)
    frames_thw, frames_layout = _to_thw(frames, "frames")
    target_thw, _ = _to_thw(target, "target")

    T, H, W = frames_thw.shape
    print(f"[PROC] Canonical THW: frames ({T}, {H}, {W}), target {target_thw.shape}")

    mask_np, prompt_frame_idx = _pick_prompt_mask(target_thw)

    # Hydra initialization is handled inside sam2_train/__init__.py.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    ckpt_path = _find_checkpoint()

    config_candidates = _get_config_candidates()
    print(f"[MODEL] Try config candidates: {config_candidates}")
    last_error = None
    predictor = None
    try:
        from sam2_train.build_sam import build_sam2_video_predictor
        for cfg_name in config_candidates:
            try:
                print(f"[MODEL] Trying config: {cfg_name}")
                predictor = build_sam2_video_predictor(
                    config_file=cfg_name,
                    ckpt_path=ckpt_path,
                    device=device,
                )
                print(f"[MODEL] Loaded with {cfg_name}, image_size={predictor.image_size}")
                break
            except Exception as e:
                last_error = e
                print(f"[MODEL WARN] Failed with {cfg_name}: {e}")
        if predictor is None:
            raise RuntimeError(
                f"Failed to load checkpoint with configs={config_candidates}. "
                f"Last error: {last_error}"
            )
    except Exception as e:
        print(f"[MODEL ERROR] {e}")
        raise

    # ==================================================================
    # Preprocess frames
    # ==================================================================
    frames_float = frames_thw.astype(np.float64)
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
            frame_idx=prompt_frame_idx,
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
            # obj_id=1 (from add_new_mask); obj_ids may be list or tuple
            obj_ids_list = list(out_obj_ids) if hasattr(out_obj_ids, "__iter__") else [out_obj_ids]
            if 1 in obj_ids_list:
                idx = obj_ids_list.index(1)
                logits = out_mask_logits[idx]
                pred = _coerce_pred_hw((logits > 0.0).cpu().numpy().astype(np.uint8), H, W)
                video_segments[out_frame_idx] = pred
            else:
                print(f"[PROPAGATE] Frame {out_frame_idx}: obj_ids={obj_ids_list} (expected 1)")
    except Exception as e:
        print(f"[PROPAGATE ERROR] {e}")
        raise

    print(f"[PROPAGATE] Got masks for {len(video_segments)}/{T} frames")

    # ==================================================================
    # Construct output in canonical (T, H, W)
    # ==================================================================
    output_thw = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        if t in video_segments:
            output_thw[t] = video_segments[t]
        elif t == 0:
            output_thw[t] = mask_np

    # Always return (H, W, T) to match evaluation: transpose(2,0,1) -> (T, H, W)
    # Do NOT use _from_thw - GC evaluation expects (H, W, T) regardless of input layout
    output = output_thw.transpose(1, 2, 0)  # (T, H, W) -> (H, W, T)

    print(f"[OUTPUT] shape={output.shape}, unique={np.unique(output)}")
    return output.astype(np.uint8)
