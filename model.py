from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# Add current directory to sys.path to ensure we can import sam2_train
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from sam2_train.build_sam import build_sam2_video_predictor

RESOURCE_PATH = Path("resources")
CHECKPOINT_PATH = Path("checkpoints") / "best_dice.pth"
CONFIG_NAME = "sam2_hiera_s.yaml"

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    """
    Implement MedSAM-2 algorithm.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (T, H, W) containing the MRI linac series.
      Note: The docstring in original file said (W, H, T) but SimpleITK loads as (T, H, W).
      We assume (T, H, W) based on inference.py usage.
    - target (numpy.ndarray): A 2D or 3D numpy array containing the MRI linac target.
    """
    
    # 1. Initialize Hydra and Model
    # Use absolute path for config_path to be safe, or relative to this file
    # config is in sam2_train folder
    config_dir = "sam2_train"
    
    if not GlobalHydra.instance().is_initialized():
        # We need to initialize hydra to find the config in sam2_train
        initialize(version_base='1.2', config_path=config_dir, job_name="trackrad_inference")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not CHECKPOINT_PATH.exists():
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
        # Fallback or error? For now print warning.

    predictor = build_sam2_video_predictor(
        config_file=CONFIG_NAME,
        ckpt_path=str(CHECKPOINT_PATH),
        device=device
    )

    # 2. Preprocess Frames
    # frames shape: (T, H, W) (SimpleITK convention)
    # Check if frames need transpose if input was actually (W, H, T)
    # Heuristic: T is usually small (<1000), W/H are usually 256 or 512.
    # If frames.shape[0] is large and frames.shape[2] is small, it might be (W, H, T).
    # But let's trust SimpleITK for now.
    
    T, H, W = frames.shape
    
    # Normalize frames to 0-255 float
    frames_min = frames.min()
    frames_max = frames.max()
    if frames_max > frames_min:
        frames_norm = (frames - frames_min) / (frames_max - frames_min) * 255.0
    else:
        frames_norm = frames.astype(np.float32) # Should be 0s or constant
        
    frames_tensor = torch.from_numpy(frames_norm).float() # (T, H, W)
    
    # Expand to 3 channels (T, 3, H, W)
    frames_tensor = frames_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    
    # Resize to model image size
    target_size = predictor.image_size
    frames_resized = F.interpolate(
        frames_tensor, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 3. Initialize Inference State
    inference_state = predictor.val_init_state(
        frames_resized, 
        video_height=H, 
        video_width=W
    )
    
    # 4. Add Initial Mask (Target)
    # target shape: (1, H, W) or (H, W)
    target_tensor = torch.from_numpy(target)
    if target_tensor.ndim == 3:
        mask = target_tensor[0] # Assume first slice if 3D
    else:
        mask = target_tensor
        
    # mask should be boolean or compatible
    mask = mask > 0
    
    # Add mask to frame 0, object 1
    predictor.add_new_mask(
        inference_state, 
        frame_idx=0, 
        obj_id=1, 
        mask=mask
    )
    
    # 5. Propagate
    video_segments = {} # frame_idx -> mask
    
    # Propagate through the whole video
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # out_mask_logits: (N_obj, H, W)
        # We assume single object (obj_id=1)
        # Find index of obj_id=1? out_obj_ids is a list
        if 1 in out_obj_ids:
            idx = out_obj_ids.index(1)
            mask_pred_logits = out_mask_logits[idx] # (H, W)
            mask_pred = (mask_pred_logits > 0.0).cpu().numpy().astype(np.uint8)
            video_segments[out_frame_idx] = mask_pred
    
    # 6. Construct Output
    output = np.zeros((T, H, W), dtype=np.uint8)
    for t in range(T):
        if t in video_segments:
            output[t] = video_segments[t]
        elif t == 0:
             # Fallback for frame 0 if not returned (it should be)
             output[t] = target_tensor.numpy().astype(np.uint8).squeeze()
             
    return output
