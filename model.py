from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to sys.path to ensure we can import sam2_train
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from sam2_train.build_sam import build_sam2_video_predictor

# ============================================================
# 配置路径
# ============================================================
# 如果使用 Grand Challenge 的 Algorithm Models 上传模型权重，
# 运行时会解压到 /opt/ml/model/
# 如果把 checkpoints 直接放在 Docker 镜像里，用下面的本地路径
CHECKPOINT_PATH_GC = Path("/opt/ml/model/best_dice.pth")
CHECKPOINT_PATH_LOCAL = Path("/opt/app/checkpoints/best_dice.pth")

CONFIG_NAME = "sam2_hiera_s.yaml"


def get_checkpoint_path():
    """按优先级查找 checkpoint 路径"""
    if CHECKPOINT_PATH_GC.exists():
        return str(CHECKPOINT_PATH_GC)
    elif CHECKPOINT_PATH_LOCAL.exists():
        return str(CHECKPOINT_PATH_LOCAL)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH_GC} or {CHECKPOINT_PATH_LOCAL}"
        )


def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:
    """
    MedSAM-2 based algorithm for TrackRAD2025.

    Args:
    - frames (numpy.ndarray): shape (W, H, T) — the cine-MRI series.
    - target (numpy.ndarray): shape (W, H, 1) — the initial tumor segmentation mask.
    - frame_rate (float): Frame rate of the MRI series.
    - magnetic_field_strength (float): B-field strength (0.35 or 1.5).
    - scanned_region (str): Anatomical region scanned.

    Returns:
    - numpy.ndarray: shape (W, H, T) — predicted segmentation masks.
    """

    # ============================================================
    # 0. 解析维度
    # ============================================================
    # 官方约定: frames.shape == (W, H, T), target.shape == (W, H, 1)
    W, H, T = frames.shape

    # 转成 (T, H, W) 方便 PyTorch 处理（标准图像格式）
    # frames: (W, H, T) -> transpose -> (T, H, W)
    frames_thw = np.transpose(frames, (2, 1, 0))  # (T, H, W)

    # target: (W, H, 1) -> 取第一帧 -> (W, H) -> 转置 -> (H, W)
    mask_hw = target[:, :, 0].T  # (H, W)

    # ============================================================
    # 1. 初始化模型
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = get_checkpoint_path()

    predictor = build_sam2_video_predictor(
        config_file=CONFIG_NAME,
        ckpt_path=ckpt_path,
        device=device,
    )

    # ============================================================
    # 2. 预处理帧 -> (T, 3, image_size, image_size)
    # ============================================================
    # 归一化到 0-255
    frames_min = frames_thw.min()
    frames_max = frames_thw.max()
    if frames_max > frames_min:
        frames_norm = (frames_thw - frames_min) / (frames_max - frames_min) * 255.0
    else:
        frames_norm = np.zeros_like(frames_thw, dtype=np.float32)

    # (T, H, W) -> torch tensor -> (T, 1, H, W) -> (T, 3, H, W)
    frames_tensor = torch.from_numpy(frames_norm).float()
    frames_tensor = frames_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # (T, 3, H, W)

    # Resize 到模型要求的 image_size (1024)
    target_size = predictor.image_size  # 1024
    frames_resized = F.interpolate(
        frames_tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )  # (T, 3, image_size, image_size)

    # ============================================================
    # 3. 初始化推理状态
    # ============================================================
    # val_init_state 接收已经 resize 好的 tensor
    # video_height/video_width 是原始分辨率，用于最终 resize 输出
    inference_state = predictor.val_init_state(
        frames_resized,
        video_height=H,
        video_width=W,
    )

    # ============================================================
    # 4. 添加初始 mask（frame 0 的 ground truth）
    # ============================================================
    # add_new_mask 要求 mask 是 2D, shape (H, W), bool 或可转为 bool
    mask_bool = (mask_hw > 0)  # numpy (H, W) bool
    predictor.add_new_mask(
        inference_state,
        frame_idx=0,
        obj_id=1,
        mask=mask_bool,
    )

    # ============================================================
    # 5. 前向传播（forward only, 满足因果性要求）
    # ============================================================
    video_segments = {}  # frame_idx -> (H, W) mask

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        # out_mask_logits shape: (num_obj, 1, H, W)
        # 找 obj_id=1 的索引
        if 1 in out_obj_ids:
            idx = out_obj_ids.index(1)
            # out_mask_logits[idx] shape: (1, H, W)
            mask_pred = (out_mask_logits[idx] > 0.0).squeeze(0)  # (H, W)
            video_segments[out_frame_idx] = mask_pred.cpu().numpy().astype(np.uint8)

    # ============================================================
    # 6. 组装输出 (T, H, W) 然后转回 (W, H, T)
    # ============================================================
    output_thw = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        if t in video_segments:
            output_thw[t] = video_segments[t]
        elif t == 0:
            # frame 0 用原始 target 作为 fallback
            output_thw[t] = mask_hw.astype(np.uint8)

    # 转回官方要求的 (W, H, T) 格式
    # (T, H, W) -> transpose -> (W, H, T)
    output = np.transpose(output_thw, (2, 1, 0))  # (W, H, T)

    return output