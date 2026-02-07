import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
import sys

# -----------------------------------------------------------------------------
# 1. 路径与环境配置
# -----------------------------------------------------------------------------
# 假设 sam2_train 文件夹与 model.py 同级
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from sam2_train.build_sam import build_sam2_video_predictor

# [CRITICAL] 根据你的截图，Grand Challenge 会将模型解压到 /opt/ml/model/
# 请确保你的 tar.gz 包里直接包含 .pth 文件，不要套文件夹
# 如果你的文件名不是 best_dice.pth，请在这里修改
CHECKPOINT_PATH = Path("/opt/ml/model/best_dice.pth") 
CONFIG_NAME = "sam2_hiera_s.yaml"

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    """
    Args:
    - frames: (T, H, W) -> SimpleITK loaded array (No transpose needed!)
    - target: (1, H, W) or (H, W) -> First frame Ground Truth
    """
    
    # -------------------------------------------------------
    # 2. 维度确认 (无需转置)
    # -------------------------------------------------------
    # inference.py 传进来的就是 (T, H, W)
    T, H, W = frames.shape
    
    # Target 处理: 确保压缩成 (H, W)
    # 比如从 (1, 512, 512) 变成 (512, 512)
    target_hw = target.squeeze() 

    # -------------------------------------------------------
    # 3. 初始化模型
    # -------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用绝对路径初始化 Hydra，避免 Docker 路径迷路
    config_dir = current_dir / "sam2_train"
    
    if not GlobalHydra.instance().is_initialized():
        # config_path 必须转为 str
        initialize(version_base='1.2', config_path=str(config_dir.name), job_name="trackrad_inference")

    # 检查权重是否存在，打印日志方便调试
    if not CHECKPOINT_PATH.exists():
        print(f"⚠️ Warning: Checkpoint not found at {CHECKPOINT_PATH}!")
        print(f"Listing /opt/ml/model/: {list(Path('/opt/ml/model/').glob('*'))}")

    predictor = build_sam2_video_predictor(CONFIG_NAME, str(CHECKPOINT_PATH), device=device)

    # -------------------------------------------------------
    # 4. 预处理: 逐帧归一化 (关键步骤)
    # -------------------------------------------------------
    # 必须转为 float32
    frames_float = frames.astype(np.float32)
    
    # [CRITICAL] 逐帧计算 Min/Max
    # axis=(1, 2) 表示在 H, W 维度上计算，保留 T 维度 -> (T, 1, 1)
    # 严禁使用全局 frames.min()，否则违反实时性规则
    f_min = frames_float.min(axis=(1, 2), keepdims=True)
    f_max = frames_float.max(axis=(1, 2), keepdims=True)
    
    # 归一化到 0-255
    den = f_max - f_min
    den[den == 0] = 1.0  # 防止除以0
    frames_norm = (frames_float - f_min) / den * 255.0
    
    # 转 Tensor: (T, H, W) -> (T, 3, H, W)
    frames_tensor = torch.from_numpy(frames_norm).float()
    frames_tensor = frames_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    
    # Resize (SAM2 需要特定尺寸输入)
    target_size = predictor.image_size
    frames_resized = F.interpolate(
        frames_tensor, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    )

    # -------------------------------------------------------
    # 5. 推理 (SAM 2)
    # -------------------------------------------------------
    # 初始化 Inference State
    inference_state = predictor.val_init_state(
        frames_resized, 
        video_height=H, 
        video_width=W
    )
    
    # 将第一帧的 Ground Truth 加入
    predictor.add_new_mask(
        inference_state, 
        frame_idx=0, 
        obj_id=1, 
        mask=torch.from_numpy(target_hw).bool()
    )
    
    # 准备输出容器: (T, H, W)
    output = np.zeros((T, H, W), dtype=np.uint8)
    output[0] = target_hw.astype(np.uint8)

    # 视频传播
    # SAM 2 会自动处理时序依赖
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if 1 in out_obj_ids:
            idx = out_obj_ids.index(1)
            # 结果转为 uint8 mask
            mask_pred = (out_mask_logits[idx] > 0.0).cpu().numpy().astype(np.uint8)
            output[out_frame_idx] = mask_pred

    # -------------------------------------------------------
    # 6. 返回结果
    # -------------------------------------------------------
    # 直接返回 (T, H, W)
    # inference.py 会使用 SimpleITK 将其保存为正确的物理序列
    return output