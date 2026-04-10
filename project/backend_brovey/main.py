from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import h5py
import cv2
import os
import uuid

from metrics import compute_metrics

app = FastAPI()

class FusionRequest(BaseModel):
    pan_file: str
    ms_file: str
    gt_file: str
    divisor: float
    # These parameters are ignored by Brovey but kept for API schema consistency
    model_path: str = ""
    pan_target_ch: int = 0
    ms_target_ch: int = 0
    head_ch: int = 0

def load_image_or_array(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            for k in ['pan', 'ms', 'lms', 'gt', 'data', 'img', 'image']:
                if k in f: return np.array(f[k])
            return np.array(list(f.values())[0])
    else:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

@app.post("/fuse/brovey")
def fuse_brovey(req: FusionRequest):
    try:
        pan_img = load_image_or_array(req.pan_file)
        ms_img  = load_image_or_array(req.ms_file)
        gt_img  = load_image_or_array(req.gt_file)
        
        if pan_img is None or ms_img is None or gt_img is None:
            return {"error": "Images could not be loaded from the shared volume. Paths: " + str(req.pan_file) + " " + str(req.ms_file) + " " + str(req.gt_file)}

        if len(pan_img.shape) == 2: pan_img = np.expand_dims(pan_img, 2)
        if len(ms_img.shape) == 2: ms_img = np.expand_dims(ms_img, 2)
        if len(gt_img.shape) == 2: gt_img = np.expand_dims(gt_img, 2)

        # Scale to [0, 1]
        pan_sc = pan_img / req.divisor
        ms_sc  = ms_img / req.divisor
        gt_sc  = gt_img / req.divisor
        
        pan_sc = pan_sc.astype(np.float32)
        ms_sc  = ms_sc.astype(np.float32)
        
        # Dimensions are (H, W, C)
        H_pan, W_pan, _ = pan_sc.shape
        
        # Interpolate MS to match PAN size
        lms_sc = cv2.resize(ms_sc, (W_pan, H_pan), interpolation=cv2.INTER_CUBIC)
        # Ensure it has 3 dims if single channel fallback
        if len(lms_sc.shape) == 2: lms_sc = np.expand_dims(lms_sc, 2)
        
        # Brovey Transform implementation
        # Fused = (MS / sum(MS)) * PAN
        intensity = np.sum(lms_sc, axis=2, keepdims=True)
        pred_sc = (lms_sc / (intensity + 1e-8)) * pan_sc

        # To conform with metrics.py which expects (1, C, H, W)
        pred_t = np.expand_dims(pred_sc.transpose(2, 0, 1), 0)
        gt_t   = np.expand_dims(gt_sc.transpose(2, 0, 1), 0)

        metrics = compute_metrics(pred_t, gt_t)
        
        metrics_md = (
            f"### Brovey Transform Objectives Evaluated\n\n"
            f"| Metric | Value |\n"
            f"|:------:|:-----:|\n"
            f"| **SAM** | {metrics['SAM']:.4f} |\n"
            f"| **ERGAS** | {metrics['ERGAS']:.4f} |\n"
            f"| **CC** | {metrics['CC']:.4f} |\n"
            f"| **SD** | {metrics['SD']:.4f} |\n"
            f"| **SF** | {metrics['SF']:.4f} |\n"
            f"| **SSIM** | {metrics['SSIM']:.4f} |\n"
            f"| **n_params** | 0 (Heuristic Formula) |\n"
        )
        
        def to_vis(img_array):
            # img_array shape is (1, C, H, W)
            img = img_array[0].transpose(1, 2, 0)
            if img.shape[2] >= 3: img = img[:, :, :3]
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            
            # Save to shared volume
            out_name = f"/tmp/gradio/{uuid.uuid4().hex}.png"
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            cv2.imwrite(out_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return out_name
        
        pred_vis_path = to_vis(pred_t)
        gt_vis_path = to_vis(gt_t)

        return {
            "metrics_md": metrics_md,
            "pred_vis_path": pred_vis_path,
            "gt_vis_path": gt_vis_path,
            "error": None
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}
