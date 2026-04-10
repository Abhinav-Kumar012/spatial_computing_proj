from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import h5py
import cv2
import os
import uuid
import torch.nn.functional as F

from fusion_model import HWViT
from metrics import compute_metrics

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FusionRequest(BaseModel):
    pan_file: str
    ms_file: str
    gt_file: str
    model_path: str
    pan_target_ch: int
    ms_target_ch: int
    head_ch: int
    divisor: float

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

@app.post("/fuse/god")
def fuse_god(req: FusionRequest):
    try:
        pan_img = load_image_or_array(req.pan_file)
        ms_img  = load_image_or_array(req.ms_file)
        gt_img  = load_image_or_array(req.gt_file)
        
        if pan_img is None or ms_img is None or gt_img is None:
            return {"error": "Images could not be loaded from the shared volume."}

        # Handle container paths vs host paths 
        # (Frontend passes host path which might be different, but docker-compose mounts it identically if it's in /tmp/gradio)
        model_path_in_container = os.path.join('/app', req.model_path) if not req.model_path.startswith('/app') else req.model_path
        
        if not os.path.exists(model_path_in_container):
            return {"error": f"Model '{model_path_in_container}' not found in backend volume."}

        if len(pan_img.shape) == 2: pan_img = np.expand_dims(pan_img, 2)
        if len(ms_img.shape) == 2: ms_img = np.expand_dims(ms_img, 2)
        if len(gt_img.shape) == 2: gt_img = np.expand_dims(gt_img, 2)

        pan_img = pan_img.transpose(2, 0, 1)
        ms_img = ms_img.transpose(2, 0, 1)
        gt_img = gt_img.transpose(2, 0, 1)

        pan_t = torch.tensor(pan_img / req.divisor).float().unsqueeze(0)
        ms_t = torch.tensor(ms_img / req.divisor).float().unsqueeze(0)
        gt_t = torch.tensor(gt_img / req.divisor).float().unsqueeze(0)

        _, _, H_pan, W_pan = pan_t.shape
        lms_t = F.interpolate(ms_t, size=(H_pan, W_pan), mode='bicubic', align_corners=False)

        model = HWViT(
            L_up_channel=ms_t.shape[1],
            pan_channel=pan_t.shape[1],
            ms_target_channel=req.ms_target_ch,
            pan_target_channel=req.pan_target_ch,
            head_channel=req.head_ch,
            dropout=0.0
        )
        
        state_dict = torch.load(model_path_in_container, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        with torch.no_grad():
            pred_t = model(pan_t.to(device), ms_t.to(device), lms_t.to(device))
        
        pred_np = pred_t.cpu().numpy()
        gt_np = gt_t.numpy()

        metrics = compute_metrics(pred_np, gt_np)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics_md = (
            f"### GOD Objectives Evaluated (GPU: {torch.cuda.is_available()})\n\n"
            f"| Metric | Value |\n"
            f"|:------:|:-----:|\n"
            f"| **SAM** | {metrics['SAM']:.4f} |\n"
            f"| **ERGAS** | {metrics['ERGAS']:.4f} |\n"
            f"| **CC** | {metrics['CC']:.4f} |\n"
            f"| **SD** | {metrics['SD']:.4f} |\n"
            f"| **SF** | {metrics['SF']:.4f} |\n"
            f"| **SSIM** | {metrics['SSIM']:.4f} |\n"
            f"| **n_params** | {n_params:,} |\n"
        )
        
        def to_vis(img_array):
            img = img_array[0].transpose(1, 2, 0)
            if img.shape[2] >= 3: img = img[:, :, :3]
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            
            # Save to shared volume
            out_name = f"/tmp/gradio/{uuid.uuid4().hex}.png"
            # Ensure dir exists safely
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            cv2.imwrite(out_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return out_name
        
        pred_vis_path = to_vis(pred_np)
        gt_vis_path = to_vis(gt_np)

        return {
            "metrics_md": metrics_md,
            "pred_vis_path": pred_vis_path,
            "gt_vis_path": gt_vis_path,
            "error": None
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}
