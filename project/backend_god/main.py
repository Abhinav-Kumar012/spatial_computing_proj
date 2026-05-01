from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import numpy as np
import h5py
import cv2
import os
import uuid
import torch.nn.functional as F
import io
import tempfile

from fusion_model import HWViT
from metrics import compute_metrics

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# Utility: Load uploaded file
# -----------------------------
def load_from_upload(file: UploadFile):
    try:
        file.file.seek(0)
        content = file.file.read()
        filename = file.filename.lower()

        # ---- .npy ----
        if filename.endswith('.npy'):
            return np.load(io.BytesIO(content))

        # ---- .h5 ----
        elif filename.endswith('.h5'):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            with h5py.File(tmp_path, 'r') as f:
                for k in ['pan', 'ms', 'lms', 'gt', 'data', 'img', 'image']:
                    if k in f:
                        return np.array(f[k])
                return np.array(list(f.values())[0])

        # ---- images ----
        else:
            file_bytes = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if img is None:
                return None

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")


# -----------------------------
# Utility: Ensure HWC format
# -----------------------------
def ensure_hwc(img):
    if img is None:
        return None

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    elif img.ndim == 3:
        # Handle (C, H, W)
        if img.shape[0] <= 8 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))

    return img


# -----------------------------
# Utility: Load model from upload
# -----------------------------
def load_model_from_upload(model_file: UploadFile):
    try:
        model_file.file.seek(0)
        content = model_file.file.read()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        state_dict = torch.load(tmp_path, map_location=device)
        return state_dict

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")


# -----------------------------
# Main API
# -----------------------------
@app.post("/fuse/god")
async def fuse_god(
    pan_file: UploadFile = File(...),
    ms_file: UploadFile = File(...),
    gt_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    pan_target_ch: int = Form(...),
    ms_target_ch: int = Form(...),
    head_ch: int = Form(...),
    divisor: float = Form(...)
):
    try:
        # ---- Load inputs ----
        pan_img = ensure_hwc(load_from_upload(pan_file))
        ms_img  = ensure_hwc(load_from_upload(ms_file))
        gt_img  = ensure_hwc(load_from_upload(gt_file))

        if pan_img is None or ms_img is None or gt_img is None:
            raise HTTPException(status_code=400, detail="Failed to load input images")

        # ---- Convert to CHW ----
        pan_img = pan_img.transpose(2, 0, 1)
        ms_img  = ms_img.transpose(2, 0, 1)
        gt_img  = gt_img.transpose(2, 0, 1)

        # ---- Normalize ----
        pan_t = torch.tensor(pan_img / divisor).float().unsqueeze(0)
        ms_t  = torch.tensor(ms_img / divisor).float().unsqueeze(0)
        gt_t  = torch.tensor(gt_img / divisor).float().unsqueeze(0)

        # ---- Resize MS ----
        _, _, H_pan, W_pan = pan_t.shape
        lms_t = F.interpolate(ms_t, size=(H_pan, W_pan), mode='bicubic', align_corners=False)

        # ---- Build model ----
        model = HWViT(
            L_up_channel=ms_t.shape[1],
            pan_channel=pan_t.shape[1],
            ms_target_channel=ms_target_ch,
            pan_target_channel=pan_target_ch,
            head_channel=head_ch,
            dropout=0.0
        )

        # ---- Load weights ----
        state_dict = load_model_from_upload(model_file)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # ---- Inference ----
        with torch.no_grad():
            pred_t = model(pan_t.to(device), ms_t.to(device), lms_t.to(device))

        pred_np = pred_t.cpu().numpy()
        gt_np   = gt_t.numpy()

        # ---- Metrics ----
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

        # -----------------------------
        # Visualization
        # -----------------------------
        def to_vis(img_array):
            img = img_array[0].transpose(1, 2, 0)

            if img.shape[2] >= 3:
                img = img[:, :, :3]

            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            out_name = f"/tmp/gradio/{uuid.uuid4().hex}.png"
            os.makedirs(os.path.dirname(out_name), exist_ok=True)

            cv2.imwrite(out_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return out_name

        pred_vis_path = to_vis(pred_np)
        gt_vis_path   = to_vis(gt_np)

        return {
            "metrics_md": metrics_md,
            "pred_vis_path": pred_vis_path,
            "gt_vis_path": gt_vis_path,
            "error": None
        }

    except Exception as e:
        import traceback
        return {
            "error": f"{str(e)}\n{traceback.format_exc()}"
        }