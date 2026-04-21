from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import numpy as np
import h5py
import cv2
import os
import uuid
import io
import tempfile

from metrics import compute_metrics

app = FastAPI()


# -----------------------------
# Utility: Load uploaded file
# -----------------------------
def load_from_upload(file: UploadFile):
    try:
        file.file.seek(0)
        content = file.file.read()
        filename = file.filename.lower()

        # ---- Handle .npy ----
        if filename.endswith('.npy'):
            return np.load(io.BytesIO(content))

        # ---- Handle .h5 ----
        elif filename.endswith('.h5'):
            # Use temp file (more stable for large files)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            with h5py.File(tmp_path, 'r') as f:
                for k in ['pan', 'ms', 'lms', 'gt', 'data', 'img', 'image']:
                    if k in f:
                        return np.array(f[k])
                return np.array(list(f.values())[0])

        # ---- Handle image formats ----
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
        # Handle (C, H, W) → (H, W, C)
        if img.shape[0] <= 8 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))

    return img


# -----------------------------
# Main API
# -----------------------------
@app.post("/fuse/brovey")
async def fuse_brovey(
    pan_file: UploadFile = File(...),
    ms_file: UploadFile = File(...),
    gt_file: UploadFile = File(...),
    divisor: float = Form(...)
):
    try:
        # ---- Load inputs ----
        pan_img = ensure_hwc(load_from_upload(pan_file))
        ms_img  = ensure_hwc(load_from_upload(ms_file))
        gt_img  = ensure_hwc(load_from_upload(gt_file))

        if pan_img is None or ms_img is None or gt_img is None:
            raise HTTPException(status_code=400, detail="Failed to load one or more images")

        # ---- Basic validation ----
        if pan_img.ndim != 3 or ms_img.ndim != 3 or gt_img.ndim != 3:
            raise HTTPException(status_code=400, detail="All inputs must be 3D arrays (H, W, C)")

        # ---- Scale to [0, 1] ----
        pan_sc = (pan_img / divisor).astype(np.float32)
        ms_sc  = (ms_img / divisor).astype(np.float32)
        gt_sc  = (gt_img / divisor).astype(np.float32)

        # ---- Resize MS to PAN resolution ----
        H_pan, W_pan, _ = pan_sc.shape
        lms_sc = cv2.resize(ms_sc, (W_pan, H_pan), interpolation=cv2.INTER_CUBIC)

        if lms_sc.ndim == 2:
            lms_sc = np.expand_dims(lms_sc, axis=2)

        # ---- Brovey Transform ----
        intensity = np.sum(lms_sc, axis=2, keepdims=True)
        pred_sc = (lms_sc / (intensity + 1e-8)) * pan_sc

        # ---- Prepare for metrics (1, C, H, W) ----
        pred_t = np.expand_dims(pred_sc.transpose(2, 0, 1), 0)
        gt_t   = np.expand_dims(gt_sc.transpose(2, 0, 1), 0)

        metrics = compute_metrics(pred_t, gt_t)

        # ---- Format metrics ----
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

        pred_vis_path = to_vis(pred_t)
        gt_vis_path   = to_vis(gt_t)

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