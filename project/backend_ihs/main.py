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

        if filename.endswith('.npy'):
            return np.load(io.BytesIO(content))

        elif filename.endswith('.h5'):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            with h5py.File(tmp_path, 'r') as f:
                for k in ['pan', 'ms', 'lms', 'gt', 'data', 'img', 'image']:
                    if k in f:
                        return np.array(f[k])
                return np.array(list(f.values())[0])

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
# Ensure HWC
# -----------------------------
def ensure_hwc(img):
    if img is None:
        return None

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    elif img.ndim == 3:
        if img.shape[0] <= 8 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))

    return img


# -----------------------------
# RGB → IHS
# -----------------------------
def rgb_to_ihs(rgb):
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    I = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-8))

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-8
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)

    return np.stack([I, H, S], axis=-1)


# -----------------------------
# IHS → RGB
# -----------------------------
def ihs_to_rgb(ihs):
    I, H, S = ihs[:, :, 0], ihs[:, :, 1], ihs[:, :, 2]
    H = H * 2 * np.pi

    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            h = H[i, j]
            s = S[i, j]
            ii = I[i, j]

            if h < 2 * np.pi / 3:
                B[i, j] = ii * (1 - s)
                R[i, j] = ii * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h))
                G[i, j] = 3 * ii - (R[i, j] + B[i, j])
            elif h < 4 * np.pi / 3:
                h -= 2 * np.pi / 3
                R[i, j] = ii * (1 - s)
                G[i, j] = ii * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h))
                B[i, j] = 3 * ii - (R[i, j] + G[i, j])
            else:
                h -= 4 * np.pi / 3
                G[i, j] = ii * (1 - s)
                B[i, j] = ii * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h))
                R[i, j] = 3 * ii - (G[i, j] + B[i, j])

    return np.stack([R, G, B], axis=-1)


# -----------------------------
# Main API
# -----------------------------
@app.post("/fuse/ihs")
async def fuse_ihs(
    pan_file: UploadFile = File(...),
    ms_file: UploadFile = File(...),
    gt_file: UploadFile = File(...),
    divisor: float = Form(...)
):
    try:
        pan_img = ensure_hwc(load_from_upload(pan_file))
        ms_img  = ensure_hwc(load_from_upload(ms_file))
        gt_img  = ensure_hwc(load_from_upload(gt_file))

        if pan_img is None or ms_img is None or gt_img is None:
            raise HTTPException(status_code=400, detail="Failed to load inputs")

        pan_sc = (pan_img / divisor).astype(np.float32)
        ms_sc  = (ms_img / divisor).astype(np.float32)
        gt_sc  = (gt_img / divisor).astype(np.float32)

        H_pan, W_pan, _ = pan_sc.shape
        lms_sc = cv2.resize(ms_sc, (W_pan, H_pan), interpolation=cv2.INTER_CUBIC)

        # Use first 3 bands for RGB
        rgb = lms_sc[:, :, :3]

        ihs = rgb_to_ihs(rgb)

        # Replace intensity with PAN
        ihs[:, :, 0] = pan_sc[:, :, 0]

        pred_rgb = ihs_to_rgb(ihs)

        pred_rgb = np.clip(pred_rgb, 0, 1)

        pred_t = np.expand_dims(pred_rgb.transpose(2, 0, 1), 0)
        gt_t   = np.expand_dims(gt_sc.transpose(2, 0, 1), 0)

        metrics = compute_metrics(pred_t, gt_t)

        metrics_md = (
            f"### RGB-IHS Transform Objectives Evaluated\n\n"
            f"| Metric | Value |\n"
            f"|:------:|:-----:|\n"
            f"| **SAM** | {metrics['SAM']:.4f} |\n"
            f"| **ERGAS** | {metrics['ERGAS']:.4f} |\n"
            f"| **CC** | {metrics['CC']:.4f} |\n"
            f"| **SD** | {metrics['SD']:.4f} |\n"
            f"| **SF** | {metrics['SF']:.4f} |\n"
            f"| **SSIM** | {metrics['SSIM']:.4f} |\n"
            f"| **n_params** | 0 (IHS Transform) |\n"
        )

        def to_vis(img_array):
            img = img_array[0].transpose(1, 2, 0)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            out_name = f"/tmp/gradio/{uuid.uuid4().hex}.png"
            os.makedirs(os.path.dirname(out_name), exist_ok=True)

            cv2.imwrite(out_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return out_name

        return {
            "metrics_md": metrics_md,
            "pred_vis_path": to_vis(pred_t),
            "gt_vis_path": to_vis(gt_t),
            "error": None
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}