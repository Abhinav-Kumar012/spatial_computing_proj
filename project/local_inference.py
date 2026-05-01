import h5py
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
import dill as pickle

from metrics import compute_metrics
from fusion_model import HWViT


# ==========================================
# CONFIGURATION
# ==========================================
PARETO_PATH = "../pansharpning_pareto_front/pareto_front.pkl"
MODEL_DIR = "../pansharpning_pareto_front/"
DATASET_PATH = "../datav1/valid_wv3.h5"

TEST_INDICES = [0, 1]

DIVISOR = 2047.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# LOAD PARETO FRONT
# ==========================================
def load_pareto():
    with open(PARETO_PATH, "rb") as f:
        return pickle.load(f)


def get_model_config(pareto_data, model_name):
    mapping = pareto_data["mapping"]

    entry = None
    for m in mapping:
        if m["model_name"] == model_name:
            entry = m
            break

    if entry is None:
        raise ValueError(f"Model {model_name} not found in pareto mapping")

    pan_raw, ms_raw, head_ch, _, _ = entry["decision"]

    head_ch = int(head_ch)
    pan_raw = int(pan_raw)
    ms_raw = int(ms_raw)

    pan_target = max(head_ch, (pan_raw // head_ch) * head_ch)
    ms_target = max(head_ch, (ms_raw // head_ch) * head_ch)

    return pan_target, ms_target, head_ch


# ==========================================
# UTILITIES
# ==========================================
def ensure_hwc(img):
    if img.ndim == 2:
        return np.expand_dims(img, axis=-1)
    if img.shape[0] < img.shape[1] and img.shape[0] <= 8:
        return np.transpose(img, (1, 2, 0))
    return img


def save_image(img_arr, path):
    img = img_arr.copy()
    img = img.astype(np.float32)

    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img)


def setup_dirs(name):
    d = os.path.join(".", f"output_{name}")
    os.makedirs(d, exist_ok=True)
    return d


def write_metrics(name, index, metrics):
    with open(f"results_{name}.txt", "a") as f:
        f.write(f"--- Index {index} ---\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write("\n")


# ==========================================
# IHS TRANSFORMS
# ==========================================
def rgb_to_ihs(rgb):
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    I = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-8))

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-8
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2*np.pi)

    return np.stack([I, H, S], axis=-1)


def ihs_to_rgb(ihs):
    I, H, S = ihs[:, :, 0], ihs[:, :, 1], ihs[:, :, 2]
    H = H * 2*np.pi

    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            h, s, ii = H[i, j], S[i, j], I[i, j]

            if h < 2*np.pi/3:
                B[i, j] = ii * (1 - s)
                R[i, j] = ii * (1 + s*np.cos(h)/np.cos(np.pi/3 - h))
                G[i, j] = 3*ii - (R[i, j] + B[i, j])
            elif h < 4*np.pi/3:
                h -= 2*np.pi/3
                R[i, j] = ii * (1 - s)
                G[i, j] = ii * (1 + s*np.cos(h)/np.cos(np.pi/3 - h))
                B[i, j] = 3*ii - (R[i, j] + G[i, j])
            else:
                h -= 4*np.pi/3
                G[i, j] = ii * (1 - s)
                B[i, j] = ii * (1 + s*np.cos(h)/np.cos(np.pi/3 - h))
                R[i, j] = 3*ii - (G[i, j] + B[i, j])

    return np.stack([R, G, B], axis=-1)


# ==========================================
# PIPELINES
# ==========================================
def run_god(pan, ms, gt, index, model_name, pareto_data):
    print(f"[GOD] Index {index} | Model: {model_name}")

    pan_target, ms_target, head_ch = get_model_config(pareto_data, model_name)
    model_path = os.path.join(MODEL_DIR, model_name)

    dir_out = setup_dirs("god")

    pan_t = torch.tensor(pan / DIVISOR).float().unsqueeze(0).permute(0, 3, 1, 2)
    ms_t  = torch.tensor(ms / DIVISOR).float().unsqueeze(0).permute(0, 3, 1, 2)
    gt_t  = torch.tensor(gt / DIVISOR).float().unsqueeze(0).permute(0, 3, 1, 2)

    _, _, H, W = pan_t.shape
    lms_t = F.interpolate(ms_t, size=(H, W), mode='bicubic', align_corners=False)

    model = HWViT(
        L_up_channel=ms_t.shape[1],
        pan_channel=pan_t.shape[1],
        ms_target_channel=ms_target,
        pan_target_channel=pan_target,
        head_channel=head_ch,
        dropout=0.0
    )

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(pan_t.to(device), ms_t.to(device), lms_t.to(device))

    pred_np = pred.cpu().numpy()
    gt_np = gt_t.numpy()

    metrics = compute_metrics(pred_np, gt_np)
    write_metrics("god", index, metrics)

    # Save inputs + output (with model name to avoid overwrite)
    save_image(pan / DIVISOR, os.path.join(dir_out, f"{index}_{model_name}_pan.png"))
    save_image(ms / DIVISOR, os.path.join(dir_out, f"{index}_{model_name}_ms.png"))
    save_image(gt / DIVISOR, os.path.join(dir_out, f"{index}_{model_name}_gt.png"))
    save_image(pred_np[0].transpose(1, 2, 0),
               os.path.join(dir_out, f"{index}_{model_name}_pred.png"))


def run_brovey(pan, ms, gt, index):
    print(f"[Brovey] Index {index}")

    dir_out = setup_dirs("brovey")

    pan_sc = (pan / DIVISOR).astype(np.float32)
    ms_sc = (ms / DIVISOR).astype(np.float32)
    gt_sc = (gt / DIVISOR).astype(np.float32)

    H, W, _ = pan_sc.shape
    lms = cv2.resize(ms_sc, (W, H), interpolation=cv2.INTER_CUBIC)

    intensity = np.sum(lms, axis=2, keepdims=True)
    pred = (lms / (intensity + 1e-8)) * pan_sc

    pred_t = np.expand_dims(pred.transpose(2, 0, 1), 0)
    gt_t = np.expand_dims(gt_sc.transpose(2, 0, 1), 0)

    metrics = compute_metrics(pred_t, gt_t)
    write_metrics("brovey", index, metrics)

    # Save inputs + output
    save_image(pan_sc, os.path.join(dir_out, f"{index}_pan.png"))
    save_image(lms, os.path.join(dir_out, f"{index}_ms.png"))
    save_image(gt_sc, os.path.join(dir_out, f"{index}_gt.png"))
    save_image(pred, os.path.join(dir_out, f"{index}_pred.png"))


def run_ihs(pan, ms, gt, index):
    print(f"[IHS] Index {index}")

    dir_out = setup_dirs("ihs")

    pan_sc = (pan / DIVISOR).astype(np.float32)
    ms_sc = (ms / DIVISOR).astype(np.float32)
    gt_sc = (gt / DIVISOR).astype(np.float32)

    H, W, _ = pan_sc.shape
    lms = cv2.resize(ms_sc, (W, H), interpolation=cv2.INTER_CUBIC)

    rgb = lms[:, :, :3]
    ihs = rgb_to_ihs(rgb)

    ihs[:, :, 0] = pan_sc[:, :, 0]

    pred_rgb = ihs_to_rgb(ihs)
    pred_rgb = np.clip(pred_rgb, 0, 1).astype(np.float32)

    pred = lms.copy()
    pred[:, :, :3] = pred_rgb

    pred_t = np.expand_dims(pred.transpose(2, 0, 1), 0)
    gt_t = np.expand_dims(gt_sc.transpose(2, 0, 1), 0)

    metrics = compute_metrics(pred_t, gt_t)
    write_metrics("ihs", index, metrics)

    # Save inputs + output
    save_image(pan_sc, os.path.join(dir_out, f"{index}_pan.png"))
    save_image(lms, os.path.join(dir_out, f"{index}_ms.png"))
    save_image(gt_sc, os.path.join(dir_out, f"{index}_gt.png"))
    save_image(pred_rgb, os.path.join(dir_out, f"{index}_pred.png"))


# ==========================================
# MAIN
# ==========================================
def main():
    pareto_data = load_pareto()
    model_names = pareto_data["model_names"]

    # ✅ CLEAR TXT FILES FIRST
    for name in ["god", "brovey", "ihs"]:
        txt_path = f"results_{name}.txt"
        if os.path.exists(txt_path):
            os.remove(txt_path)

    with h5py.File(DATASET_PATH, 'r') as f:
        for idx in TEST_INDICES:
            print(f"\n=== Index {idx} ===")

            pan = ensure_hwc(np.array(f['pan'])[idx])
            ms  = ensure_hwc(np.array(f['ms'])[idx])
            gt  = ensure_hwc(np.array(f['gt'])[idx])

            for model_name in model_names:
                run_god(pan, ms, gt, idx, model_name, pareto_data)

            run_brovey(pan, ms, gt, idx)
            run_ihs(pan, ms, gt, idx)


if __name__ == "__main__":
    main()