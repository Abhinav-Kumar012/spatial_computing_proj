import h5py
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
import dill as pickle
import pandas as pd
import matplotlib.pyplot as plt

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
    os.makedirs("results_txt", exist_ok=True)
    with open(f"results_txt/results_{name}.txt", "a") as f:
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
# GLOBAL STORAGE FOR METRICS
# ==========================================
all_results = []


# ==========================================
# METRIC FIXING
# ==========================================
def restore_metric(metric_name, value):
    if metric_name in ["SAM", "ERGAS", "SD"]:
        return value * 10.0
    elif metric_name == "CC":
        return -value
    elif metric_name == "SF":
        return -value * 10.0
    elif metric_name == "SSIM":
        return -value
    return value


def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def metric_direction(metric):
    if metric in ["CC", "SF", "SSIM"]:
        return "↑ better"
    else:
        return "↓ better"

# ==========================================
# PIPELINES
# ==========================================
def run_god(pan, ms, gt, index, model_name, pareto_data, model_idx):
    print(f"[GOD] Index {index} | Model: {model_name}")

    pan_target, ms_target, head_ch = get_model_config(pareto_data, model_name)
    model_path = os.path.join(MODEL_DIR, model_name)

    base_dir = setup_dirs("god")
    model_dir = os.path.join(base_dir, str(model_idx))

    pan_dir = os.path.join(model_dir, "pan")
    ms_dir = os.path.join(model_dir, "ms")
    gt_dir = os.path.join(model_dir, "gt")
    pred_dir = os.path.join(model_dir, "pred")

    for d in [pan_dir, ms_dir, gt_dir, pred_dir]:
        os.makedirs(d, exist_ok=True)

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

    metrics_row = {"pipeline": "GOD", "model": model_idx, "index": index}
    metrics_row.update(metrics)
    all_results.append(metrics_row)

    save_image(pan / DIVISOR, os.path.join(pan_dir, f"{index}.png"))
    save_image(ms / DIVISOR, os.path.join(ms_dir, f"{index}.png"))
    save_image(gt / DIVISOR, os.path.join(gt_dir, f"{index}.png"))
    save_image(pred_np[0].transpose(1, 2, 0),
               os.path.join(pred_dir, f"{index}.png"))


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

    metrics_row = {"pipeline": "BROVEY", "model": "brovey", "index": index}
    metrics_row.update(metrics)
    all_results.append(metrics_row)


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

    metrics_row = {"pipeline": "IHS", "model": "ihs", "index": index}
    metrics_row.update(metrics)
    all_results.append(metrics_row)


# ==========================================
# PLOTTING
# ==========================================
def plot_metrics():
    df = pd.DataFrame(all_results)
    metrics_list = ["SAM", "ERGAS", "CC", "SD", "SF", "SSIM"]

    indices = df["index"].unique()

    for idx in indices:
        dir_brovey = os.path.join("plots_god_vs_brovey", str(idx))
        dir_ihs = os.path.join("plots_god_vs_ihs", str(idx))
        os.makedirs(dir_brovey, exist_ok=True)
        os.makedirs(dir_ihs, exist_ok=True)

        idx_df = df[df["index"] == idx]
        god_df = idx_df[idx_df["pipeline"] == "GOD"]
        brovey_df = idx_df[idx_df["pipeline"] == "BROVEY"]
        ihs_df = idx_df[idx_df["pipeline"] == "IHS"]

        for metric in metrics_list:
            # restore values
            god_vals = [restore_metric(metric, v) for v in god_df[metric].values]
            
            brovey_val = restore_metric(metric, brovey_df[metric].iloc[0]) if not brovey_df.empty else 0
            ihs_val = restore_metric(metric, ihs_df[metric].iloc[0]) if not ihs_df.empty else 0

            # normalize
            combined = god_vals + [brovey_val]
            norm = normalize(combined)
            god_norm = norm[:-1]
            brovey_norm = norm[-1]

            combined2 = god_vals + [ihs_val]
            norm2 = normalize(combined2)
            god_norm2 = norm2[:-1]
            ihs_norm = norm2[-1]

            direction = metric_direction(metric)

            # GOD vs Brovey
            plt.figure()
            plt.bar(range(len(god_norm)), god_norm)
            plt.plot([brovey_norm]*len(god_norm), color='red')

            plt.title(f"{metric} ({direction}) (GOD vs Brovey) - Index {idx}")
            plt.xlabel("GOD Models")
            plt.ylabel("Normalized Value")
            plt.figtext(0.5, -0.08, "Line = Brovey", ha="center")

            plt.savefig(os.path.join(dir_brovey, f"{metric}.png"), bbox_inches='tight')
            plt.close()

            # GOD vs IHS
            plt.figure()
            plt.bar(range(len(god_norm2)), god_norm2)
            plt.plot([ihs_norm]*len(god_norm2), color='green')

            plt.title(f"{metric} ({direction}) (GOD vs IHS) - Index {idx}")
            plt.xlabel("GOD Models")
            plt.ylabel("Normalized Value")
            plt.figtext(0.5, -0.08, "Line = IHS", ha="center")

            plt.savefig(os.path.join(dir_ihs, f"{metric}.png"), bbox_inches='tight')
            plt.close()


# ==========================================
# MAIN (UNCHANGED)
# ==========================================
def main():
    pareto_data = load_pareto()
    model_names = pareto_data["model_names"]

    for name in ["god", "brovey", "ihs"]:
        txt_path = f"results_txt/results_{name}.txt"
        if os.path.exists(txt_path):
            os.remove(txt_path)

    with h5py.File(DATASET_PATH, 'r') as f:
        for idx in TEST_INDICES:
            print(f"\n=== Index {idx} ===")

            pan = ensure_hwc(np.array(f['pan'])[idx])
            ms  = ensure_hwc(np.array(f['ms'])[idx])
            gt  = ensure_hwc(np.array(f['gt'])[idx])

            for i, model_name in enumerate(model_names):
                run_god(pan, ms, gt, idx, model_name, pareto_data, i)

            run_brovey(pan, ms, gt, idx)
            run_ihs(pan, ms, gt, idx)

    plot_metrics()


if __name__ == "__main__":
    main()