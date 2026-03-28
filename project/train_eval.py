import torch
import numpy as np
import h5py
from fusion_model import HWViT
from metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(path, ratio):

    with h5py.File(path, 'r') as f:
        pan = np.array(f['pan'])
        ms  = np.array(f['ms'])
        lms = np.array(f['lms'])
        gt  = np.array(f['gt']) if 'gt' in f else None

    print(f"  [load_dataset] {path}")
    print(f"    pan: {pan.shape}  ms: {ms.shape}  lms: {lms.shape}  gt: {gt.shape if gt is not None else 'N/A'}")

    pan_t = torch.tensor(pan / ratio).float()
    ms_t  = torch.tensor(ms  / ratio).float()
    lms_t = torch.tensor(lms / ratio).float()
    gt_t  = torch.tensor(gt  / ratio).float() if gt is not None else None

    return pan_t, gt_t, ms_t, lms_t


def train_and_evaluate(params, train_path, test_path):

    ratio = 2047

    print(f"\n[train_and_evaluate] Params: {params}")
    print(f"  Device: {device}")

    print("  Loading training data...")
    pan, gt, ms, lms = load_dataset(train_path, ratio)

    model = HWViT(
        L_up_channel=ms.shape[1],
        pan_channel=pan.shape[1],
        ms_target_channel=params["ms_target_channel"],
        pan_target_channel=params["pan_target_channel"],
        head_channel=params["head_channel"],
        dropout=params["dropout"]
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.L1Loss()
    epochs    = params["epochs"]

    # Mini-batch training — avoids allocating a (1080, heads, 256, 256) attention matrix
    train_ds = torch.utils.data.TensorDataset(pan, gt, ms, lms)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)

    print(f"  Training: {epochs} epochs, {len(train_dl)} batches/epoch")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for pan_b, gt_b, ms_b, lms_b in train_dl:
            pan_b = pan_b.to(device)
            gt_b  = gt_b.to(device)
            ms_b  = ms_b.to(device)
            lms_b = lms_b.to(device)

            optimizer.zero_grad()
            out  = model(pan_b, ms_b, lms_b)
            loss = criterion(out, gt_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dl)
        print(f"  Epoch [{epoch+1}/{epochs}]  avg_loss={avg_loss:.6f}")

    # Mini-batch evaluation (test file has no 'gt', accumulate predictions)
    print("  Loading test data...")
    pan_t, gt_t, ms_t, lms_t = load_dataset(test_path, ratio)

    eval_ds = torch.utils.data.TensorDataset(pan_t, ms_t, lms_t)
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=8, shuffle=False)

    print(f"  Evaluating: {len(eval_dl)} batches")
    model.eval()
    preds = []

    with torch.no_grad():
        for pan_b, ms_b, lms_b in eval_dl:
            pred = model(pan_b.to(device), ms_b.to(device), lms_b.to(device)).cpu().numpy()
            preds.append(pred)

    pred_all = np.concatenate(preds, axis=0)
    gt_all   = gt_t.numpy() if gt_t is not None else pred_all  # fallback if no gt

    metrics = compute_metrics(pred_all, gt_all)

    print(f"  Metrics: SAM={metrics['SAM']:.4f}  ERGAS={metrics['ERGAS']:.4f}  "
          f"CC={metrics['CC']:.4f}  SSIM={metrics['SSIM']:.4f}  "
          f"SF={metrics['SF']:.4f}  SD={metrics['SD']:.4f}")

    return metrics