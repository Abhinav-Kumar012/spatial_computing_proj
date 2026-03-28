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

    pan_t = torch.tensor(pan / ratio).float()
    ms_t  = torch.tensor(ms  / ratio).float()
    lms_t = torch.tensor(lms / ratio).float()
    gt_t  = torch.tensor(gt  / ratio).float() if gt is not None else None

    return pan_t, gt_t, ms_t, lms_t


def train_and_evaluate(params, train_path, test_path):

    ratio = 2047

    pan, gt, ms, lms = load_dataset(train_path, ratio)

    model = HWViT(
        L_up_channel=ms.shape[1],
        pan_channel=pan.shape[1],
        ms_target_channel=params["ms_target_channel"],
        pan_target_channel=params["pan_target_channel"],
        head_channel=params["head_channel"],
        dropout=params["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.L1Loss()
    epochs    = params["epochs"]

    # Mini-batch training — avoids allocating a (1080, heads, 256, 256) attention matrix
    train_ds = torch.utils.data.TensorDataset(pan, gt, ms, lms)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)

    model.train()

    for _ in range(epochs):
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

    # Mini-batch evaluation (test file has no 'gt', accumulate predictions)
    pan_t, gt_t, ms_t, lms_t = load_dataset(test_path, ratio)

    eval_ds = torch.utils.data.TensorDataset(pan_t, ms_t, lms_t)
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=8, shuffle=False)

    model.eval()
    preds = []

    with torch.no_grad():
        for pan_b, ms_b, lms_b in eval_dl:
            pred = model(pan_b.to(device), ms_b.to(device), lms_b.to(device)).cpu().numpy()
            preds.append(pred)

    pred_all = np.concatenate(preds, axis=0)
    gt_all   = gt_t.numpy() if gt_t is not None else pred_all  # fallback if no gt

    metrics = compute_metrics(pred_all, gt_all)

    return metrics