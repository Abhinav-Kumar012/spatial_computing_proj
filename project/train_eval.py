import torch
import numpy as np
import h5py
from fusion_model import HWViT
from metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(path, ratio):

    with h5py.File(path,'r') as f:
        gt = np.array(f['gt'])
        pan = np.array(f['pan'])
        ms = np.array(f['ms'])
        lms = np.array(f['lms'])

    return (
        torch.tensor(pan/ratio).float(),
        torch.tensor(gt/ratio).float(),
        torch.tensor(ms/ratio).float(),
        torch.tensor(lms/ratio).float()
    )


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

    epochs = params["epochs"]

    pan, gt, ms, lms = pan.to(device), gt.to(device), ms.to(device), lms.to(device)

    model.train()

    for _ in range(epochs):

        optimizer.zero_grad()

        out = model(pan, ms, lms)

        loss = criterion(out, gt)

        loss.backward()

        optimizer.step()

    # evaluation
    pan_t, gt_t, ms_t, lms_t = load_dataset(test_path, ratio)

    pan_t, gt_t, ms_t, lms_t = pan_t.to(device), gt_t.to(device), ms_t.to(device), lms_t.to(device)

    model.eval()

    with torch.no_grad():

        pred = model(pan_t, ms_t, lms_t).cpu().numpy()
        gt_np = gt_t.cpu().numpy()

    metrics = compute_metrics(pred, gt_np)

    return metrics