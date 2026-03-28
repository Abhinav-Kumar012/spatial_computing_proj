"""
Diagnostic script to inspect the .h5 datasets and trace
the exact tensor shapes through HWViT's forward pass.
Run from the project/ directory:
    python3 inspect_data.py
"""

import h5py
import numpy as np
import torch

train_path = "Dataset/WV3/train_wv3-001.h5"
test_path  = "Dataset/WV3/test_wv3_multiExm1.h5"

# ── 1. Raw HDF5 inspection ─────────────────────────────────────────────────────
for label, path in [("TRAIN", train_path), ("TEST", test_path)]:
    print(f"\n{'='*60}")
    print(f"  {label} file: {path}")
    print(f"{'='*60}")
    with h5py.File(path, 'r') as f:
        print(f"  Keys: {list(f.keys())}")
        for key in f.keys():
            arr = np.array(f[key])
            print(f"\n  [{key}]")
            print(f"    shape  : {arr.shape}")
            print(f"    dtype  : {arr.dtype}")
            print(f"    min    : {arr.min():.4f}")
            print(f"    max    : {arr.max():.4f}")
            print(f"    mean   : {arr.mean():.4f}")

# ── 2. Tensor shapes after load_dataset ───────────────────────────────────────
ratio = 2047

print(f"\n{'='*60}")
print("  Tensor shapes after load_dataset (train)")
print(f"{'='*60}")
with h5py.File(train_path, 'r') as f:
    gt  = torch.tensor(np.array(f['gt'])  / ratio).float()
    pan = torch.tensor(np.array(f['pan']) / ratio).float()
    ms  = torch.tensor(np.array(f['ms'])  / ratio).float()
    lms = torch.tensor(np.array(f['lms']) / ratio).float()

print(f"  pan : {tuple(pan.shape)}   (B, C_pan, H, W)")
print(f"  gt  : {tuple(gt.shape)}   (B, C_ms,  H, W)")
print(f"  ms  : {tuple(ms.shape)}   (B, C_ms,  H/4, W/4) ?")
print(f"  lms : {tuple(lms.shape)}  (B, C_ms,  H/4, W/4) ?")

B, C_pan, H_pan, W_pan = pan.shape
B, C_ms,  H_ms,  W_ms  = ms.shape
B, C_ms2, H_lms, W_lms = lms.shape

print(f"\n  --- Derived ---")
print(f"  PAN channels (C_pan)   : {C_pan}")
print(f"  MS  channels (C_ms)    : {C_ms}")
print(f"  PAN spatial  (H x W)   : {H_pan} x {W_pan}")
print(f"  MS  spatial  (H x W)   : {H_ms}  x {W_ms}")
print(f"  LMS spatial  (H x W)   : {H_lms} x {W_lms}")
print(f"  Scale factor pan/ms H  : {H_pan / H_ms:.2f}x")
print(f"  Scale factor pan/ms W  : {W_pan / W_ms:.2f}x")

# ── 3. Trace intermediate shapes in HWViT.forward ─────────────────────────────
print(f"\n{'='*60}")
print("  Intermediate shape trace in HWViT.forward")
print(f"  (using example pan_target_channel=40, ms_target_channel=61)")
print(f"{'='*60}")

pan_target_channel = 40
ms_target_channel  = 61

# lms_1 = PixelShuffle(4) applied to ms (C_ms → C_ms*16 → C_ms after shuffle)
# PixelShuffle(4): C_ms*16 channels, spatial /4 → C_ms channels, spatial *4
lms_1_C = C_ms
lms_1_H = H_ms * 4
lms_1_W = W_ms * 4
print(f"\n  lms_1  (after pixelshuffle): ({B}, {lms_1_C}, {lms_1_H}, {lms_1_W})")

# lms_2 = lms_raise_channel(lms_1): C_ms → ms_target_channel
lms_2_C = ms_target_channel
lms_2_H = lms_1_H
lms_2_W = lms_1_W
print(f"  lms_2  (after raise_channel): ({B}, {lms_2_C}, {lms_2_H}, {lms_2_W})")

# pan after pan_raise_channel: C_pan → pan_target_channel
pan_raised_C = pan_target_channel
pan_raised_H = H_pan
pan_raised_W = W_pan
print(f"  pan    (after raise_channel): ({B}, {pan_raised_C}, {pan_raised_H}, {pan_raised_W})")

# L_MWiT inputs:
# pan_ll = pan_down_2(pan) → stride-2 pool
pan_ll_C = pan_target_channel
pan_ll_H = pan_raised_H // 2
pan_ll_W = pan_raised_W // 2
print(f"\n  [L_MWiT inputs]")
print(f"  pan_ll (pan_down_2):      ({B}, {pan_ll_C}, {pan_ll_H}, {pan_ll_W})")

L_up_C = ms_target_channel
L_up_H = lms_2_H // 4
L_up_W = lms_2_W // 4
print(f"  L_up   (lms_down_4):      ({B}, {L_up_C}, {L_up_H}, {L_up_W})")

back_img_C = ms_target_channel
back_img_H = H_ms
back_img_W = W_ms
print(f"  back_img (ms_raise_ch):   ({B}, {back_img_C}, {back_img_H}, {back_img_W})")

# After DWT on pan_ll: 4 * pan_ll_C channels, H/2, W/2
wd_ll_C = pan_ll_C   # each subband = pan_ll_C channels
wd_H    = pan_ll_H // 2
wd_W    = pan_ll_W // 2
print(f"\n  wd_ll  (DWT subband):     ({B}, {wd_ll_C}, {wd_H}, {wd_W})")
print(f"  L_up   going into combine:({B}, {L_up_C},  {L_up_H}, {L_up_W})")
print(f"\n  >>> combine x1 channels = {wd_ll_C}  (pan_target_channel)")
print(f"  >>> combine x2 channels = {L_up_C}  (ms_target_channel)")
print(f"  >>> MISMATCH = {wd_ll_C != L_up_C}  ← this is the bug")

print(f"\n{'='*60}")
print("  Summary")
print(f"{'='*60}")
print(f"  pan channels (C_pan)         : {C_pan}")
print(f"  ms / lms channels (C_ms)     : {C_ms}")
print(f"  Are pan and ms same channels?: {C_pan == C_ms}")
