import numpy as np
from skimage.metrics import structural_similarity as ssim

def SAM(pred, gt):

    dot = np.sum(pred*gt, axis=1)
    norm = np.linalg.norm(pred,axis=1)*np.linalg.norm(gt,axis=1)

    return np.mean(np.arccos(dot/(norm+1e-8)))

def ERGAS(pred, gt, ratio=4):

    rmse = np.sqrt(np.mean((pred-gt)**2, axis=(2,3)))
    mean = np.mean(gt, axis=(2,3))

    return 100/ratio*np.sqrt(np.mean((rmse/mean)**2))

def CC(pred, gt):

    return np.corrcoef(pred.flatten(),gt.flatten())[0,1]

def SD_diff(pred,gt):

    return abs(np.std(pred)-np.std(gt))

def spatial_frequency(img):

    RF = np.sqrt(np.mean((img[:,:,1:]-img[:,:,:-1])**2))
    CF = np.sqrt(np.mean((img[:,1:,:]-img[:,:-1,:])**2))

    return np.sqrt(RF**2+CF**2)

def edge_preservation(pred, pan):

    sobel = np.gradient(pan)

    return np.mean(np.abs(sobel))

def SSIM_metric(pred, gt):

    s = []
    for i in range(pred.shape[1]):
        s.append(ssim(gt[0,i],pred[0,i],data_range=1))

    return np.mean(s)

def compute_metrics(pred,gt):

    return {
        "SAM": SAM(pred,gt),
        "ERGAS": ERGAS(pred,gt),
        "CC": CC(pred,gt),
        "SD": SD_diff(pred,gt),
        "SF": spatial_frequency(pred[0]),
        "SSIM": SSIM_metric(pred,gt),
    }