import h5py
import requests
import numpy as np
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# ⚠️ USER: Update this path to the actual God Model .pt file you want to test
GOD_MODEL_PATH = "REPLACE_WITH_YOUR_MODEL_PATH.pt"  

# Dataset to pull from (You requested valid_wv3.h5 to ensure 'gt' is available)
DATASET_PATH = "../datav1/valid_wv3.h5"

# The index of the array in the .h5 dataset to test
INDEX_TO_EXTRACT = 0

# GOD neural net hyperparameters
PAN_TARGET_CH = 32
MS_TARGET_CH = 32
HEAD_CH = 8

# Image scaling factor
DIVISOR = 2047.0

# Backend container URLs
GOD_URL    = "http://localhost:8000/fuse/god"
BROVEY_URL = "http://localhost:8001/fuse/brovey"
IHS_URL    = "http://localhost:8002/fuse/ihs"

# ------------------------------------------------------------------


def extract_and_save_temp(h5_path, index, out_prefix="tmp"):
    """
    Extracts pan, ms, and gt arrays from a provided .h5 dataset index 
    and saves them as separate, independent .h5 files for upload.
    """
    logging.info(f"Extracting image index {index} from '{h5_path}'...")
    
    with h5py.File(h5_path, 'r') as f:
        pan = np.array(f['pan'])[index]
        ms = np.array(f['ms'])[index]
        gt = np.array(f['gt'])[index]
        
    pan_path = f"{out_prefix}_pan.h5"
    ms_path  = f"{out_prefix}_ms.h5"
    gt_path  = f"{out_prefix}_gt.h5"
    
    # Save standard HDF5 datasets
    with h5py.File(pan_path, 'w') as f: f.create_dataset('pan', data=pan)
    with h5py.File(ms_path, 'w') as f: f.create_dataset('ms', data=ms)
    with h5py.File(gt_path, 'w') as f: f.create_dataset('gt', data=gt)
    
    logging.info(f"Successfully generated temporary chunks: {pan_path}, {ms_path}, {gt_path}")
    return pan_path, ms_path, gt_path


def cleanup(*paths):
    """Deletes temporary files"""
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def ping_container(name, url, pan_path, ms_path, gt_path, extra_files=None, extra_data=None):
    """
    Fires off a robust multipart/form-data POST request to a given container backend.
    """
    logging.info(f"Pinging {name} container at '{url}'...")
    
    # The file tuple format: (filename, file_object, content_type)
    files = {
        'pan_file': (pan_path, open(pan_path, 'rb'), 'application/x-hdf5'),
        'ms_file':  (ms_path, open(ms_path, 'rb'), 'application/x-hdf5'),
        'gt_file':  (gt_path, open(gt_path, 'rb'), 'application/x-hdf5')
    }
    
    if extra_files:
        files.update(extra_files)
        
    data = {'divisor': DIVISOR}
    if extra_data:
        data.update(extra_data)
        
    try:
        req = requests.post(url, files=files, data=data, timeout=300)
        req.raise_for_status()
        resp = req.json()
        
        if resp.get('error'):
            logging.error(f"[{name}] Backend processed request but returned an internal logic error: {resp['error']}")
        else:
            logging.info(f"[{name}] Successfully calculated metrics.")
            
        return resp

    except Exception as e:
        logging.error(f"Failed to ping {name}: {e}")
        return {"error": str(e)}

    finally:
        # Prevent dangling open file descriptors
        for field_name, file_tuple in files.items():
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()


def main():
    if not os.path.exists(DATASET_PATH):
        logging.error(f"Dataset not found at {DATASET_PATH}. Please check the path and try again.")
        return

    # Extract single frame chunk
    pan_p, ms_p, gt_p = extract_and_save_temp(DATASET_PATH, INDEX_TO_EXTRACT)

    # -------------------------------------
    # 1. Pipeline: GOD (Neural Network)
    # -------------------------------------
    god_extra_files = None
    if os.path.exists(GOD_MODEL_PATH):
        god_extra_files = {
            'model_file': (GOD_MODEL_PATH, open(GOD_MODEL_PATH, 'rb'), 'application/octet-stream')
        }
        god_data = {
            'pan_target_ch': PAN_TARGET_CH,
            'ms_target_ch': MS_TARGET_CH,
            'head_ch': HEAD_CH
        }
        god_resp = ping_container("GOD", GOD_URL, pan_p, ms_p, gt_p, extra_files=god_extra_files, extra_data=god_data)

    else:
        # Provide a graceful warning to the user instead of outright breaking
        logging.warning("="*60)
        logging.warning(f" GOD model file not found at: '{GOD_MODEL_PATH}'")
        logging.warning(" The GOD pipeline requires a PyTorch weights file to run.")
        logging.warning(" Please edit the 'GOD_MODEL_PATH' variable at the top of this script.")
        logging.warning(" Skipping GOD ping...")
        logging.warning("="*60)
        god_resp = {"error": f"Model file '{GOD_MODEL_PATH}' not found. Update configuration."}
        

    # -------------------------------------
    # 2. Pipeline: Brovey Transform
    # -------------------------------------
    brovey_resp = ping_container("Brovey", BROVEY_URL, pan_p, ms_p, gt_p)
    

    # -------------------------------------
    # 3. Pipeline: RGB-IHS Transform
    # -------------------------------------
    ihs_resp = ping_container("IHS", IHS_URL, pan_p, ms_p, gt_p)


    # -------------------------------------
    # Write Result Payloads
    # -------------------------------------
    logging.info("Dumping final JSON metrics and container visualizations to disk...")
    
    with open('results_god.json', 'w') as f:
        json.dump(god_resp, f, indent=4)
        
    with open('results_brovey.json', 'w') as f:
        json.dump(brovey_resp, f, indent=4)
        
    with open('results_ihs.json', 'w') as f:
        json.dump(ihs_resp, f, indent=4)

    # Cleanup .h5 stream chunks
    cleanup(pan_p, ms_p, gt_p)
    logging.info("Automation sequence completed. Check 'results_*.json' for output data.")

if __name__ == "__main__":
    main()
