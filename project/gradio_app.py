import gradio as gr
import torch
import numpy as np
import h5py
import cv2
import os
import torch.nn.functional as F

from fusion_model import HWViT
from metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image_or_array(file_path):
    if file_path is None: return None
    
    if isinstance(file_path, str):
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                # Prioritize keys that might be standard in the project
                for k in ['pan', 'ms', 'lms', 'gt', 'data', 'img', 'image']:
                    if k in f: return np.array(f[k])
                return np.array(list(f.values())[0])
        else:
            # Try to read as a standard image
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read {file_path}")
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    elif hasattr(file_path, 'name'):
        return load_image_or_array(file_path.name)
    else:
        # Assuming it is already an array
        return file_path

def process_images(model_path, pan_file, ms_file, gt_file, 
                   pan_target_channel, ms_target_channel, head_channel, 
                   divisor):
    try:
        pan_img = load_image_or_array(pan_file)
        ms_img  = load_image_or_array(ms_file)
        gt_img  = load_image_or_array(gt_file)
        
        if pan_img is None or ms_img is None or gt_img is None:
            return "❌ Please provide all 3 images (MS, PAN, GT).", None, None

        if not os.path.exists(model_path):
            return f"❌ Model path '{model_path}' not found.", None, None

        # Convert simple (H, W) to (H, W, 1)
        if len(pan_img.shape) == 2: pan_img = np.expand_dims(pan_img, 2)
        if len(ms_img.shape) == 2: ms_img = np.expand_dims(ms_img, 2)
        if len(gt_img.shape) == 2: gt_img = np.expand_dims(gt_img, 2)

        # Transpose to (C, H, W)
        pan_img = pan_img.transpose(2, 0, 1)
        ms_img = ms_img.transpose(2, 0, 1)
        gt_img = gt_img.transpose(2, 0, 1)

        # Normalize 
        pan_t = torch.tensor(pan_img / divisor).float().unsqueeze(0)
        ms_t = torch.tensor(ms_img / divisor).float().unsqueeze(0)
        gt_t = torch.tensor(gt_img / divisor).float().unsqueeze(0)

        # Upsample MS to PAN spatial size to generate LMS
        _, _, H_pan, W_pan = pan_t.shape
        lms_t = F.interpolate(ms_t, size=(H_pan, W_pan), mode='bicubic', align_corners=False)

        pan_ch = pan_t.shape[1]
        L_up_ch = ms_t.shape[1]

        # Init Model
        model = HWViT(
            L_up_channel=L_up_ch,
            pan_channel=pan_ch,
            ms_target_channel=int(ms_target_channel),
            pan_target_channel=int(pan_target_channel),
            head_channel=int(head_channel),
            dropout=0.0
        )
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        with torch.no_grad():
            pred_t = model(pan_t.to(device), ms_t.to(device), lms_t.to(device))
        
        pred_np = pred_t.cpu().numpy()
        gt_np = gt_t.numpy()

        # Metrics Calculate
        metrics = compute_metrics(pred_np, gt_np)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics_md = (
            f"### Extracted Objectives\n\n"
            f"| Metric | Value |\n"
            f"|:------:|:-----:|\n"
            f"| **SAM** | {metrics['SAM']:.4f} |\n"
            f"| **ERGAS** | {metrics['ERGAS']:.4f} |\n"
            f"| **CC** | {metrics['CC']:.4f} |\n"
            f"| **SD** | {metrics['SD']:.4f} |\n"
            f"| **SF** | {metrics['SF']:.4f} |\n"
            f"| **SSIM** | {metrics['SSIM']:.4f} |\n"
            f"| **n_params** | {n_params:,} |\n"
        )
        
        # Vis preprocessing
        def to_vis(img_array):
            img = img_array[0].transpose(1, 2, 0)
            if img.shape[2] >= 3:
                img = img[:, :, :3]
            # Convert back to standard 8-bit dynamic range to render securely in Gradio
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            return img
        
        # The pred_np is normalized using divisor, so it's in [0, 1] range. 
        # But if the data had original dynamic ranges mapping different, it's safer to just multiply by 255.
        pred_vis = to_vis(pred_np)
        gt_vis = to_vis(gt_np)

        return metrics_md, pred_vis, gt_vis
    except Exception as e:
        import traceback
        return f"### Error Occurred:\n```\n{str(e)}\n{traceback.format_exc()}\n```", None, None

# CSS injecting dynamic glassmorphism and animations!
custom_css = """
body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #121212) !important;
    background-size: 400% 400% !important;
    animation: gradientBG 15s ease infinite !important;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.gradio-container {
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.glass-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}
#run-btn {
    background: linear-gradient(90deg, #ff8a00, #e52e71) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(229, 46, 113, 0.3) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    border-radius: 12px !important;
    font-weight: bold;
    font-size: 1.1em;
}
#run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(229, 46, 113, 0.6) !important;
}
.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    text-align: center;
    color: #ffffff;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
"""

theme = gr.themes.Soft(
    primary_hue="pink",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(0, 0, 0, 0.3)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.1)",
    block_title_text_color="rgba(255,255,255,0.9)",
    input_background_fill="rgba(255, 255, 255, 0.05)",
    button_primary_background_fill="#ff8a00"
)

with gr.Blocks(theme=theme, css=custom_css) as app:
    with gr.Column(elem_classes=["glass-panel"], scale=1):
        gr.Markdown(
            """
            # 🌌 Spatial Computing Pansharpening Inference
            Upload your low-resolution Multispectral image, high-resolution Panchromatic image, and Ground Truth. Evaluate any optimization checkpoint instantly.
            """,
            elem_classes=["markdown-text"]
        )
    
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Load Images")
                pan_input = gr.File(label="PAN Image (.npy, .h5, standard info)", file_types=[".npy", ".h5", "image"])
                ms_input  = gr.File(label="MS Image (.npy, .h5, standard info)", file_types=[".npy", ".h5", "image"])
                gt_input  = gr.File(label="GT Image (.npy, .h5, standard info)", file_types=[".npy", ".h5", "image"])
            
            with gr.Column():
                gr.Markdown("### 2. Configure Model")
                model_path = gr.Textbox(label="Model Path (.pt file)", placeholder="e.g. results/all_models/model_checkpoint.pt")
                pan_target_ch = gr.Number(label="PAN Target Channel", value=32, precision=0)
                ms_target_ch = gr.Number(label="MS Target Channel", value=32, precision=0)
                head_ch = gr.Number(label="Attention Head Channel", value=8, precision=0)
                divisor = gr.Number(label="Image Divisor (Scale to [0,1])", value=2047, precision=0)
                
                run_btn = gr.Button("🚀 Run Inference Pipeline", elem_id="run-btn")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Evaluation Metrics")
                metrics_output = gr.Markdown("Waiting for inference...")
            with gr.Column():
                gr.Markdown("### Visualization")
                fused_viz = gr.Image(label="Fused Image Prediction")
                gt_viz = gr.Image(label="Ground Truth Reference")

    run_btn.click(
        fn=process_images,
        inputs=[model_path, pan_input, ms_input, gt_input, pan_target_ch, ms_target_ch, head_ch, divisor],
        outputs=[metrics_output, fused_viz, gt_viz]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
