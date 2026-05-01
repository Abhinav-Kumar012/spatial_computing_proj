import gradio as gr
import requests
import os

GOD_URL = os.environ.get("GOD_URL", "http://backend_god:8000/fuse/god")
BROVEY_URL = os.environ.get("BROVEY_URL", "http://backend_brovey:8001/fuse/brovey")

def process_images(pipeline_choice, model_path, pan_file, ms_file, gt_file, 
                   pan_target_ch, ms_target_ch, head_ch, divisor):
    if not pan_file or not ms_file or not gt_file:
        return "❌ Please provide all 3 images (MS, PAN, GT).", None, None
        
    payload = {
        "pan_file": pan_file.name if hasattr(pan_file, 'name') else pan_file,
        "ms_file": ms_file.name if hasattr(ms_file, 'name') else ms_file,
        "gt_file": gt_file.name if hasattr(gt_file, 'name') else gt_file,
        "model_path": model_path,
        "pan_target_ch": int(pan_target_ch),
        "ms_target_ch": int(ms_target_ch),
        "head_ch": int(head_ch),
        "divisor": float(divisor)
    }
    
    url = GOD_URL if pipeline_choice == "GOD (Neural Net)" else BROVEY_URL
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        if data.get("error"):
            return f"### Error Occurred:\n```\n{data['error']}\n```", None, None
            
        return data["metrics_md"], data["pred_vis_path"], data["gt_vis_path"]
        
    except requests.exceptions.RequestException as e:
        return f"### Network Error:\nCould not reach backend microservice at {url}.\n```\n{str(e)}\n```", None, None

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
            # 🌌 Spatial Computing Pansharpening Inference (Microservice Architecture)
            Select your fusion pipeline and upload imagery to trigger backend ML processing across the internal container network.
            """,
            elem_classes=["markdown-text"]
        )
    
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Load Images & Select Pipeline")
                pipeline_choice = gr.Radio(
                    choices=["GOD (Neural Net)", "Brovey Transform"], 
                    value="GOD (Neural Net)", 
                    label="Active Fusion Pipeline"
                )
                pan_input = gr.File(label="PAN Image (.npy, .h5, standard img)", file_types=[".npy", ".h5", "image"])
                ms_input  = gr.File(label="MS Image (.npy, .h5, standard img)", file_types=[".npy", ".h5", "image"])
                gt_input  = gr.File(label="GT Image (.npy, .h5, standard img)", file_types=[".npy", ".h5", "image"])
            
            with gr.Column():
                gr.Markdown("### 2. Configure Model (For GOD)")
                model_path = gr.Textbox(label="Model Path (.pt file inside volume)", placeholder="e.g. results/all_models/model_checkpoint.pt")
                with gr.Row():
                    pan_target_ch = gr.Number(label="PAN Target Ch", value=32, precision=0)
                    ms_target_ch = gr.Number(label="MS Target Ch", value=32, precision=0)
                with gr.Row():
                    head_ch = gr.Number(label="Attention Head Ch", value=8, precision=0)
                    divisor = gr.Number(label="Image Divisor", value=2047, precision=0)
                
                run_btn = gr.Button("🚀 Run Inference Pipeline", elem_id="run-btn")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Evaluation Metrics")
                metrics_output = gr.Markdown("Waiting for inference...")
            with gr.Column():
                gr.Markdown("### Visualization")
                fused_viz = gr.Image(label="Fused Image Prediction (From Shared Volume)", type="filepath")
                gt_viz = gr.Image(label="Ground Truth Reference (From Shared Volume)", type="filepath")

    run_btn.click(
        fn=process_images,
        inputs=[pipeline_choice, model_path, pan_input, ms_input, gt_input, pan_target_ch, ms_target_ch, head_ch, divisor],
        outputs=[metrics_output, fused_viz, gt_viz]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
