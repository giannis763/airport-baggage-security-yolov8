import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import json

# ════════════════════════════════════════
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
# ════════════════════════════════════════

COLORS = [
    (255, 59,  48),   # Gun
    (255, 149, 0),    # Knife
    (255, 204, 0),    # Plier
    (52,  199, 89),   # Razor blade
    (0,   122, 255),  # Screw
    (175, 82,  222),  # Shuriken
]

THREAT_CLASSES = ["Gun", "Knife", "Razor blade", "Shuriken"]

def detect(image, conf_threshold=0.25):
    if image is None:
        return None, "⚠ No image provided."

    img = Image.fromarray(image).convert("RGB")
    results = model.predict(source=np.array(img), conf=conf_threshold, iou=0.45, verbose=False)
    pred = results[0]

    draw = ImageDraw.Draw(img)
    detections = []

    if pred.boxes and len(pred.boxes) > 0:
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            name  = model.names[cls]
            color = COLORS[cls % len(COLORS)]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{name} {conf:.0%}"
            bbox_text = draw.textbbox((x1, y1 - 22), label)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, y1 - 22), label, fill="white")

            detections.append({
                "class": name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

        threat = any(d["class"] in THREAT_CLASSES for d in detections)
        status = "🚨 THREAT DETECTED" if threat else "⚠ SUSPICIOUS ITEM"

        summary = f"{status}\n\n"
        summary += f"Total detections: {len(detections)}\n"
        summary += "─" * 35 + "\n"
        for d in detections:
            icon = "🔴" if d["class"] in THREAT_CLASSES else "🟡"
            summary += f"{icon}  {d['class']:<15} {d['confidence']*100:.1f}%\n"
        summary += "─" * 35 + "\n"
        summary += f"\nJSON:\n{json.dumps(detections, indent=2)}"
    else:
        summary = "✅  CLEAR\n\nNo prohibited items detected.\n\nJSON:\n[]"

    return img, summary


css = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

body, .gradio-container {
    background: #0a0e1a !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: #e2e8f0 !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

.header-block {
    background: linear-gradient(135deg, #0a0e1a, #0d1b2a, #0a0e1a);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.header-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.header-block h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    letter-spacing: 3px !important; color: #00d4ff !important;
    text-transform: uppercase !important; margin: 0 !important;
}
.header-block p {
    font-family: 'Share Tech Mono', monospace !important;
    color: #64748b !important; font-size: 0.8rem !important;
    margin: 8px 0 0 !important; letter-spacing: 1px !important;
}

.metrics-bar { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card {
    flex: 1; background: #1a2035;
    border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 16px 18px; text-align: center;
    position: relative; overflow: hidden;
}
.metric-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: #00d4ff; opacity: 0.5;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.6rem !important; color: #00d4ff !important; line-height: 1;
}
.metric-label {
    font-size: 0.7rem !important; color: #64748b !important;
    letter-spacing: 2px !important; text-transform: uppercase; margin-top: 6px;
}

input[type=range] { accent-color: #00d4ff !important; }
label span {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    color: #e2e8f0 !important;
}

.gr-button-primary, button.primary {
    background: #00d4ff !important; color: #000 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 6px !important;
    padding: 12px 28px !important; width: 100% !important;
    margin-top: 14px !important; transition: all 0.2s !important;
}
button.primary:hover {
    background: #00b8d9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
}

textarea {
    background: #0d1117 !important; color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; line-height: 1.6 !important;
}

.footer-block {
    background: #1a2035; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 14px 20px; margin-top: 16px;
    display: flex; justify-content: space-between; align-items: center;
}
.footer-block span {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important; color: #64748b !important; letter-spacing: 1px;
}
.status-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #34c759; border-radius: 50%;
    margin-right: 6px; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
"""

with gr.Blocks(title="Airport Security AI") as demo:

    gr.HTML(f"<style>{css}</style>")

    gr.HTML("""
    <div class="header-block">
        <h1>✈ Airport Security AI</h1>
        <p>SHARP OBJECTS DETECTION SYSTEM &nbsp;|&nbsp; YOLOV8s &nbsp;|&nbsp; GDXRAY DATASET</p>
    </div>
    """)

    gr.HTML("""
    <div class="metrics-bar">
        <div class="metric-card"><div class="metric-value">91.3%</div><div class="metric-label">mAP @ 0.5</div></div>
        <div class="metric-card"><div class="metric-value">94.1%</div><div class="metric-label">Precision</div></div>
        <div class="metric-card"><div class="metric-value">90.5%</div><div class="metric-label">Recall</div></div>
        <div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Classes</div></div>
        <div class="metric-card"><div class="metric-value">5,566</div><div class="metric-label">Train Images</div></div>
    </div>
    """)

    conf_slider = gr.Slider(
        minimum=0.10, maximum=0.90, value=0.25, step=0.05,
        label="CONFIDENCE THRESHOLD",
        info="Lower = more detections · Higher = more certain"
    )

    with gr.Row(equal_height=True):
        input_img  = gr.Image(label="INPUT IMAGE", type="numpy", height=380, sources=["upload"])
        output_img = gr.Image(label="DETECTIONS",  type="pil",  height=380, interactive=False)

    output_txt = gr.Textbox(label="ANALYSIS OUTPUT", lines=12)

    btn = gr.Button("⟶  RUN DETECTION", variant="primary")
    btn.click(fn=detect, inputs=[input_img, conf_slider], outputs=[output_img, output_txt])

    gr.HTML("""
    <div class="footer-block">
        <span><span class="status-dot"></span>SYSTEM ONLINE</span>
        <span>CLASSES: GUN · KNIFE · PLIER · RAZOR BLADE · SCREW · SHURIKEN</span>
        <span>MODEL: YOLOV8s · 11.1M PARAMS</span>
    </div>
    """)

demo.launch(share=False)