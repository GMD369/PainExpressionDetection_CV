import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / 'week4' / 'Week4_Results' / 'best_seg.pt'
CLASSES    = ['No_Pain', 'Mild', 'Moderate', 'Severe']
COLORS     = {
    'No_Pain':  (46,  204, 113),
    'Mild':     (241, 196,  15),
    'Moderate': (230, 126,  34),
    'Severe':   (231,  76,  60),
}
PAIN_INFO  = {
    'No_Pain':  ('No pain detected',        '#2ecc71'),
    'Mild':     ('Mild pain expression',    '#f1c40f'),
    'Moderate': ('Moderate pain expression','#e67e22'),
    'Severe':   ('Severe pain expression',  '#e74c3c'),
}

model = YOLO(str(MODEL_PATH))

def predict(image: np.ndarray):
    if image is None:
        return None, "No image provided."

    results = model.predict(
        source=image,
        imgsz=224,
        conf=0.25,
        device='cpu',
        verbose=False,
    )[0]

    annotated = image.copy()
    detections = []

    if results.masks is not None:
        for i, (mask_xy, cls_id, conf) in enumerate(zip(
            results.masks.xy,
            results.boxes.cls.int().tolist(),
            results.boxes.conf.tolist(),
        )):
            cls_name = CLASSES[cls_id]
            color    = COLORS[cls_name]

            # Draw filled mask
            if len(mask_xy) > 0:
                pts = mask_xy.astype(np.int32)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], color)
                annotated = cv2.addWeighted(annotated, 0.55, overlay, 0.45, 0)
                cv2.polylines(annotated, [pts], True, color, 2)

            # Draw label
            if results.boxes is not None:
                x1, y1, x2, y2 = results.boxes.xyxy[i].int().tolist()
                label = f'{cls_name}  {conf*100:.1f}%'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                cv2.putText(annotated, label, (x1+3, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

            detections.append((cls_name, conf))

    if not detections:
        # No face / detection — show plain image with note
        msg = "No pain expression detected (or face not found). Try a clearer frontal face image."
        return annotated, msg

    # Build result text
    best_cls, best_conf = max(detections, key=lambda x: x[1])
    desc, _ = PAIN_INFO[best_cls]
    lines = [f"Result: {desc}  ({best_conf*100:.1f}% confidence)"]
    if len(detections) > 1:
        lines.append(f"Total detections: {len(detections)}")
    for cls, conf in detections:
        lines.append(f"  • {cls:<12} {conf*100:.1f}%")

    return annotated, "\n".join(lines)


with gr.Blocks(title="Pain Expression Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Pain Expression Detection
    **Model:** YOLOv8n-seg | **Classes:** No_Pain · Mild · Moderate · Severe
    **Mask mAP@0.5:** 98.79% &nbsp;|&nbsp; **Precision:** 97.51% &nbsp;|&nbsp; **Recall:** 97.27%
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload Face Image", type="numpy", height=320)
            btn = gr.Button("Detect Pain Expression", variant="primary")

        with gr.Column(scale=1):
            out_img = gr.Image(label="Segmentation Result", height=320)
            out_txt = gr.Textbox(label="Detection Result", lines=6)

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_txt])
    inp.change(fn=predict, inputs=inp, outputs=[out_img, out_txt])

    gr.Markdown("""
    ---
    **Pain Level Mapping** (based on facial AUs):
    `No Pain` — neutral / happiness &nbsp;|&nbsp; `Mild` — sadness
    `Moderate` — fear / surprise &nbsp;|&nbsp; `Severe` — anger / disgust
    """)

if __name__ == '__main__':
    demo.launch(share=False, inbrowser=True)
