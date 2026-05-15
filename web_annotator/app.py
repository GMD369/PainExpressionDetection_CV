from flask import Flask, render_template, request, jsonify, send_file, session
import os, json, csv, io, uuid, zipfile, shutil
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.secret_key = "pain-annotator-2024"

UPLOAD_ROOT = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

PAIN_LABELS = ["no_pain", "mild_pain", "moderate_pain", "severe_pain", "extreme_pain"]
COLORS = {
    "no_pain":       "#2ecc71",
    "mild_pain":     "#f1c40f",
    "moderate_pain": "#e67e22",
    "severe_pain":   "#e74c3c",
    "extreme_pain":  "#8e44ad",
}
SAVE_FILE = "annotations.json"
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── Per-user state ─────────────────────────────────────────────────────────────
_sessions: dict = {}

def get_state() -> dict:
    sid = session.get("id")
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        session["id"] = sid
        _sessions[sid] = {"folder": "", "image_list": [], "annotations": {}}
    return _sessions[sid]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    get_state()
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    st    = get_state()
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    sid        = session["id"]
    upload_dir = os.path.join(UPLOAD_ROOT, sid)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)

    images = []
    for file in files:
        # webkitRelativePath arrives as filename: "FolderName/sub/file.png"
        rel   = file.filename.replace("\\", "/")
        parts = rel.split("/")
        # strip the top-level folder name that the browser prepends
        rel   = "/".join(parts[1:]) if len(parts) > 1 else parts[0]

        if not rel or os.path.splitext(rel)[1].lower() not in IMG_EXTS:
            continue

        dest = os.path.join(upload_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        file.save(dest)
        images.append(rel)

    if not images:
        return jsonify({"error": "No valid image files found in upload"}), 400

    images.sort()
    st["folder"]      = upload_dir
    st["image_list"]  = images
    st["annotations"] = {}

    return jsonify({"images": images, "annotations": {}})


@app.route("/api/image/<int:index>")
def get_image(index):
    st = get_state()
    il = st["image_list"]
    if not il or index < 0 or index >= len(il):
        return "Not found", 404
    return send_file(os.path.join(st["folder"], il[index]))


@app.route("/api/annotations", methods=["POST"])
def save_annotations():
    st = get_state()
    if not st["folder"]:
        return jsonify({"error": "No dataset loaded"}), 400
    body = request.get_json() or {}
    st["annotations"] = body.get("annotations", st["annotations"])
    with open(os.path.join(st["folder"], SAVE_FILE), "w") as f:
        json.dump(st["annotations"], f, indent=2)
    return jsonify({"ok": True})


@app.route("/api/export-csv")
def export_csv():
    st  = get_state()
    ann = st["annotations"]
    if not ann:
        return jsonify({"error": "No annotations to export"}), 400

    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(["filename", "classification", "ann_type",
                "x1", "y1", "x2", "y2", "label", "polygon_pts", "note"])
    for fname, data in ann.items():
        cls = data.get("classification", "")
        for b in data.get("boxes", []):
            w.writerow([fname, cls, "bbox",
                        b["x1"], b["y1"], b["x2"], b["y2"], b["label"], "", ""])
        for p in data.get("polygons", []):
            w.writerow([fname, cls, "polygon",
                        "", "", "", "", p["label"], str(p["points"]), ""])
        if not data.get("boxes") and not data.get("polygons"):
            w.writerow([fname, cls, "classification",
                        "", "", "", "", cls, "", ""])

    buf = io.BytesIO(si.getvalue().encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name="annotations.csv")


@app.route("/api/export-images")
def export_images():
    st  = get_state()
    ann = st["annotations"]
    if not ann:
        return jsonify({"error": "No annotations to export"}), 400

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in ann.items():
            img_path = os.path.join(st["folder"], fname)
            if not os.path.exists(img_path):
                continue

            img     = Image.open(img_path).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)

            # Polygon fills (semi-transparent) on overlay layer
            for poly in data.get("polygons", []):
                rgb = _hex_rgb(COLORS.get(poly["label"], "#ffffff"))
                pts = [tuple(p) for p in poly["points"]]
                ov_draw.polygon(pts, fill=(*rgb, 70), outline=(*rgb, 255))

            img  = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

            # Bounding boxes
            for box in data.get("boxes", []):
                rgb   = _hex_rgb(COLORS.get(box["label"], "#ffffff"))
                coords = [box["x1"], box["y1"], box["x2"], box["y2"]]
                draw.rectangle(coords, outline=rgb, width=3)
                lbl  = box["label"].replace("_", " ").title()
                tw, th = _txt_size(draw, lbl, font)
                draw.rectangle(
                    [box["x1"], box["y1"] - th - 6, box["x1"] + tw + 10, box["y1"]],
                    fill=(*rgb, 220)
                )
                draw.text((box["x1"] + 5, box["y1"] - th - 3),
                          lbl, fill=(255, 255, 255), font=font)

            # Classification badge (top-left corner)
            cls = data.get("classification", "")
            if cls:
                rgb   = _hex_rgb(COLORS.get(cls, "#ffffff"))
                badge = cls.replace("_", " ").upper()
                tw, th = _txt_size(draw, badge, font)
                draw.rectangle([4, 4, tw + 16, th + 14], fill=(*rgb, 230))
                draw.text((10, 8), badge, fill=(255, 255, 255), font=font)

            out = io.BytesIO()
            img.convert("RGB").save(out, format="PNG")
            out.seek(0)
            zf.writestr(f"annotated/{fname}", out.getvalue())

    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="annotated_images.zip")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _hex_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _txt_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        return len(text) * 8, 14


if __name__ == "__main__":
    app.run(debug=True, port=5000)
