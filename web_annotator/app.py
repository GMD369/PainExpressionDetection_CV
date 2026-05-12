from flask import Flask, render_template, request, jsonify, send_file
import os, json, csv, io

app = Flask(__name__)

PAIN_LABELS = ["no_pain", "mild_pain", "moderate_pain", "severe_pain", "extreme_pain"]
SAVE_FILE   = "annotations.json"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_state = {"folder": "", "image_list": [], "annotations": {}}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/open-folder", methods=["POST"])
def open_folder():
    data   = request.get_json() or {}
    folder = data.get("path", "").strip()

    if not os.path.isdir(folder):
        return jsonify({"error": "Folder not found"}), 400

    # Collect images from the folder itself and any direct subfolders.
    # Paths are stored relative to `folder` using forward slashes as keys.
    images = []

    flat = sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    images.extend(flat)

    for sub in sorted(os.listdir(folder)):
        sub_abs = os.path.join(folder, sub)
        if not os.path.isdir(sub_abs):
            continue
        sub_imgs = sorted([
            f"{sub}/{f}"
            for f in os.listdir(sub_abs)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        images.extend(sub_imgs)

    if not images:
        return jsonify({"error": "No images found in folder or its subfolders"}), 400

    _state["folder"]     = folder
    _state["image_list"] = images

    # Load root-level annotations.json if it exists; otherwise try to merge
    # any per-subfolder annotations.json files (re-keyed with subfolder prefix).
    save_path = os.path.join(folder, SAVE_FILE)
    if os.path.exists(save_path):
        with open(save_path) as f:
            _state["annotations"] = json.load(f)
    else:
        merged = {}
        for sub in sorted(os.listdir(folder)):
            sub_abs = os.path.join(folder, sub)
            sub_ann = os.path.join(sub_abs, SAVE_FILE)
            if os.path.isdir(sub_abs) and os.path.exists(sub_ann):
                with open(sub_ann) as f:
                    for fname, ann in json.load(f).items():
                        merged[f"{sub}/{fname}"] = ann
        _state["annotations"] = merged

    return jsonify({"images": images, "annotations": _state["annotations"]})


@app.route("/api/image/<int:index>")
def get_image(index):
    il = _state["image_list"]
    if not il or index < 0 or index >= len(il):
        return "Not found", 404
    return send_file(os.path.join(_state["folder"], il[index]))


@app.route("/api/annotations", methods=["POST"])
def save_annotations():
    if not _state["folder"]:
        return jsonify({"error": "No folder loaded"}), 400
    body = request.get_json() or {}
    _state["annotations"] = body.get("annotations", _state["annotations"])
    with open(os.path.join(_state["folder"], SAVE_FILE), "w") as f:
        json.dump(_state["annotations"], f, indent=2)
    return jsonify({"ok": True})


@app.route("/api/export-csv")
def export_csv():
    ann = _state["annotations"]
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
