"""
Mini Medical Image Annotation Tool
Pain Expression Detection Project
Supports: Classification, Bounding Box (Detection), Polygon (Segmentation)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from PIL import Image, ImageTk
import csv

# ── Constants ────────────────────────────────────────────────────────────────
PAIN_LABELS = ["No_Pain", "Mild", "Moderate", "Severe"]
COLORS      = {"No_Pain": "#2ecc71", "Mild": "#f1c40f",
               "Moderate": "#e67e22", "Severe": "#e74c3c"}
MODES       = ["Classification", "Detection (BBox)", "Segmentation (Polygon)"]
SAVE_FILE   = "annotations.json"


# ── Main Application ─────────────────────────────────────────────────────────
class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pain Expression Annotator")
        self.root.configure(bg="#1e1e2e")
        self.root.state("zoomed")

        # State
        self.image_folder  = ""
        self.image_list    = []
        self.current_index = 0
        self.annotations   = {}       # { filename: {classification, boxes, polygons} }
        self.current_image = None
        self.tk_image      = None
        self.scale         = 1.0

        # Drawing state
        self.mode          = tk.StringVar(value=MODES[0])
        self.selected_label= tk.StringVar(value=PAIN_LABELS[0])
        self.drawing       = False
        self.start_x = self.start_y = 0
        self.current_rect  = None
        self.polygon_points= []
        self.poly_dots     = []
        self.poly_lines    = []
        self.drawn_items   = []       # canvas item ids for current image

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, bg="#181825", pady=6)
        toolbar.pack(fill="x")

        tk.Button(toolbar, text="Open Folder", command=self.open_folder,
                  bg="#7c3aed", fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=12).pack(side="left", padx=6)
        tk.Button(toolbar, text="Save", command=self.save_annotations,
                  bg="#059669", fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=12).pack(side="left", padx=4)
        tk.Button(toolbar, text="Export CSV", command=self.export_csv,
                  bg="#0369a1", fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=12).pack(side="left", padx=4)
        tk.Button(toolbar, text="Clear All", command=self.clear_annotations,
                  bg="#dc2626", fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=12).pack(side="left", padx=4)

        # Status label (right side of toolbar)
        self.status_var = tk.StringVar(value="No folder loaded")
        tk.Label(toolbar, textvariable=self.status_var,
                 bg="#181825", fg="#94a3b8",
                 font=("Segoe UI", 9)).pack(side="right", padx=12)

        # Main layout: left panel + canvas + right panel
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True)

        self._build_left_panel(main)
        self._build_canvas(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        panel = tk.Frame(parent, bg="#181825", width=220)
        panel.pack(side="left", fill="y", padx=(8, 0), pady=8)
        panel.pack_propagate(False)

        self._section(panel, "ANNOTATION MODE")
        for m in MODES:
            tk.Radiobutton(panel, text=m, variable=self.mode, value=m,
                           bg="#181825", fg="#e2e8f0", selectcolor="#7c3aed",
                           activebackground="#181825",
                           font=("Segoe UI", 9),
                           command=self._on_mode_change).pack(anchor="w", padx=12, pady=2)

        self._section(panel, "PAIN LABEL")
        for lbl in PAIN_LABELS:
            color = COLORS[lbl]
            tk.Radiobutton(panel, text=lbl.replace("_", " ").title(),
                           variable=self.selected_label, value=lbl,
                           bg="#181825", fg=color, selectcolor="#7c3aed",
                           activebackground="#181825",
                           font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=2)

        self._section(panel, "NAVIGATION")
        nav = tk.Frame(panel, bg="#181825")
        nav.pack(fill="x", padx=8, pady=4)
        tk.Button(nav, text="◀ Prev", command=self.prev_image,
                  bg="#334155", fg="white", relief="flat",
                  font=("Segoe UI", 9)).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(nav, text="Next ▶", command=self.next_image,
                  bg="#334155", fg="white", relief="flat",
                  font=("Segoe UI", 9)).pack(side="right", expand=True, fill="x", padx=2)

        self.progress_var = tk.StringVar(value="0 / 0")
        tk.Label(panel, textvariable=self.progress_var,
                 bg="#181825", fg="#94a3b8",
                 font=("Segoe UI", 9)).pack(pady=4)

        self._section(panel, "INSTRUCTIONS")
        tips = [
            "Classification:",
            "  Select label → click Apply",
            "",
            "Detection (BBox):",
            "  Click & drag on image",
            "",
            "Segmentation:",
            "  Click to add points",
            "  Double-click to close",
            "  Right-click to undo point",
        ]
        for t in tips:
            tk.Label(panel, text=t, bg="#181825",
                     fg="#64748b", font=("Courier", 8),
                     anchor="w").pack(fill="x", padx=12)

        # Apply classification button
        tk.Button(panel, text="Apply Classification",
                  command=self.apply_classification,
                  bg="#7c3aed", fg="white", relief="flat",
                  font=("Segoe UI", 9, "bold")).pack(fill="x", padx=12, pady=(12, 4))

        # Finish polygon button
        tk.Button(panel, text="Finish Polygon",
                  command=self.finish_polygon,
                  bg="#0369a1", fg="white", relief="flat",
                  font=("Segoe UI", 9, "bold")).pack(fill="x", padx=12, pady=4)

    def _build_canvas(self, parent):
        canvas_frame = tk.Frame(parent, bg="#1e1e2e")
        canvas_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.filename_var = tk.StringVar(value="No image loaded")
        tk.Label(canvas_frame, textvariable=self.filename_var,
                 bg="#1e1e2e", fg="#e2e8f0",
                 font=("Segoe UI", 10, "bold")).pack()

        self.canvas = tk.Canvas(canvas_frame, bg="#0f0f1a",
                                cursor="crosshair",
                                highlightthickness=2,
                                highlightbackground="#7c3aed")
        self.canvas.pack(fill="both", expand=True)

        # Mouse bindings
        self.canvas.bind("<ButtonPress-1>",   self.on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>",        self.on_right_click)

        # Keyboard shortcuts
        self.root.bind("<Left>",  lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<s>",     lambda e: self.save_annotations())
        self.root.bind("<Delete>", lambda e: self.clear_annotations())

    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg="#181825", width=260)
        panel.pack(side="right", fill="y", padx=(0, 8), pady=8)
        panel.pack_propagate(False)

        self._section(panel, "CURRENT ANNOTATIONS")

        self.ann_text = tk.Text(panel, bg="#0f0f1a", fg="#e2e8f0",
                                font=("Courier", 8), relief="flat",
                                state="disabled", wrap="word")
        self.ann_text.pack(fill="both", expand=True, padx=8, pady=4)

        self._section(panel, "DATASET STATS")
        self.stats_text = tk.Text(panel, bg="#0f0f1a", fg="#94a3b8",
                                  font=("Courier", 8), relief="flat",
                                  state="disabled", height=10)
        self.stats_text.pack(fill="x", padx=8, pady=4)

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg="#181825", fg="#7c3aed",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        tk.Frame(parent, bg="#7c3aed", height=1).pack(fill="x", padx=12)

    # ── File Operations ───────────────────────────────────────────────────────
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self.image_folder = folder
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self.image_list = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not self.image_list:
            messagebox.showwarning("No Images", "No image files found in selected folder.")
            return

        # Load existing annotations if present
        save_path = os.path.join(folder, SAVE_FILE)
        if os.path.exists(save_path):
            with open(save_path) as f:
                self.annotations = json.load(f)

        self.current_index = 0
        self.load_image()
        self.update_stats()

    def save_annotations(self):
        if not self.image_folder:
            messagebox.showwarning("No Folder", "Open a folder first.")
            return
        path = os.path.join(self.image_folder, SAVE_FILE)
        with open(path, "w") as f:
            json.dump(self.annotations, f, indent=2)
        self.status_var.set(f"Saved → {SAVE_FILE}")
        self.root.after(3000, lambda: self.status_var.set(""))

    def export_csv(self):
        if not self.annotations:
            messagebox.showwarning("Nothing", "No annotations to export.")
            return
        path = os.path.join(self.image_folder, "annotations.csv")
        rows = []
        for fname, data in self.annotations.items():
            cls = data.get("classification", "")
            for box in data.get("boxes", []):
                rows.append([fname, cls, "bbox",
                             box["x1"], box["y1"], box["x2"], box["y2"],
                             box["label"], "", ""])
            for poly in data.get("polygons", []):
                pts = str(poly["points"])
                rows.append([fname, cls, "polygon", "", "", "", "",
                             poly["label"], pts, ""])
            if not data.get("boxes") and not data.get("polygons"):
                rows.append([fname, cls, "classification",
                             "", "", "", "", cls, "", ""])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "classification", "ann_type",
                             "x1", "y1", "x2", "y2", "label", "polygon_pts", "note"])
            writer.writerows(rows)
        messagebox.showinfo("Exported", f"CSV saved to:\n{path}")

    # ── Image Loading ─────────────────────────────────────────────────────────
    def load_image(self):
        if not self.image_list:
            return
        fname = self.image_list[self.current_index]
        path  = os.path.join(self.image_folder, fname)

        img = Image.open(path).convert("RGB")

        # Scale to fit canvas
        self.canvas.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        iw, ih  = img.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        new_w = int(iw * self.scale)
        new_h = int(ih * self.scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        self.current_image = img
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_offset_x = (cw - new_w) // 2
        self.img_offset_y = (ch - new_h) // 2

        self.canvas.delete("all")
        self.drawn_items = []
        self.polygon_points = []
        self.poly_dots = []
        self.poly_lines = []

        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                  anchor="nw", image=self.tk_image)

        self.filename_var.set(f"{fname}  [{self.current_index+1}/{len(self.image_list)}]")
        self.progress_var.set(f"{self.current_index+1} / {len(self.image_list)}")

        self._redraw_annotations(fname)
        self._update_ann_panel(fname)

    def _redraw_annotations(self, fname):
        data = self.annotations.get(fname, {})
        for box in data.get("boxes", []):
            self._draw_box_on_canvas(box["x1"], box["y1"],
                                     box["x2"], box["y2"], box["label"])
        for poly in data.get("polygons", []):
            self._draw_polygon_on_canvas(poly["points"], poly["label"])

    # ── Drawing ───────────────────────────────────────────────────────────────
    def on_mouse_press(self, event):
        mode = self.mode.get()
        if mode == "Detection (BBox)":
            self.drawing = True
            self.start_x, self.start_y = event.x, event.y
            self.current_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline=COLORS[self.selected_label.get()], width=2)
        elif mode == "Segmentation (Polygon)":
            self.polygon_points.append((event.x, event.y))
            dot = self.canvas.create_oval(event.x-3, event.y-3,
                                          event.x+3, event.y+3,
                                          fill=COLORS[self.selected_label.get()],
                                          outline="white")
            self.poly_dots.append(dot)
            if len(self.polygon_points) > 1:
                p1, p2 = self.polygon_points[-2], self.polygon_points[-1]
                line = self.canvas.create_line(*p1, *p2,
                                               fill=COLORS[self.selected_label.get()],
                                               width=2)
                self.poly_lines.append(line)

    def on_mouse_drag(self, event):
        if self.drawing and self.current_rect:
            self.canvas.coords(self.current_rect,
                               self.start_x, self.start_y, event.x, event.y)

    def on_mouse_release(self, event):
        if self.mode.get() == "Detection (BBox)" and self.drawing:
            self.drawing = False
            x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
            x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                self.canvas.delete(self.current_rect)
                return
            label = self.selected_label.get()
            # Convert to image coords
            ix1 = round((x1 - self.img_offset_x) / self.scale)
            iy1 = round((y1 - self.img_offset_y) / self.scale)
            ix2 = round((x2 - self.img_offset_x) / self.scale)
            iy2 = round((y2 - self.img_offset_y) / self.scale)

            fname = self.image_list[self.current_index]
            self.annotations.setdefault(fname, {"classification": "", "boxes": [], "polygons": []})
            self.annotations[fname]["boxes"].append(
                {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2, "label": label})

            self._draw_box_on_canvas(x1, y1, x2, y2, label, canvas_coords=True)
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            self._update_ann_panel(fname)
            self.update_stats()

    def on_double_click(self, event):
        if self.mode.get() == "Segmentation (Polygon)":
            self.finish_polygon()

    def on_right_click(self, event):
        if self.mode.get() == "Segmentation (Polygon)" and self.polygon_points:
            self.polygon_points.pop()
            if self.poly_dots:
                self.canvas.delete(self.poly_dots.pop())
            if self.poly_lines:
                self.canvas.delete(self.poly_lines.pop())

    def finish_polygon(self):
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Polygon", "Need at least 3 points.")
            return
        label  = self.selected_label.get()
        color  = COLORS[label]
        flat   = [c for pt in self.polygon_points for c in pt]
        self.canvas.create_polygon(flat, outline=color,
                                   fill=color, stipple="gray25", width=2)
        # Close line
        self.canvas.create_line(*self.polygon_points[-1],
                                *self.polygon_points[0],
                                fill=color, width=2)
        # Store in image coords
        img_pts = [
            (round((x - self.img_offset_x) / self.scale),
             round((y - self.img_offset_y) / self.scale))
            for x, y in self.polygon_points
        ]
        fname = self.image_list[self.current_index]
        self.annotations.setdefault(fname, {"classification": "", "boxes": [], "polygons": []})
        self.annotations[fname]["polygons"].append({"points": img_pts, "label": label})

        # Clear temp drawing state
        for d in self.poly_dots:  self.canvas.delete(d)
        for l in self.poly_lines: self.canvas.delete(l)
        self.polygon_points = []
        self.poly_dots      = []
        self.poly_lines     = []
        self._update_ann_panel(fname)
        self.update_stats()

    def apply_classification(self):
        if not self.image_list:
            return
        fname = self.image_list[self.current_index]
        label = self.selected_label.get()
        self.annotations.setdefault(fname, {"classification": "", "boxes": [], "polygons": []})
        self.annotations[fname]["classification"] = label
        self._update_ann_panel(fname)
        self.update_stats()
        self.status_var.set(f"Classified: {label}")
        self.root.after(2000, lambda: self.status_var.set(""))

    # ── Canvas Drawing Helpers ────────────────────────────────────────────────
    def _draw_box_on_canvas(self, x1, y1, x2, y2, label, canvas_coords=False):
        if not canvas_coords:
            x1 = x1 * self.scale + self.img_offset_x
            y1 = y1 * self.scale + self.img_offset_y
            x2 = x2 * self.scale + self.img_offset_x
            y2 = y2 * self.scale + self.img_offset_y
        color = COLORS.get(label, "#ffffff")
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
        self.canvas.create_rectangle(x1, y1, x1 + len(label)*7 + 8, y1 + 16,
                                     fill=color, outline="")
        self.canvas.create_text(x1 + 4, y1 + 8, anchor="w",
                                text=label.replace("_"," ").title(),
                                fill="white", font=("Segoe UI", 7, "bold"))

    def _draw_polygon_on_canvas(self, points, label):
        color = COLORS.get(label, "#ffffff")
        canvas_pts = [
            (x * self.scale + self.img_offset_x,
             y * self.scale + self.img_offset_y)
            for x, y in points
        ]
        flat = [c for pt in canvas_pts for c in pt]
        self.canvas.create_polygon(flat, outline=color, fill=color, stipple="gray25", width=2)

    # ── Panel Updates ─────────────────────────────────────────────────────────
    def _update_ann_panel(self, fname):
        data = self.annotations.get(fname, {})
        lines = [f"File: {fname}\n"]
        cls = data.get("classification", "")
        if cls:
            lines.append(f"Classification: {cls}\n")
        boxes = data.get("boxes", [])
        if boxes:
            lines.append(f"BBoxes ({len(boxes)}):")
            for i, b in enumerate(boxes):
                lines.append(f"  [{i+1}] {b['label']}  ({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
        polys = data.get("polygons", [])
        if polys:
            lines.append(f"\nPolygons ({len(polys)}):")
            for i, p in enumerate(polys):
                lines.append(f"  [{i+1}] {p['label']}  {len(p['points'])} pts")
        if not cls and not boxes and not polys:
            lines.append("(no annotations yet)")

        self.ann_text.config(state="normal")
        self.ann_text.delete("1.0", "end")
        self.ann_text.insert("end", "\n".join(lines))
        self.ann_text.config(state="disabled")

    def update_stats(self):
        total = len(self.image_list)
        annotated = len(self.annotations)
        label_counts = {l: 0 for l in PAIN_LABELS}
        for data in self.annotations.values():
            cls = data.get("classification", "")
            if cls in label_counts:
                label_counts[cls] += 1
        lines = [
            f"Total images : {total}",
            f"Annotated    : {annotated}",
            f"Remaining    : {total - annotated}",
            "",
            "── Label Distribution ──",
        ]
        for lbl, cnt in label_counts.items():
            bar = "█" * cnt
            lines.append(f"{lbl[:12]:<12} {cnt:>3}  {bar}")

        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("end", "\n".join(lines))
        self.stats_text.config(state="disabled")

    # ── Navigation ────────────────────────────────────────────────────────────
    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def _on_mode_change(self):
        self.polygon_points = []
        for d in self.poly_dots:  self.canvas.delete(d)
        for l in self.poly_lines: self.canvas.delete(l)
        self.poly_dots  = []
        self.poly_lines = []

    def clear_annotations(self):
        if not self.image_list:
            return
        fname = self.image_list[self.current_index]
        if fname in self.annotations:
            if messagebox.askyesno("Clear", f"Clear all annotations for {fname}?"):
                del self.annotations[fname]
                self.load_image()
                self.update_stats()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        from PIL import Image, ImageTk
    except ImportError:
        messagebox.showerror("Missing Library",
                             "Pillow not installed.\nRun: pip install Pillow")
        root.destroy()
        raise SystemExit

    app = AnnotatorApp(root)
    root.mainloop()
