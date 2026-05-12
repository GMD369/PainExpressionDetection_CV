"use strict";

// ── Constants ──────────────────────────────────────────────────────────────────
const PAIN_LABELS = ["no_pain","mild_pain","moderate_pain","severe_pain","extreme_pain"];
const COLORS = {
  no_pain:       "#2ecc71",
  mild_pain:     "#f1c40f",
  moderate_pain: "#e67e22",
  severe_pain:   "#e74c3c",
  extreme_pain:  "#8e44ad",
};

// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  imageList:     [],
  currentIndex:  0,
  annotations:   {},
  currentImage:  null,
  scale:         1.0,
  offsetX:       0,
  offsetY:       0,
  imgW:          0,
  imgH:          0,
  mode:          "Classification",
  label:         "no_pain",
  // bbox drawing
  drawing:       false,
  startX:        0,
  startY:        0,
  mouseX:        0,
  mouseY:        0,
  // polygon drawing
  polyPts:       [],
};

// ── Init ───────────────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  buildLabelRadios();
  initCanvas();
  initKeyboard();
  updateStats();
});

function buildLabelRadios() {
  const g = document.getElementById("label-group");
  PAIN_LABELS.forEach(lbl => {
    const name = lbl.replace(/_/g," ").replace(/\b\w/g, c => c.toUpperCase());
    const label = document.createElement("label");
    label.className = "pain-lbl";
    label.style.color = COLORS[lbl];
    label.innerHTML =
      `<input type="radio" name="label" value="${lbl}" ${lbl===S.label?"checked":""}
              onchange="S.label=this.value"> ${name}`;
    g.appendChild(label);
  });
}

// ── Canvas ─────────────────────────────────────────────────────────────────────
function initCanvas() {
  const canvas = document.getElementById("canvas");
  // sync canvas pixel size to CSS display size
  new ResizeObserver(() => {
    const r = canvas.getBoundingClientRect();
    canvas.width  = Math.floor(r.width);
    canvas.height = Math.floor(r.height);
    if (S.currentImage) recalcLayout();
    redraw();
  }).observe(canvas);

  canvas.addEventListener("mousedown",   onDown);
  canvas.addEventListener("mousemove",   onMove);
  canvas.addEventListener("mouseup",     onUp);
  canvas.addEventListener("dblclick",    onDbl);
  canvas.addEventListener("contextmenu", onRight);
}

function recalcLayout() {
  const c = document.getElementById("canvas");
  const iw = S.currentImage.naturalWidth;
  const ih = S.currentImage.naturalHeight;
  S.scale   = Math.min(c.width / iw, c.height / ih, 1.0);
  S.imgW    = Math.floor(iw * S.scale);
  S.imgH    = Math.floor(ih * S.scale);
  S.offsetX = Math.floor((c.width  - S.imgW) / 2);
  S.offsetY = Math.floor((c.height - S.imgH) / 2);
}

// ── Keyboard ───────────────────────────────────────────────────────────────────
function initKeyboard() {
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft")  prevImage();
    if (e.key === "ArrowRight") nextImage();
    if (e.key === "s" && !e.ctrlKey) saveAnnotations();
    if (e.key === "Delete") clearAnnotations();
  });
}

// ── Folder loading ─────────────────────────────────────────────────────────────
function promptFolder() {
  document.getElementById("folder-path").focus();
  setStatus("Paste the full folder path then press Enter or click Load");
}

async function loadFolder() {
  const path = document.getElementById("folder-path").value.trim();
  if (!path) return;
  try {
    const res  = await fetch("/api/open-folder", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({path}),
    });
    const data = await res.json();
    if (!res.ok) { setStatus("Error: " + data.error, true); return; }

    S.imageList    = data.images;
    S.annotations  = data.annotations || {};
    S.currentIndex = 0;
    setStatus(`Loaded ${data.images.length} images`);
    loadImage(0);
    updateStats();
  } catch {
    setStatus("Could not reach server", true);
  }
}

// ── Image loading ──────────────────────────────────────────────────────────────
function loadImage(index) {
  if (!S.imageList.length) return;
  S.currentIndex = index;
  S.polyPts      = [];
  S.drawing      = false;

  document.getElementById("progress").textContent =
    `${index + 1} / ${S.imageList.length}`;

  const img = new Image();
  img.onload = () => {
    S.currentImage = img;
    recalcLayout();
    redraw();
    document.getElementById("filename-bar").textContent =
      `${S.imageList[index]}  [${index + 1} / ${S.imageList.length}]`;
    updateAnnPanel();
  };
  img.src = `/api/image/${index}?_=${Date.now()}`;
}

// ── Mouse events ───────────────────────────────────────────────────────────────
function onDown(e) {
  e.preventDefault();
  const {x, y} = pos(e);
  if (S.mode === "Detection (BBox)") {
    S.drawing = true;
    S.startX = x; S.startY = y;
    S.mouseX = x; S.mouseY = y;
  } else if (S.mode === "Segmentation (Polygon)") {
    S.polyPts.push([x, y]);
    redraw();
  }
}

function onMove(e) {
  const {x, y} = pos(e);
  S.mouseX = x; S.mouseY = y;
  if (S.drawing || (S.mode === "Segmentation (Polygon)" && S.polyPts.length))
    redraw();
}

function onUp(e) {
  if (S.mode !== "Detection (BBox)" || !S.drawing) return;
  S.drawing = false;
  const {x, y} = pos(e);
  const x1 = Math.min(S.startX, x), y1 = Math.min(S.startY, y);
  const x2 = Math.max(S.startX, x), y2 = Math.max(S.startY, y);
  if (Math.abs(x2-x1) < 5 || Math.abs(y2-y1) < 5) { redraw(); return; }

  const fname = currentFname();
  ensureAnn(fname);
  S.annotations[fname].boxes.push({
    x1: toImgX(x1), y1: toImgY(y1),
    x2: toImgX(x2), y2: toImgY(y2),
    label: S.label,
  });
  redraw();
  updateAnnPanel();
  updateStats();
}

function onDbl(e) {
  if (S.mode === "Segmentation (Polygon)") finishPolygon();
}

function onRight(e) {
  e.preventDefault();
  if (S.mode === "Segmentation (Polygon)" && S.polyPts.length) {
    S.polyPts.pop();
    redraw();
  }
}

// ── Annotation actions ─────────────────────────────────────────────────────────
function applyClassification() {
  if (!S.imageList.length) return;
  const fname = currentFname();
  ensureAnn(fname);
  S.annotations[fname].classification = S.label;
  updateAnnPanel();
  updateStats();
  setStatus("Classified: " + S.label);
  setTimeout(() => setStatus(""), 2000);
}

function finishPolygon() {
  if (S.polyPts.length < 3) {
    alert("Need at least 3 points to close polygon."); return;
  }
  const imgPts = S.polyPts.map(([cx,cy]) => [toImgX(cx), toImgY(cy)]);
  const fname  = currentFname();
  ensureAnn(fname);
  S.annotations[fname].polygons.push({points: imgPts, label: S.label});
  S.polyPts = [];
  redraw();
  updateAnnPanel();
  updateStats();
}

function onModeChange() {
  S.mode    = document.querySelector('input[name="mode"]:checked').value;
  S.polyPts = [];
  S.drawing = false;
  redraw();
}

// ── Navigation ─────────────────────────────────────────────────────────────────
function prevImage() { if (S.currentIndex > 0) loadImage(S.currentIndex - 1); }
function nextImage() { if (S.currentIndex < S.imageList.length-1) loadImage(S.currentIndex + 1); }

// ── Save / Export / Clear ──────────────────────────────────────────────────────
async function saveAnnotations() {
  if (!S.imageList.length) { setStatus("No folder loaded", true); return; }
  try {
    const res = await fetch("/api/annotations", {
      method:  "POST",
      headers: {"Content-Type":"application/json"},
      body:    JSON.stringify({annotations: S.annotations}),
    });
    if (res.ok) { setStatus("Saved → annotations.json"); setTimeout(()=>setStatus(""),3000); }
    else        { setStatus("Save failed", true); }
  } catch { setStatus("Save failed", true); }
}

function exportCSV() { window.location.href = "/api/export-csv"; }

function clearAnnotations() {
  if (!S.imageList.length) return;
  const fname = currentFname();
  if (!S.annotations[fname]) return;
  if (!confirm(`Clear all annotations for ${fname}?`)) return;
  delete S.annotations[fname];
  S.polyPts = [];
  redraw();
  updateAnnPanel();
  updateStats();
}

// ── Canvas rendering ───────────────────────────────────────────────────────────
function redraw() {
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!S.currentImage) return;

  ctx.drawImage(S.currentImage, S.offsetX, S.offsetY, S.imgW, S.imgH);

  const data = S.annotations[currentFname()] || {};
  (data.boxes    || []).forEach(b => drawBox(ctx, b.x1, b.y1, b.x2, b.y2, b.label));
  (data.polygons || []).forEach(p => drawPoly(ctx, p.points, p.label));

  // Live bbox
  if (S.drawing) {
    ctx.strokeStyle = COLORS[S.label];
    ctx.lineWidth   = 2;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(
      Math.min(S.startX, S.mouseX), Math.min(S.startY, S.mouseY),
      Math.abs(S.mouseX - S.startX), Math.abs(S.mouseY - S.startY)
    );
    ctx.setLineDash([]);
  }

  // Live polygon
  if (S.mode === "Segmentation (Polygon)" && S.polyPts.length) {
    const color = COLORS[S.label];
    ctx.setLineDash([]);
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;

    if (S.polyPts.length > 1) {
      ctx.beginPath();
      ctx.moveTo(S.polyPts[0][0], S.polyPts[0][1]);
      S.polyPts.slice(1).forEach(([x,y]) => ctx.lineTo(x, y));
      ctx.stroke();
    }

    // Ghost edge to cursor
    const last = S.polyPts[S.polyPts.length-1];
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color + "99";
    ctx.moveTo(last[0], last[1]);
    ctx.lineTo(S.mouseX, S.mouseY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Vertex dots
    S.polyPts.forEach(([px,py]) => {
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI*2);
      ctx.fillStyle   = color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth   = 1;
      ctx.stroke();
    });
  }
}

function drawBox(ctx, ix1, iy1, ix2, iy2, label) {
  const x1 = toCanX(ix1), y1 = toCanY(iy1);
  const x2 = toCanX(ix2), y2 = toCanY(iy2);
  const c  = COLORS[label] || "#fff";
  ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.setLineDash([]);
  ctx.strokeRect(x1, y1, x2-x1, y2-y1);
  const txt = label.replace(/_/g," ").replace(/\b\w/g,ch=>ch.toUpperCase());
  ctx.font = "bold 11px 'Segoe UI',sans-serif";
  const tw = ctx.measureText(txt).width;
  ctx.fillStyle = c;
  ctx.fillRect(x1, y1, tw + 8, 17);
  ctx.fillStyle = "#fff";
  ctx.fillText(txt, x1 + 4, y1 + 12);
}

function drawPoly(ctx, points, label) {
  if (points.length < 2) return;
  const c   = COLORS[label] || "#fff";
  const pts = points.map(([x,y]) => [toCanX(x), toCanY(y)]);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  pts.slice(1).forEach(([x,y]) => ctx.lineTo(x, y));
  ctx.closePath();
  ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle   = c + "40";   // 25% opacity fill
  ctx.fill();
}

// ── Panel updates ──────────────────────────────────────────────────────────────
function updateAnnPanel() {
  const fname = currentFname();
  const data  = S.annotations[fname] || {};
  const lines = [`File: ${fname || "(none)"}\n`];
  const cls   = data.classification || "";
  if (cls) lines.push(`Classification: ${cls}\n`);
  const boxes = data.boxes || [];
  if (boxes.length) {
    lines.push(`BBoxes (${boxes.length}):`);
    boxes.forEach((b,i) =>
      lines.push(`  [${i+1}] ${b.label}  (${b.x1},${b.y1})-(${b.x2},${b.y2})`)
    );
  }
  const polys = data.polygons || [];
  if (polys.length) {
    lines.push(`\nPolygons (${polys.length}):`);
    polys.forEach((p,i) =>
      lines.push(`  [${i+1}] ${p.label}  ${p.points.length} pts`)
    );
  }
  if (!cls && !boxes.length && !polys.length) lines.push("(no annotations yet)");
  document.getElementById("ann-panel").textContent = lines.join("\n");
}

function updateStats() {
  const total     = S.imageList.length;
  const annotated = Object.keys(S.annotations).length;
  const counts    = Object.fromEntries(PAIN_LABELS.map(l => [l, 0]));
  Object.values(S.annotations).forEach(d => {
    if (d.classification && counts.hasOwnProperty(d.classification))
      counts[d.classification]++;
  });
  const lines = [
    `Total images : ${total}`,
    `Annotated    : ${annotated}`,
    `Remaining    : ${total - annotated}`,
    "",
    "── Label Distribution ──",
  ];
  for (const [lbl, cnt] of Object.entries(counts)) {
    lines.push(`${lbl.substring(0,12).padEnd(12)} ${String(cnt).padStart(3)}  ${"█".repeat(cnt)}`);
  }
  document.getElementById("stats-panel").textContent = lines.join("\n");
}

function setStatus(msg, err=false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.style.color = err ? "#ef4444" : "#94a3b8";
}

// ── Coordinate helpers ─────────────────────────────────────────────────────────
function pos(e) {
  const r = document.getElementById("canvas").getBoundingClientRect();
  return {x: e.clientX - r.left, y: e.clientY - r.top};
}
function toImgX(cx) { return Math.round((cx - S.offsetX) / S.scale); }
function toImgY(cy) { return Math.round((cy - S.offsetY) / S.scale); }
function toCanX(ix) { return ix * S.scale + S.offsetX; }
function toCanY(iy) { return iy * S.scale + S.offsetY; }
function currentFname() { return S.imageList[S.currentIndex] || ""; }
function ensureAnn(fname) {
  if (!S.annotations[fname])
    S.annotations[fname] = {classification:"", boxes:[], polygons:[]};
}
