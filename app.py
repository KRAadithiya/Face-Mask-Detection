"""
Face Mask Detection - Flask Web Dashboard
"""

import os
import base64
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from flask import Flask, render_template, request, jsonify, Response

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SRC_DIR)
if not os.path.isdir(os.path.join(BASE_DIR, "templates")):
    BASE_DIR = SRC_DIR

FACE_PROTO = os.path.join(BASE_DIR, "models", "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL = os.path.join(BASE_DIR, "models", "mask_detector.h5")
TMPL_DIR   = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CONFIDENCE  = 0.5
ALLOWED_IMG = {"jpg", "jpeg", "png", "bmp", "webp"}

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TMPL_DIR, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

face_net     = None
mask_net     = None
models_error = None
frame_stats  = {"mask": 0, "no_mask": 0, "total_frames": 0}

# ── Analytics State ───────────────────────────────────────────────────────────
# Each entry: {timestamp, time_label, mask, no_mask, total, compliance}
analytics_log = deque(maxlen=500)

# Webcam dedup: only log when face count or label changes (not every frame)
_last_webcam_signature = None


def load_models_once():
    global face_net, mask_net, models_error
    if face_net is not None:
        return True
    if models_error is not None:
        return False

    missing = []
    for path, label in [
        (FACE_PROTO, "deploy.prototxt"),
        (FACE_MODEL, "res10_300x300_ssd_iter_140000.caffemodel"),
        (MASK_MODEL, "mask_detector.h5"),
    ]:
        if not os.path.isfile(path):
            missing.append(f"{label}  →  {path}")

    if missing:
        models_error = "Missing model files:\n" + "\n".join(missing)
        print(f"[ERROR] {models_error}")
        return False

    try:
        print("[INFO] Loading face detector...")
        face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
        print("[INFO] Loading mask classifier...")
        from tensorflow.keras.models import load_model as _load
        mask_net = _load(MASK_MODEL)
        print("[INFO] All models loaded successfully.")
        return True
    except Exception as e:
        models_error = str(e)
        face_net = mask_net = None
        return False


def preprocess_face(face_bgr):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    return face


def detect(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs = [], []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONFIDENCE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        sX, sY, eX, eY = box.astype("int")
        sX, sY = max(0, sX), max(0, sY)
        eX, eY = min(w - 1, eX), min(h - 1, eY)
        crop = frame[sY:eY, sX:eX]
        if crop.size == 0:
            continue
        faces.append(preprocess_face(crop))
        locs.append((sX, sY, eX, eY))

    results = []
    if faces:
        preds = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)
        for (sX, sY, eX, eY), pred in zip(locs, preds):
            mask_p, nm = float(pred[0]), float(pred[1])
            label = "Mask" if mask_p > nm else "No Mask"
            color = (0, 200, 100) if label == "Mask" else (0, 60, 255)
            cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
            cv2.rectangle(frame, (sX, max(0, sY - 30)), (eX, sY), color, -1)
            cv2.putText(
                frame, f"{label}: {max(mask_p, nm)*100:.1f}%",
                (sX + 6, max(6, sY - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            results.append({
                "label":        label,
                "mask_prob":    round(mask_p * 100, 1),
                "no_mask_prob": round(nm * 100, 1),
                "box":          [int(sX), int(sY), int(eX), int(eY)],
            })
    return frame, results


def frame_to_b64(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def record_compliance(results, source="image"):
    """
    Log one compliance snapshot.
    For webcam: only logs when the detection result actually changes
    (e.g. a new face appears or mask status changes) — prevents duplicate counts.
    For image: always logs once per upload.
    """
    global _last_webcam_signature
    if not results:
        return

    mask_count   = sum(1 for r in results if r["label"] == "Mask")
    nomask_count = sum(1 for r in results if r["label"] == "No Mask")
    total        = mask_count + nomask_count
    if total == 0:
        return

    # For webcam frames: skip if same result as last frame (dedup)
    if source == "webcam":
        sig = (mask_count, nomask_count)
        if sig == _last_webcam_signature:
            return
        _last_webcam_signature = sig

    compliance_pct = round(mask_count / total * 100, 1)
    now = datetime.now()
    analytics_log.append({
        "timestamp":  now.strftime("%Y-%m-%d %H:%M:%S"),
        "time_label": now.strftime("%I:%M:%S %p"),   # e.g. "02:45:30 PM"
        "date":       now.strftime("%Y-%m-%d"),
        "mask":       mask_count,
        "no_mask":    nomask_count,
        "total":      total,
        "compliance": compliance_pct,
    })


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    if not load_models_once():
        return jsonify({"error": f"Models not loaded. {models_error}"}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_IMG:
        return jsonify({"error": f"Unsupported format '.{ext}'"}), 400
    buf   = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400
    annotated, results = detect(frame)
    record_compliance(results, source="image")   # log once per image upload
    return jsonify({
        "image":         frame_to_b64(annotated),
        "results":       results,
        "count":         len(results),
        "mask_count":    sum(1 for r in results if r["label"] == "Mask"),
        "no_mask_count": sum(1 for r in results if r["label"] == "No Mask"),
    })


@app.route("/api/webcam/frame", methods=["POST"])
def api_webcam_frame():
    if not load_models_once():
        return jsonify({"error": models_error}), 503
    data = (request.json or {}).get("frame", "")
    if not data:
        return jsonify({"error": "No frame data"}), 400
    try:
        _, encoded = data.split(",", 1)
        buf   = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode frame")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    annotated, results = detect(frame)
    frame_stats["total_frames"] += 1
    frame_stats["mask"]    += sum(1 for r in results if r["label"] == "Mask")
    frame_stats["no_mask"] += sum(1 for r in results if r["label"] == "No Mask")
    record_compliance(results, source="webcam")  # only logs on change
    return jsonify({
        "image":   frame_to_b64(annotated),
        "results": results,
        "stats":   frame_stats,
    })


@app.route("/api/stats/reset", methods=["POST"])
def reset_stats():
    frame_stats.update({"mask": 0, "no_mask": 0, "total_frames": 0})
    return jsonify({"status": "reset"})


@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok",
        "models_ready": face_net is not None,
        "models_error": models_error,
        "base_dir":     BASE_DIR,
        "templates":    TMPL_DIR,
    })


# ── Analytics Routes ──────────────────────────────────────────────────────────
@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics/data")
def analytics_data():
    log = list(analytics_log)
    if not log:
        return jsonify({"log": [], "summary": {}, "hourly": []})

    compliances  = [e["compliance"] for e in log]
    total_mask   = sum(e["mask"]    for e in log)
    total_nomask = sum(e["no_mask"] for e in log)
    total_faces  = total_mask + total_nomask

    # Hourly aggregation — fall back to per-minute if all data within same hour
    unique_hours = set(e["timestamp"][:13] for e in log)
    use_minutes  = len(unique_hours) <= 1

    hourly = {}
    for e in log:
        key = e["timestamp"][:16] if use_minutes else e["timestamp"][:13]
        if key not in hourly:
            hourly[key] = {"mask": 0, "no_mask": 0}
        hourly[key]["mask"]    += e["mask"]
        hourly[key]["no_mask"] += e["no_mask"]

    hourly_list = []
    for key, d in sorted(hourly.items()):
        t = d["mask"] + d["no_mask"]
        try:
            fmt   = "%Y-%m-%d %H:%M" if use_minutes else "%Y-%m-%d %H"
            dt    = datetime.strptime(key, fmt)
            label = dt.strftime("%I:%M %p").lstrip("0")
        except Exception:
            label = key[11:]
        hourly_list.append({
            "hour":       label,
            "compliance": round(d["mask"] / t * 100, 1) if t else 0,
            "total":      t,
            "type":       "minute" if use_minutes else "hour",
        })

    return jsonify({
        "log": log[-100:],
        "summary": {
            "avg_compliance":  round(sum(compliances) / len(compliances), 1),
            "min_compliance":  min(compliances),
            "max_compliance":  max(compliances),
            "total_faces":     total_faces,
            "total_mask":      total_mask,
            "total_no_mask":   total_nomask,
            "total_readings":  len(log),
        },
        "hourly": hourly_list,
    })


@app.route("/api/analytics/reset", methods=["POST"])
def analytics_reset():
    global _last_webcam_signature
    analytics_log.clear()
    _last_webcam_signature = None
    return jsonify({"status": "cleared"})


@app.route("/api/analytics/export")
def analytics_export():
    import io
    if not analytics_log:
        return "No data yet", 204
    out = io.StringIO()
    out.write("Timestamp,Time,Mask,No Mask,Total,Compliance %\n")
    for e in analytics_log:
        out.write(f"{e['timestamp']},{e['time_label']},{e['mask']},"
                  f"{e['no_mask']},{e['total']},{e['compliance']}\n")
    return Response(
        out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=compliance_report.csv"}
    )


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Face Mask Detection Dashboard")
    print("=" * 55)
    print(f"  Base dir  : {BASE_DIR}")
    for path, label in [(FACE_PROTO, "deploy.prototxt"),
                        (FACE_MODEL, "caffemodel"),
                        (MASK_MODEL, "mask_detector.h5")]:
        status = "OK     " if os.path.isfile(path) else "MISSING"
        print(f"  [{status}]  {label}")
    print("=" * 55)
    print("  Main:      http://127.0.0.1:5000")
    print("  Analytics: http://127.0.0.1:5000/analytics")
    print("  Export:    http://127.0.0.1:5000/api/analytics/export")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)