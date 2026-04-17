"""
Face Mask Detection - Real-time Detection
Supports: webcam stream | image file | video file
Usage:
  python src/detect_mask.py --mode webcam
  python src/detect_mask.py --mode image  --input "C:/Users/You/Downloads/photo.jpg"
  python src/detect_mask.py --mode video  --input "C:/Users/You/Videos/clip.mp4"
"""

import argparse
import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ─── Resolve model paths relative to THIS file (not CWD) ─────────────────────
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SRC_DIR)

FACE_PROTO   = os.path.join(BASE_DIR, "models", "deploy.prototxt")
FACE_WEIGHTS = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL   = os.path.join(BASE_DIR, "models", "mask_detector.h5")

CONFIDENCE_THRESHOLD = 0.5
LABEL_COLORS = {
    "Mask":    (0, 200, 100),
    "No Mask": (0,  60, 255),
}


def load_models():
    for path, name in [(FACE_PROTO,   "deploy.prototxt"),
                       (FACE_WEIGHTS, "res10_300x300_ssd_iter_140000.caffemodel"),
                       (MASK_MODEL,   "mask_detector.h5")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[ERROR] Model file not found: {name}\n"
                f"        Expected at: {path}"
            )
    print(f"[INFO] Loading face detector...")
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)
    print(f"[INFO] Loading mask classifier...")
    mask_net = load_model(MASK_MODEL)
    print("[INFO] Models loaded successfully.")
    return face_net, mask_net


def detect_and_predict_mask(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob   = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs = [], []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (sX, sY, eX, eY) = box.astype("int")
        sX, sY = max(0, sX), max(0, sY)
        eX, eY = min(w - 1, eX), min(h - 1, eY)
        face = frame[sY:eY, sX:eX]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(img_to_array(face))
        faces.append(face)
        locs.append((sX, sY, eX, eY))

    preds = []
    if faces:
        preds = mask_net.predict(np.array(faces, dtype="float32"), batch_size=32, verbose=0)
    return locs, preds


def annotate_frame(frame, locs, preds):
    for (box, pred) in zip(locs, preds):
        (sX, sY, eX, eY) = box
        (mask, no_mask)   = pred
        label = "Mask" if mask > no_mask else "No Mask"
        conf  = max(mask, no_mask) * 100
        color = LABEL_COLORS[label]
        text  = f"{label}: {conf:.1f}%"
        cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
        cv2.rectangle(frame, (sX, max(0, sY - 30)), (eX, sY), color, -1)
        cv2.putText(frame, text, (sX + 6, max(6, sY - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def process_webcam(face_net, mask_net):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] Cannot open webcam.")
    print("[INFO] Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        locs, preds = detect_and_predict_mask(frame, face_net, mask_net)
        frame       = annotate_frame(frame, locs, preds)
        cv2.imshow("Face Mask Detector  [press Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path, face_net, mask_net, output_path=None):
    image_path = os.path.abspath(image_path)
    print(f"[INFO] Loading image: {image_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"[ERROR] Image not found: {image_path}\n"
            f"        Use the full path, e.g.:\n"
            f'        --input "C:/Users/YourName/Downloads/photo.jpg"'
        )
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"[ERROR] OpenCV could not read: {image_path}\n"
                         f"        Make sure the file is a valid JPG/PNG.")
    print(f"[INFO] Image size: {frame.shape[1]}x{frame.shape[0]}")
    locs, preds = detect_and_predict_mask(frame, face_net, mask_net)
    frame       = annotate_frame(frame, locs, preds)
    print(f"[INFO] Detected {len(locs)} face(s).")

    if output_path:
        out = os.path.abspath(output_path)
    else:
        base, ext = os.path.splitext(image_path)
        out = base + "_result" + (ext if ext else ".jpg")

    cv2.imwrite(out, frame)
    print(f"[INFO] Result saved to: {out}")
    cv2.imshow("Result  [press any key to close]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, face_net, mask_net, output_path=None):
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"[ERROR] Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path:
        out_path = os.path.abspath(output_path)
    else:
        base, _ = os.path.splitext(video_path)
        out_path = base + "_result.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[INFO] Processing video → {out_path}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        locs, preds = detect_and_predict_mask(frame, face_net, mask_net)
        frame       = annotate_frame(frame, locs, preds)
        writer.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")
    cap.release()
    writer.release()
    print(f"[INFO] Done. {frame_count} frames saved to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Face Mask Detector",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python src/detect_mask.py --mode webcam
  python src/detect_mask.py --mode image --input photo.jpg
  python src/detect_mask.py --mode image --input "C:/Users/You/Downloads/photo.jpg"
  python src/detect_mask.py --mode video --input clip.mp4
        """
    )
    ap.add_argument("--mode",   choices=["webcam", "image", "video"], default="webcam")
    ap.add_argument("--input",  default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        face_net, mask_net = load_models()
    except FileNotFoundError as e:
        print(e); exit(1)

    try:
        if args.mode == "webcam":
            process_webcam(face_net, mask_net)
        elif args.mode == "image":
            if not args.input:
                ap.error('--input required. Example: --input "C:/Users/You/Downloads/photo.jpg"')
            process_image(args.input, face_net, mask_net, args.output)
        elif args.mode == "video":
            if not args.input:
                ap.error("--input required for video mode.")
            process_video(args.input, face_net, mask_net, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(e); exit(1)