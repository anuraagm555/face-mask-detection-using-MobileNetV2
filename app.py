from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.config import MODEL_PATH
from src.face_detector import HaarFaceDetector
from src.predictor import MaskPredictor

app = Flask(__name__)

predictor: MaskPredictor | None = None
face_detector: HaarFaceDetector | None = None
startup_error: str | None = None

try:
    predictor = MaskPredictor(MODEL_PATH)
    face_detector = HaarFaceDetector()
except Exception as exc:  # noqa: BLE001 - surface startup failures via API/UI
    startup_error = str(exc)


def decode_data_url_to_bgr(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Invalid image payload")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image bytes")
    return frame


@app.route("/")
def index():
    return render_template("index.html", startup_error=startup_error)


@app.route("/health")
def health() -> tuple[Any, int]:
    if startup_error:
        return jsonify({"status": "error", "message": startup_error}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int]:
    if startup_error or predictor is None or face_detector is None:
        return (
            jsonify(
                {
                    "error": "Model startup failed.",
                    "details": startup_error or "Unknown startup error.",
                }
            ),
            500,
        )

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing 'image' field in JSON payload."}), 400

    try:
        frame = decode_data_url_to_bgr(image_data)
        faces = face_detector.detect(frame)
        predictions = predictor.predict_frame(frame, faces)

        frame_h, frame_w = frame.shape[:2]
        response_detections = []
        for pred in predictions:
            x, y, w, h = pred["box"]
            response_detections.append(
                {
                    "box": {
                        "x": x / frame_w,
                        "y": y / frame_h,
                        "w": w / frame_w,
                        "h": h / frame_h,
                    },
                    "label": pred["label"],
                    "confidence": float(pred["confidence"]),
                    "is_mask": bool(pred.get("is_mask", pred["label"] == "with_mask")),
                }
            )

        return jsonify({"detections": response_detections}), 200

    except Exception as exc:  # noqa: BLE001 - keep API robust for client-side polling
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
