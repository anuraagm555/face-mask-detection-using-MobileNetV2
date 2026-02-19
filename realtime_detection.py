from __future__ import annotations

import argparse

import cv2

from src.config import MODEL_PATH
from src.face_detector import HaarFaceDetector
from src.predictor import MaskPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face mask detection with webcam")
    parser.add_argument("--model_path", default=str(MODEL_PATH), help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    return parser.parse_args()


def draw_prediction(frame, prediction):
    x, y, w, h = prediction["box"]
    label = prediction["label"]
    confidence = float(prediction["confidence"])
    is_mask = bool(prediction.get("is_mask", label == "with_mask"))
    color = (46, 204, 113) if is_mask else (0, 0, 255)
    text = f"{'Mask' if is_mask else 'No Mask'}: {confidence:.2f}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        text,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    predictor = MaskPredictor(args.model_path)
    face_detector = HaarFaceDetector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions and index.")

    print("Starting webcam detection. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        faces = face_detector.detect(frame)
        predictions = predictor.predict_frame(frame, faces)

        for pred in predictions:
            draw_prediction(frame, pred)

        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
