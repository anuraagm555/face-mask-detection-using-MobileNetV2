from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import CLASS_NAMES_PATH, DEFAULT_CLASS_NAMES, IMG_SIZE, MODEL_PATH


class MaskPredictor:
    """Load trained model and predict mask status from face crops."""

    def __init__(
        self, model_path: Path | str = MODEL_PATH, class_names_path: Path | str | None = None
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run train.py first."
            )

        self.model = tf.keras.models.load_model(self.model_path)
        if class_names_path is None:
            class_names_path = self.model_path.parent / "class_names.json"
        self.class_names = self._load_class_names(Path(class_names_path))
        if self.class_names == DEFAULT_CLASS_NAMES and CLASS_NAMES_PATH != Path(class_names_path):
            self.class_names = self._load_class_names(CLASS_NAMES_PATH)
        if len(self.class_names) != 2:
            raise ValueError(
                f"Expected 2 classes for binary model, got {self.class_names}"
            )
        self.no_mask_index, self.mask_index = self._infer_class_indices(self.class_names)

    @staticmethod
    def _load_class_names(path: Path) -> list[str]:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, list) and len(loaded) == 2:
                    return loaded
        return DEFAULT_CLASS_NAMES

    @staticmethod
    def _infer_class_indices(class_names: list[str]) -> tuple[int, int]:
        lowered = [name.lower() for name in class_names]

        no_mask_tokens = ("without", "no_mask", "nomask", "no-mask")
        no_mask_index = next(
            (
                idx
                for idx, name in enumerate(lowered)
                if any(token in name for token in no_mask_tokens)
            ),
            None,
        )

        if no_mask_index is None:
            mask_index = next(
                (
                    idx
                    for idx, name in enumerate(lowered)
                    if "with" in name and "mask" in name
                ),
                None,
            )
            no_mask_index = 1 - mask_index if mask_index is not None else 1

        mask_index = 1 - no_mask_index
        return no_mask_index, mask_index

    @staticmethod
    def _clip_box(
        x: int, y: int, w: int, h: int, frame_shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        return x, y, w, h

    def _prepare_face(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, IMG_SIZE)
        arr = face_rgb.astype("float32")
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def predict_face(self, face_bgr: np.ndarray) -> tuple[str, float, bool]:
        input_tensor = self._prepare_face(face_bgr)
        class_1_prob = float(self.model.predict(input_tensor, verbose=0)[0][0])
        no_mask_prob = class_1_prob if self.no_mask_index == 1 else (1.0 - class_1_prob)

        if no_mask_prob >= 0.5:
            return self.class_names[self.no_mask_index], no_mask_prob, False
        return self.class_names[self.mask_index], 1.0 - no_mask_prob, True

    def predict_frame(
        self, frame_bgr: np.ndarray, face_boxes: list[tuple[int, int, int, int]]
    ) -> list[dict[str, object]]:
        predictions: list[dict[str, object]] = []

        for x, y, w, h in face_boxes:
            x, y, w, h = self._clip_box(x, y, w, h, frame_bgr.shape)
            face = frame_bgr[y : y + h, x : x + w]
            if face.size == 0:
                continue

            label, confidence, is_mask = self.predict_face(face)
            predictions.append(
                {
                    "box": [x, y, w, h],
                    "label": label,
                    "confidence": confidence,
                    "is_mask": is_mask,
                }
            )

        return predictions
