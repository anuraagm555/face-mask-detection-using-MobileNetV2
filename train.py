from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE,
    METRICS_DIR,
    MODEL_PATH,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face mask detector with MobileNetV2")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Dataset root folder")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Validation split ratio (used as test set)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--model_out",
        type=Path,
        default=MODEL_PATH,
        help="Output path for trained model",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=METRICS_DIR,
        help="Directory to save metrics and confusion matrix",
    )
    return parser.parse_args()


def load_datasets(
    data_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names

    def preprocess(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.cast(images, tf.float32)
        return preprocess_input(images), labels

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    metrics_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(metrics_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def evaluate(
    model: tf.keras.Model, val_ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    y_true = []
    y_scores = []

    for batch_images, batch_labels in val_ds:
        probs = model.predict(batch_images, verbose=0).ravel()
        y_scores.extend(probs.tolist())
        y_true.extend(batch_labels.numpy().astype(int).ravel().tolist())

    y_true_arr = np.array(y_true, dtype=np.int32)
    y_scores_arr = np.array(y_scores, dtype=np.float32)
    y_pred_arr = (y_scores_arr >= 0.5).astype(np.int32)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    return y_true_arr, y_pred_arr, acc, prec, rec


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_dir}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_ds, val_ds, class_names = load_datasets(
        data_dir=args.data_dir,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    print(f"Detected classes: {class_names}")
    class_names_path = args.model_out.parent / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)

    model = build_model(input_shape=(*IMG_SIZE, 3))
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating model...")
    y_true, y_pred, acc, prec, rec = evaluate(model, val_ds)

    save_training_curves(history, args.metrics_dir)
    save_confusion_matrix(y_true, y_pred, class_names, args.metrics_dir)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "num_validation_samples": int(len(y_true)),
        "class_names": class_names,
        "model_path": str(args.model_out),
    }

    metrics_path = args.metrics_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {args.model_out}")
    print(f"Saved class names to: {class_names_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {args.metrics_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
