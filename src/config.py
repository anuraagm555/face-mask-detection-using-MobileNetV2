from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODEL_PATH = MODEL_DIR / "mask_detector_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Fallback class names if metadata file is missing.
DEFAULT_CLASS_NAMES = ["with_mask", "without_mask"]
