# Face Mask Detection System (MobileNetV2 + OpenCV + Flask)

This project detects whether a detected face is wearing a mask (`with_mask`) or not (`without_mask`) from webcam frames using a CNN classifier (MobileNetV2 transfer learning).

## Project Features
- Transfer learning with **MobileNetV2** on 224x224 RGB images
- **80/20 split** for training and validation (used as test set)
- Metrics: **accuracy, precision, recall, confusion matrix**
- Real-time detection via **OpenCV** + Haar Cascade face detector
- Live web app using **Flask** + browser webcam stream

## Folder Structure

```text
Face-Mask-Detection/
├── app.py
├── train.py
├── realtime_detection.py
├── requirements.txt
├── data/
│   ├── with_mask/
│   └── without_mask/
├── src/
│   ├── config.py
│   ├── face_detector.py
│   └── predictor.py
├── templates/
│   └── index.html
├── static/
│   ├── css/styles.css
│   └── js/app.js
└── outputs/
    ├── models/
    └── metrics/
```

## 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset Format

Your dataset should be:

```text
data/
  with_mask/
    img1.jpg
    img2.jpg
  without_mask/
    img1.jpg
    img2.jpg
```

## 3. Train the Model

```bash
python train.py --epochs 12 --batch_size 32
```

Artifacts generated:
- Model: `outputs/models/mask_detector_mobilenetv2.keras`
- Class names: `outputs/models/class_names.json`
- Metrics JSON: `outputs/metrics/metrics.json`
- Confusion matrix: `outputs/metrics/confusion_matrix.png`
- Training curves: `outputs/metrics/training_curves.png`

## 4. Real-Time Webcam Detection (OpenCV Window)

```bash
python realtime_detection.py
```

Press `q` to exit.

## 5. Run as Live Web App (Flask)

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser and allow camera access.

## Notes
- The classifier is built with MobileNetV2 base layers frozen and a custom head trained on your dataset.
- Typical accuracy can be high on this dataset, but exact numbers depend on training epochs, augmentation, and camera conditions.
- If you want higher robustness, switch face detection from Haar Cascade to an OpenCV DNN detector and add stronger augmentation.
