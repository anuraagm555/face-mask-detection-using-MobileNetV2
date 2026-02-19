const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const statusNode = document.getElementById("status");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let detections = [];
let inFlight = false;

function setStatus(text) {
  statusNode.textContent = text;
}

function resizeOverlay() {
  if (!video.videoWidth || !video.videoHeight) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function drawDetections() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  detections.forEach((det) => {
    const { box, label, confidence } = det;
    const x = box.x * overlay.width;
    const y = box.y * overlay.height;
    const w = box.w * overlay.width;
    const h = box.h * overlay.height;

    const isMask = Boolean(det.is_mask ?? (label === "with_mask"));
    const color = isMask ? "#2ecc71" : "#ff4d4d";
    const tag = `${isMask ? "Mask" : "No Mask"} ${confidence.toFixed(2)}`;

    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeRect(x, y, w, h);

    overlayCtx.fillStyle = color;
    overlayCtx.font = "16px 'IBM Plex Mono', monospace";
    const textWidth = overlayCtx.measureText(tag).width;
    overlayCtx.fillRect(x, Math.max(0, y - 24), textWidth + 14, 22);

    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(tag, x + 7, Math.max(16, y - 8));
  });
}

async function sendFrameForPrediction() {
  if (inFlight || video.readyState < 2) return;
  inFlight = true;

  try {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const image = captureCanvas.toDataURL("image/jpeg", 0.82);
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Prediction failed");
    }

    const data = await response.json();
    detections = data.detections || [];
    setStatus(`Live · Faces: ${detections.length}`);
  } catch (error) {
    setStatus(`Error · ${error.message}`);
  } finally {
    inFlight = false;
  }
}

function animationLoop() {
  drawDetections();
  requestAnimationFrame(animationLoop);
}

async function start() {
  try {
    if (!window.isSecureContext) {
      throw new Error(
        "Camera requires HTTPS or localhost. Open via https://<host>:5000 or http://127.0.0.1:5000."
      );
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "getUserMedia is unavailable in this browser/context. Use a modern browser over HTTPS."
      );
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    resizeOverlay();
    window.addEventListener("resize", resizeOverlay);
    setStatus("Camera connected");

    setInterval(sendFrameForPrediction, 350);
    requestAnimationFrame(animationLoop);
  } catch (error) {
    setStatus(`Camera error · ${error.message}`);
  }
}

start();
