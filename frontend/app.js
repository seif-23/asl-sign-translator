const API_BASE = "http://127.0.0.1:8000";

const statusSpan = document.getElementById("backend-status");
const statusPill = document.getElementById("backend-status-pill");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("video-input");
const fileInfo = document.getElementById("file-info");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const loader = document.getElementById("loader");
const resultDiv = document.getElementById("result");
const videoPreview = document.getElementById("video-preview");

let selectedFile = null;

// ===== Backend status =====
async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/ping`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    if (data.status === "ok") {
      statusSpan.textContent = "Online";
      statusPill.classList.add("online");
    } else {
      statusSpan.textContent = "Unexpected";
      statusPill.classList.remove("online");
    }
  } catch (err) {
    console.error(err);
    statusSpan.textContent = "Offline";
    statusPill.classList.remove("online");
  }
}

checkBackend();

// ===== Helpers =====
function updateFileInfo() {
  if (!selectedFile) {
    fileInfo.textContent = "No file selected yet.";
    predictBtn.disabled = true;
    videoPreview.style.display = "none";
    videoPreview.src = "";
    return;
  }

  fileInfo.textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)`;
  predictBtn.disabled = false;

  const url = URL.createObjectURL(selectedFile);
  videoPreview.src = url;
  videoPreview.style.display = "block";
}

function setLoading(isLoading) {
  if (isLoading) {
    loader.style.display = "flex";
    predictBtn.disabled = true;
    resultDiv.style.display = "none";
  } else {
    loader.style.display = "none";
    if (selectedFile) predictBtn.disabled = false;
  }
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ===== Drop zone events =====
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("active");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("active");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("active");
  if (!e.dataTransfer.files || !e.dataTransfer.files.length) return;

  const file = e.dataTransfer.files[0];
  if (!file.type.startsWith("video/")) {
    alert("Please drop a video file.");
    return;
  }
  selectedFile = file;
  updateFileInfo();
});

fileInput.addEventListener("change", (e) => {
  if (!e.target.files.length) return;
  const file = e.target.files[0];
  if (!file.type.startsWith("video/")) {
    alert("Please select a video file.");
    fileInput.value = "";
    return;
  }
  selectedFile = file;
  updateFileInfo();
});

// ===== Predict button =====
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  setLoading(true);
  resultDiv.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json().catch(() => ({}));
    console.log("API Response:", data);

    if (!res.ok) {
      const errText =
        data.detail
          ? typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail)
          : data.error
          ? data.error
          : `HTTP error: ${res.status}`;

      resultDiv.textContent = `Error: ${errText}`;
      resultDiv.style.display = "block";
      return;
    }

    const pred = escapeHtml(data.prediction ?? "N/A");
    const confPercent = data.confidence != null ? (data.confidence * 100).toFixed(1) : "N/A";

    // Top-5 UI
    let top5Html = "";
    if (Array.isArray(data.top5) && data.top5.length) {
      top5Html = `
        <div style="margin-top:12px;">
          <strong>Top-5:</strong>
          <ol style="margin:8px 0 0 18px;">
            ${data.top5
              .map((p) => {
                const label = escapeHtml(p.label ?? "N/A");
                const pConf = p.confidence != null ? (p.confidence * 100).toFixed(1) : "N/A";
                return `<li>${label} <span style="opacity:.8">(${pConf}%)</span></li>`;
              })
              .join("")}
          </ol>
        </div>
      `;
    } else {
      top5Html = `<div style="margin-top:12px; opacity:.8;">No top5 returned.</div>`;
    }

    resultDiv.innerHTML = `
      <div><strong>Prediction:</strong> ${pred}</div>
      <div><strong>Confidence:</strong> ${confPercent}%</div>
      ${top5Html}
    `;
    resultDiv.style.display = "block";
  } catch (err) {
    console.error(err);
    resultDiv.textContent = `Network error: ${err.message}`;
    resultDiv.style.display = "block";
  } finally {
    setLoading(false);
  }
});

// ===== Clear button =====
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  updateFileInfo();
  resultDiv.style.display = "none";
});

// Smooth scroll for navbar links (optional)
document.querySelectorAll(".nav-links a[href^='#']").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const targetId = link.getAttribute("href").slice(1);
    const target = document.getElementById(targetId);
    if (target) {
      window.scrollTo({
        top: target.offsetTop - 80,
        behavior: "smooth",
      });
    }
  });
});

// =======================
// Webcam → Record → Predict
// =======================
const camVideo = document.getElementById("cam");
const camStartBtn = document.getElementById("cam-start");
const recStartBtn = document.getElementById("rec-start");
const recStopBtn  = document.getElementById("rec-stop");
const camStatus   = document.getElementById("cam-status");

let camStream = null;
let mediaRecorder = null;
let recordedChunks = [];

function setCamStatus(txt) {
  camStatus.textContent = `Camera: ${txt}`;
}

function pickBestMimeType() {
  // Prefer mp4 if available, else webm (most common)
  const candidates = [
    "video/mp4;codecs=avc1",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const t of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

async function startCamera() {
  try {
    setCamStatus("requesting permission...");
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    camVideo.srcObject = camStream;

    recStartBtn.disabled = false;
    setCamStatus("ready ✅");
  } catch (err) {
    console.error(err);
    setCamStatus("permission denied / not available ❌");
    alert("Camera access failed. Check browser permissions.");
  }
}

function stopCamera() {
  if (!camStream) return;
  camStream.getTracks().forEach((t) => t.stop());
  camStream = null;
  camVideo.srcObject = null;
  recStartBtn.disabled = true;
  recStopBtn.disabled = true;
  setCamStatus("stopped");
}

camStartBtn.addEventListener("click", async () => {
  if (!camStream) {
    await startCamera();
    camStartBtn.textContent = "Stop Camera";
  } else {
    stopCamera();
    camStartBtn.textContent = "Start Camera";
  }
});

async function sendBlobToPredict(blob, filename) {
  const formData = new FormData();
  formData.append("file", blob, filename);

  setLoading(true);
  resultDiv.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json().catch(() => ({}));
    console.log("API Response:", data);

    if (!res.ok) {
      const errText =
        data.detail
          ? typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail)
          : data.error
          ? data.error
          : `HTTP error: ${res.status}`;

      resultDiv.textContent = `Error: ${errText}`;
      resultDiv.style.display = "block";
      return;
    }

    const pred = escapeHtml(data.prediction ?? "N/A");
    const confPercent = data.confidence != null ? (data.confidence * 100).toFixed(1) : "N/A";

    let top5Html = "";
    if (Array.isArray(data.top5) && data.top5.length) {
      top5Html = `
        <div style="margin-top:12px;">
          <strong>Top-5:</strong>
          <ol style="margin:8px 0 0 18px;">
            ${data.top5
              .map((p) => {
                const label = escapeHtml(p.label ?? "N/A");
                const pConf = p.confidence != null ? (p.confidence * 100).toFixed(1) : "N/A";
                return `<li>${label} <span style="opacity:.8">(${pConf}%)</span></li>`;
              })
              .join("")}
          </ol>
        </div>
      `;
    } else {
      top5Html = `<div style="margin-top:12px; opacity:.8;">No top5 returned.</div>`;
    }

    resultDiv.innerHTML = `
      <div><strong>Prediction:</strong> ${pred}</div>
      <div><strong>Confidence:</strong> ${confPercent}%</div>
      ${top5Html}
    `;
    resultDiv.style.display = "block";
  } catch (err) {
    console.error(err);
    resultDiv.textContent = `Network error: ${err.message}`;
    resultDiv.style.display = "block";
  } finally {
    setLoading(false);
  }
}

function startRecording() {
  if (!camStream) return;

  recordedChunks = [];
  const mimeType = pickBestMimeType();

  try {
    mediaRecorder = new MediaRecorder(camStream, mimeType ? { mimeType } : undefined);
  } catch (e) {
    console.error(e);
    alert("MediaRecorder failed to start (browser may not support chosen format).");
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstart = () => {
    recStartBtn.disabled = true;
    recStopBtn.disabled = false;
    setCamStatus("recording... ⏺");
  };

  mediaRecorder.onstop = async () => {
    recStartBtn.disabled = false;
    recStopBtn.disabled = true;
    setCamStatus("sending to model... 🚀");

    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" });
    const ext = (mediaRecorder.mimeType || "video/webm").includes("mp4") ? "mp4" : "webm";
    await sendBlobToPredict(blob, `webcam_clip.${ext}`);

    setCamStatus("ready ✅");
  };

  // Start + record a short clip automatically (2.5s) — تقدر تغيّرها
  mediaRecorder.start();
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  }, 2500);
}

function stopRecordingManual() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

recStartBtn.addEventListener("click", startRecording);
recStopBtn.addEventListener("click", stopRecordingManual);

