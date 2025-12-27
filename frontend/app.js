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

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setLoading(isLoading) {
  loader.style.display = isLoading ? "flex" : "none";
  resultDiv.style.display = isLoading ? "none" : resultDiv.style.display;

  if (isLoading) {
    predictBtn.disabled = true;
  } else {
    predictBtn.disabled = !selectedFile;
  }
}

function updateFileUI() {
  if (!selectedFile) {
    fileInfo.textContent = "No file selected yet.";
    predictBtn.disabled = true;
    videoPreview.style.display = "none";
    videoPreview.src = "";
    return;
  }

  fileInfo.textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(
    2
  )} MB)`;
  predictBtn.disabled = false;

  const url = URL.createObjectURL(selectedFile);
  videoPreview.src = url;
  videoPreview.style.display = "block";
}

function renderResult(data) {
  const pred = escapeHtml(data.prediction ?? "N/A");
  const confPercent = data.confidence != null ? (data.confidence * 100).toFixed(1) : "N/A";

  let top5Html = `<div style="margin-top:12px; opacity:.8;">No top5 returned.</div>`;
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
  }

  resultDiv.innerHTML = `
    <div><strong>Prediction:</strong> ${pred}</div>
    <div><strong>Confidence:</strong> ${confPercent}%</div>
    ${top5Html}
  `;
  resultDiv.style.display = "block";
}

function renderError(data, resStatus) {
  const errText =
    data?.detail
      ? typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail)
      : data?.error
      ? data.error
      : `HTTP error: ${resStatus}`;

  resultDiv.textContent = `Error: ${errText}`;
  resultDiv.style.display = "block";
}

async function predictFile(fileOrBlob, filenameOverride) {
  const formData = new FormData();
  if (filenameOverride) {
    formData.append("file", fileOrBlob, filenameOverride);
  } else {
    formData.append("file", fileOrBlob);
  }

  setLoading(true);
  resultDiv.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    console.log("API Response:", data);

    if (!res.ok) return renderError(data, res.status);
    renderResult(data);
  } catch (err) {
    console.error(err);
    resultDiv.textContent = `Network error: ${err.message}`;
    resultDiv.style.display = "block";
  } finally {
    setLoading(false);
  }
}

function setSelectedFile(file) {
  selectedFile = file;
  updateFileUI();
}

function validateVideoFile(file) {
  return file && file.type && file.type.startsWith("video/");
}

/* Drop zone */
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

  const file = e.dataTransfer?.files?.[0];
  if (!file) return;

  if (!validateVideoFile(file)) {
    alert("Please drop a video file.");
    return;
  }
  setSelectedFile(file);
});

fileInput.addEventListener("change", (e) => {
  const file = e.target?.files?.[0];
  if (!file) return;

  if (!validateVideoFile(file)) {
    alert("Please select a video file.");
    fileInput.value = "";
    return;
  }
  setSelectedFile(file);
});

/* Predict */
predictBtn.addEventListener("click", () => {
  if (!selectedFile) return;
  predictFile(selectedFile);
});

/* Clear */
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  updateFileUI();
  resultDiv.style.display = "none";
});

/* Smooth scroll */
document.querySelectorAll(".nav-links a[href^='#']").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const targetId = link.getAttribute("href").slice(1);
    const target = document.getElementById(targetId);
    if (!target) return;

    window.scrollTo({
      top: target.offsetTop - 80,
      behavior: "smooth",
    });
  });
});

updateFileUI();
