const searchForm = document.getElementById("search-form");
const audioInput = document.getElementById("audio-input");
const recordBtn = document.getElementById("record-btn");
const statusEl = document.getElementById("status");
const waveEl = document.getElementById("wave");
const spinnerEl = document.getElementById("spinner");

const MAX_SECONDS = 15;

let mediaRecorder = null;
let chunks = [];
let maxDurationHandle = null;

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      clearTimeout(maxDurationHandle);
      recordBtn.classList.remove("recording");
      waveEl.hidden = true;
      recordBtn.hidden = true;
      spinnerEl.hidden = false;
      submitAudio(new Blob(chunks, { type: "audio/webm" }));
    };

    mediaRecorder.start();
    recordBtn.classList.add("recording");
    recordBtn.setAttribute("aria-label", "Stop");
    waveEl.hidden = false;
    maxDurationHandle = setTimeout(() => mediaRecorder.stop(), MAX_SECONDS * 1000);
  } catch (err) {
    setStatus(`Microphone access denied: ${err.message}`);
  }
});

function setStatus(message) {
  statusEl.textContent = message;
}

function submitAudio(blob) {
  const file = new File([blob], "query.webm", { type: blob.type });
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  audioInput.files = dataTransfer.files;
  searchForm.submit();
}
