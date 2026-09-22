const recordBtn = document.getElementById("record-btn");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const resultsEl = document.getElementById("results");

const MAX_SECONDS = 15;

let mediaRecorder = null;
let chunks = [];
let timerHandle = null;
let seconds = 0;

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
      stopTimer();
      recordBtn.classList.remove("recording");
      recordBtn.setAttribute("aria-label", "Record");
      const blob = new Blob(chunks, { type: "audio/webm" });
      submitAudio(blob);
    };

    mediaRecorder.start();
    recordBtn.classList.add("recording");
    recordBtn.setAttribute("aria-label", "Stop");
    startTimer();
  } catch (err) {
    setStatus(`Microphone access denied: ${err.message}`);
  }
});

function startTimer() {
  seconds = 0;
  timerEl.textContent = "0s";
  timerHandle = setInterval(() => {
    seconds += 1;
    timerEl.textContent = `${seconds}s`;
    if (seconds >= MAX_SECONDS) {
      mediaRecorder.stop();
    }
  }, 1000);
}

function stopTimer() {
  clearInterval(timerHandle);
  timerEl.textContent = "0s";
}

function setStatus(message) {
  statusEl.textContent = message;
}

async function submitAudio(blob) {
  setStatus("Searching...");
  resultsEl.innerHTML = "";

  const formData = new FormData();
  formData.append("audio", blob, "query.webm");

  try {
    const response = await fetch("/api/search", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Search failed");
      return;
    }

    renderResult(data.result);
  } catch (err) {
    setStatus(`Request failed: ${err.message}`);
  }
}

function renderResult(track) {
  resultsEl.innerHTML = "";
  if (!track) {
    setStatus("No match found");
    return;
  }

  setStatus("");
  const meta = [track.raag, track.taal, track.artists].filter(Boolean).join(" · ");
  const li = document.createElement("li");
  li.innerHTML = `
    <div class="title">${track.title}</div>
    <div class="meta">${meta}</div>
  `;
  resultsEl.appendChild(li);
}
