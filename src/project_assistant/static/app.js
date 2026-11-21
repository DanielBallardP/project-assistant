const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const log = document.querySelector("#log");
const sendBtn = document.querySelector("#send");
const stopBtn = document.querySelector("#stop");
let controller = null;

function addMessage(cls, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text; // textContent, never innerHTML: model output is untrusted
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

// "event: error\ndata: {...}" -> { event: "error", data: {...} }
function parseFrame(frame) {
  let event = "message";
  const dataLines = [];

  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim() || "message";
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  return {
    event,
    data: dataLines.length ? JSON.parse(dataLines.join("\n")) : {},
  };
}

async function* readSse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      break;
    }
    // stream: true keeps an incomplete UTF-8 character buffered until more
    // bytes arrive; TextDecoder would otherwise replace it immediately.
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop();
    for (const frame of frames) {
      if (frame.trim()) yield parseFrame(frame);
    }
  }

}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || controller) return;

  addMessage("user", message);
  input.value = "";
  const answer = addMessage("assistant thinking");
  let reasoningDetails = null;
  let reasoningText = null;
  controller = new AbortController();
  sendBtn.disabled = true;
  stopBtn.hidden = false;
  input.disabled = true;

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    for await (const { event, data } of readSse(response)) {
      if (event === "error") {
        addMessage("error", data.detail ?? data.error ?? "Unknown error");
        continue;
      }
      if (event === "reasoning" && typeof data.reasoning === "string") {
        if (!reasoningDetails) {
          reasoningDetails = document.createElement("details");
          reasoningDetails.className = "reasoning";
          reasoningDetails.open = true;

          const summary = document.createElement("summary");
          summary.textContent = "Denkprozess";
          reasoningText = document.createElement("div");

          reasoningDetails.append(summary, reasoningText);
          log.insertBefore(reasoningDetails, answer);
        }
        reasoningText.textContent += data.reasoning;
        log.scrollTop = log.scrollHeight;
        continue;
      }
      if (event === "message" && typeof data.token === "string") {
        answer.classList.remove("thinking");
        answer.textContent += data.token;
        log.scrollTop = log.scrollHeight;
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      addMessage("error", err.message || "Unable to connect");
    }
  } finally {
    controller = null;
    sendBtn.disabled = false;
    stopBtn.hidden = true;
    input.disabled = false;
    answer.classList.remove("thinking");
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

stopBtn.addEventListener("click", () => controller?.abort());