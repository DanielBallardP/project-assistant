const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const log = document.querySelector("#log");
const sendBtn = document.querySelector("#send");
const stopBtn = document.querySelector("#stop");
let controller = null;
let nextSourceChipId = 0;

function addMessage(cls, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function renderAnswerText(container, text, sourceChips) {
  if (!sourceChips.size) {
    container.textContent = text;
    return;
  }

  container.replaceChildren();
  const citations = /\[([^\]\r\n]+)\]/g;
  let cursor = 0;

  for (const match of text.matchAll(citations)) {
    const chip = sourceChips.get(match[1]);
    if (!chip) continue;

    const start = match.index;
    container.append(document.createTextNode(text.slice(cursor, start)));

    const citation = document.createElement("a");
    citation.className = "citation";
    citation.href = `#${chip.id}`;
    citation.textContent = match[0];
    container.append(citation);
    cursor = start + match[0].length;
  }

  container.append(document.createTextNode(text.slice(cursor)));
}

function applyCitations(answer, sourcesElement, sourceChips, { cited = [], abstained = false }) {
  if (!sourceChips.size) return;

  const citedIds = new Set(cited);
  const citedChips = [];
  const otherChips = [];
  for (const [docId, chip] of sourceChips) {
    if (citedIds.has(docId)) {
      chip.classList.add("cited");
      citedChips.push(chip);
    } else {
      otherChips.push(chip);
    }
  }

  sourcesElement.replaceChildren();

  if (!citedChips.length) {
    const note = document.createElement("div");
    note.className = "sources-note";
    note.textContent = abstained
      ? "Keine passende Quelle gefunden."
      : "Die Antwort nennt keine Quelle.";
    sourcesElement.append(note);
  }
  sourcesElement.append(...citedChips);

  if (otherChips.length) {
    const more = document.createElement("details");
    more.className = "more-sources";
    const summary = document.createElement("summary");
    const label = citedChips.length ? "Weitere durchsuchte" : "Durchsuchte";
    summary.textContent = `${label} Dokumente (${otherChips.length})`;
    more.append(summary, ...otherChips);
    sourcesElement.append(more);
  }

  answer.append(sourcesElement);
}

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
  const answerTextElement = document.createElement("div");
  answer.append(answerTextElement);
  const sourcesElement = document.createElement("div");
  sourcesElement.className = "sources";
  const sourceChips = new Map();
  let answerText = "";
  let reasoningDetails = null;
  let reasoningText = null;
  let sawDone = false;
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
        sawDone = true;
        addMessage("error", data.detail ?? data.error ?? "Unknown error");
        continue;
      }
      if (event === "done") {
        sawDone = true;
        continue;
      }
      if (event === "citations") {
        applyCitations(answer, sourcesElement, sourceChips, data);
        log.scrollTop = log.scrollHeight;
        continue;
      }
      if (event === "sources") {
        sourceChips.clear();
        sourcesElement.replaceChildren();

        if (Array.isArray(data.sources)) {
          for (const source of data.sources) {
            if (
              !source ||
              typeof source.doc_id !== "string" ||
              typeof source.revision !== "string" ||
              sourceChips.has(source.doc_id)
            ) {
              continue;
            }

            const chip = document.createElement("span");
            chip.className = "source-chip";
            chip.id = `source-chip-${nextSourceChipId++}`;
            chip.textContent = `${source.doc_id} · Rev. ${source.revision}`;
            sourcesElement.append(chip);
            sourceChips.set(source.doc_id, chip);
          }
        }

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
        if (reasoningDetails) reasoningDetails.open = false;
        answerText += data.token;
        renderAnswerText(answerTextElement, answerText, sourceChips);
        log.scrollTop = log.scrollHeight;
      }
    }
    if (!sawDone) {
      addMessage("error", "Stream ended before completion");
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
    if (!answerText) answer.remove();
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

log.addEventListener("click", (event) => {
  const link = event.target.closest("a.citation");
  if (!link) return;

  event.preventDefault();
  const chip = document.getElementById(link.getAttribute("href").slice(1));
  if (!chip) return;

  chip.scrollIntoView({ block: "nearest", behavior: "smooth" });
  chip.classList.add("flash");
  chip.addEventListener("animationend", () => chip.classList.remove("flash"), {
    once: true,
  });
});

stopBtn.addEventListener("click", () => controller?.abort());