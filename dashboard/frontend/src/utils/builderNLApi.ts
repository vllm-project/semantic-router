import type {
  BuilderNLGenerateRequest,
  BuilderNLGenerateResponse,
  BuilderNLProgressEvent,
} from "@/types/dsl";

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text();
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`;
  }

  try {
    const parsed = JSON.parse(body) as { error?: string; message?: string };
    return parsed.message || parsed.error || body;
  } catch {
    return body;
  }
}

export async function generateBuilderNLDraftStreaming(
  input: BuilderNLGenerateRequest,
  onProgress: (event: BuilderNLProgressEvent) => void,
): Promise<BuilderNLGenerateResponse> {
  const response = await fetch("/api/router/config/nl/generate/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  if (!response.body) {
    throw new Error("Builder NL progress stream body is empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  let eventData = "";
  let finalResult: BuilderNLGenerateResponse | null = null;

  const dispatchEvent = () => {
    if (!eventData) {
      eventType = "";
      return;
    }

    if (eventType === "progress") {
      onProgress(JSON.parse(eventData) as BuilderNLProgressEvent);
    } else if (eventType === "result") {
      finalResult = JSON.parse(eventData) as BuilderNLGenerateResponse;
    } else if (eventType === "error") {
      const payload = JSON.parse(eventData) as { message?: string };
      throw new Error(payload.message || "Builder NL generation failed");
    }

    eventType = "";
    eventData = "";
  };

  const processBuffer = (final = false) => {
    const lines = buffer.split("\n");
    buffer = final ? "" : lines.pop() || "";

    for (const rawLine of lines) {
      const line = rawLine.replace(/\r$/, "");
      if (line.startsWith(":")) {
        continue;
      }

      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
        continue;
      }

      if (line.startsWith("data:")) {
        const chunk = line.slice(5).trimStart();
        eventData = eventData ? `${eventData}\n${chunk}` : chunk;
        continue;
      }

      if (line === "") {
        dispatchEvent();
      }
    }

    if (final) {
      dispatchEvent();
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      processBuffer();
    }
    buffer += decoder.decode();
    processBuffer(true);
  } finally {
    reader.releaseLock();
  }

  if (!finalResult) {
    throw new Error("Builder NL generation stream ended without a staged draft");
  }

  return finalResult;
}
