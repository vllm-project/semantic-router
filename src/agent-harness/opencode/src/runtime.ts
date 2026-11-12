import { isRecord } from "./settings.ts";
import type { Settings } from "./settings.ts";

export type Failure =
  | "cancelled"
  | "timeout"
  | "offline"
  | "http"
  | "invalid-response"
  | "unsupported"
  | "queue-full";
export type Result<T> =
  | { ok: true; value: T; elapsedMs: number }
  | { ok: false; code: Failure; elapsedMs: number; status?: number };
export interface Reply {
  answers: Record<string, unknown>;
  spans: Record<string, unknown>;
}
export interface ModelInfo {
  id: string;
  questionTypes: string[];
}
export interface RuntimeStatus {
  reachable: boolean | null;
  retryAt: number;
  models: ModelInfo[];
}
interface Dependencies {
  fetch?: typeof fetch;
  now?: () => number;
  token?: () => string | undefined;
  retryMs?: number;
}
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;

class ResponseError extends Error {}
async function jsonBody(response: Response): Promise<unknown> {
  const reader = response.body?.getReader();
  if (!reader) throw new ResponseError();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) break;
      size += next.value.byteLength;
      if (size > MAX_RESPONSE_BYTES) {
        await reader.cancel();
        throw new ResponseError();
      }
      chunks.push(next.value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new ResponseError();
  }
}
function failure<T>(
  code: Failure,
  elapsedMs: number,
  status?: number,
): Result<T> {
  return {
    ok: false,
    code,
    elapsedMs,
    ...(status === undefined ? {} : { status }),
  };
}
function waitFor(turn: Promise<void>, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const aborted = () => {
      cleanup();
      reject(signal.reason);
    };
    const cleanup = () => signal.removeEventListener("abort", aborted);
    signal.addEventListener("abort", aborted, { once: true });
    void turn.then(() => {
      cleanup();
      resolve();
    });
  });
}

export class RuntimeClient {
  private readonly fetcher: typeof fetch;
  private readonly now: () => number;
  private readonly token: () => string | undefined;
  private readonly retryMs: number;
  private tail: Promise<void> = Promise.resolve();
  private pending = 0;
  private closed = new AbortController();
  private state: RuntimeStatus = { reachable: null, retryAt: 0, models: [] };
  private readonly settings: Settings;
  constructor(settings: Settings, dependencies: Dependencies = {}) {
    this.settings = structuredClone(settings);
    this.fetcher = dependencies.fetch ?? fetch;
    this.now = dependencies.now ?? Date.now;
    this.token =
      dependencies.token ?? (() => process.env[this.settings.runtimeTokenEnv]);
    this.retryMs = dependencies.retryMs ?? 60_000;
  }
  status(): RuntimeStatus {
    return structuredClone(this.state);
  }
  close(): void {
    this.closed.abort();
  }
  private async request(
    url: string,
    body: unknown | undefined,
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<Result<unknown>> {
    const start = this.now();
    const elapsed = () => Math.max(0, this.now() - start);
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0)
      return failure("timeout", elapsed());
    if (signal?.aborted || this.closed.signal.aborted)
      return failure("cancelled", elapsed());
    if (this.pending >= 32) return failure("queue-full", elapsed());
    if (this.state.retryAt > start) return failure("offline", elapsed());
    const timer = new AbortController();
    const alarm = setTimeout(() => timer.abort(), timeoutMs);
    const combined = AbortSignal.any([
      timer.signal,
      this.closed.signal,
      ...(signal ? [signal] : []),
    ]);
    const previous = this.tail;
    let release!: () => void;
    this.tail = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.pending++;
    let admitted = false;
    try {
      await waitFor(previous, combined);
      admitted = true;
      // Recheck after queueing: an earlier request may have opened the breaker.
      if (this.state.retryAt > this.now()) return failure("offline", elapsed());
      const token = this.token();
      const response = await this.fetcher(url, {
        method: body === undefined ? "GET" : "POST",
        redirect: "error",
        signal: combined,
        headers: {
          "content-type": "application/json",
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
      this.state.reachable = true;
      this.state.retryAt = 0;
      if (!response.ok) {
        await response.body?.cancel();
        return failure("http", elapsed(), response.status);
      }
      const value = await jsonBody(response);
      return { ok: true, value, elapsedMs: elapsed() };
    } catch (error) {
      if (combined.aborted)
        return failure(
          signal?.aborted || this.closed.signal.aborted
            ? "cancelled"
            : "timeout",
          elapsed(),
        );
      if (error instanceof ResponseError)
        return failure("invalid-response", elapsed());
      this.state.reachable = false;
      this.state.retryAt = this.now() + this.retryMs;
      return failure("offline", elapsed());
    } finally {
      clearTimeout(alarm);
      this.pending--;
      // An expired queued request must not let its successors overtake the active one.
      if (admitted) release();
      else void previous.then(release);
    }
  }
  async decide(
    state: string | Record<string, string>,
    questions: Record<string, unknown>,
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<Result<Reply>> {
    const selected = this.state.models.find(
      (model) =>
        model.id === this.settings.runtimeModel ||
        (this.settings.runtimeModel === "" && this.state.models.length === 1),
    );
    if (
      selected &&
      selected.questionTypes.length > 0 &&
      Object.values(questions).some(
        (question) =>
          isRecord(question) &&
          typeof question.type === "string" &&
          !selected.questionTypes.includes(question.type),
      )
    )
      return failure("unsupported", 0);
    const result = await this.request(
      this.settings.runtimeUrl,
      {
        ...(this.settings.runtimeModel
          ? { model: this.settings.runtimeModel }
          : {}),
        state,
        questions,
      },
      timeoutMs,
      signal,
    );
    if (!result.ok) return result;
    if (
      !isRecord(result.value) ||
      !isRecord(result.value.answers) ||
      (result.value.spans !== undefined && !isRecord(result.value.spans))
    )
      return failure("invalid-response", result.elapsedMs);
    const answers = result.value.answers;
    if (Object.keys(questions).some((key) => !Object.hasOwn(answers, key)))
      return failure("invalid-response", result.elapsedMs);
    return {
      ok: true,
      value: { answers: result.value.answers, spans: result.value.spans ?? {} },
      elapsedMs: result.elapsedMs,
    };
  }
  async discover(
    timeoutMs = 3000,
    signal?: AbortSignal,
  ): Promise<Result<ModelInfo[]>> {
    const started = this.now();
    const base = this.settings.runtimeUrl.replace(/\/v1\/decisions$/, "");
    let result = await this.request(
      `${base}/v1/systemone/models`,
      undefined,
      timeoutMs,
      signal,
    );
    if (!result.ok && result.code === "http" && result.status === 404)
      result = await this.request(
        `${base}/v1/models`,
        undefined,
        timeoutMs - (this.now() - started),
        signal,
      );
    if (!result.ok) return result;
    if (!isRecord(result.value) || !Array.isArray(result.value.data))
      return failure("invalid-response", this.now() - started);
    const models: ModelInfo[] = [];
    for (const item of result.value.data) {
      if (
        !isRecord(item) ||
        typeof item.id !== "string" ||
        (item.question_types !== undefined &&
          (!Array.isArray(item.question_types) ||
            item.question_types.some((type) => typeof type !== "string")))
      )
        return failure("invalid-response", this.now() - started);
      models.push({
        id: item.id,
        questionTypes: (item.question_types as string[] | undefined) ?? [],
      });
    }
    if (
      !models.length ||
      (this.settings.runtimeModel &&
        !models.some((model) => model.id === this.settings.runtimeModel))
    )
      return failure("invalid-response", this.now() - started);
    this.state.models = models;
    return {
      ok: true,
      value: structuredClone(models),
      elapsedMs: this.now() - started,
    };
  }
}
