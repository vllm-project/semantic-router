import { isRecord } from "./settings.ts";
import type { Settings } from "./settings.ts";
import type { Target } from "./connection.ts";

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
  // The span presets it defines (pii, halu) and what it reads besides text (image); a listing without them names none.
  presets?: string[];
  modalities?: string[];
}
const strings = (value: unknown): string[] | undefined =>
  Array.isArray(value) && value.every((item) => typeof item === "string")
    ? [...value]
    : undefined;
export interface RuntimeStatus {
  reachable: boolean | null;
  retryAt: number;
  models: ModelInfo[];
  // Where the runtime came from: the plugin options, or a `/connect` credential.
  source: "options" | "connect";
}
interface Dependencies {
  fetch?: typeof fetch;
  now?: () => number;
  token?: () => string | undefined;
  retryMs?: number;
}
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;

// The models a listing names: the System One API's vllm-sr/4.0 schema and vllm-srun list them under `data`; the
// vllm-sr/5.0 schema, a Router listener's default, lists them under `models`, each with its name, its question types
// as `features` and the 4.0 entry as its `card`.
export function listedModels(listing: unknown): unknown[] | undefined {
  if (!isRecord(listing)) return undefined;
  if (Array.isArray(listing.data)) return listing.data;
  if (!Array.isArray(listing.models)) return undefined;
  return listing.models.map((model) =>
    isRecord(model) && isRecord(model.card)
      ? model.card
      : isRecord(model)
        ? { id: model.name, question_types: model.features }
        : model,
  );
}
function modelsOf(listing: unknown): ModelInfo[] | undefined {
  const items = listedModels(listing);
  if (!items) return undefined;
  const models: ModelInfo[] = [];
  for (const item of items) {
    if (
      !isRecord(item) ||
      typeof item.id !== "string" ||
      (item.question_types !== undefined &&
        (!Array.isArray(item.question_types) ||
          item.question_types.some((type) => typeof type !== "string")))
    )
      return undefined;
    const presets = strings(item.presets),
      modalities = strings(item.modalities);
    models.push({
      id: item.id,
      questionTypes: (item.question_types as string[] | undefined) ?? [],
      ...(presets ? { presets } : {}),
      ...(modalities ? { modalities } : {}),
    });
  }
  return models;
}

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
  private token: () => string | undefined;
  private readonly configured: { url: string; model: string };
  private readonly configuredToken: () => string | undefined;
  private readonly retryMs: number;
  private tail: Promise<void> = Promise.resolve();
  private pending = 0;
  private closed = new AbortController();
  private state: RuntimeStatus = {
    reachable: null,
    retryAt: 0,
    models: [],
    source: "options",
  };
  private readonly settings: Settings;
  constructor(settings: Settings, dependencies: Dependencies = {}) {
    this.settings = structuredClone(settings);
    this.fetcher = dependencies.fetch ?? fetch;
    this.now = dependencies.now ?? Date.now;
    this.configuredToken =
      dependencies.token ?? (() => process.env[this.settings.runtimeTokenEnv]);
    this.token = this.configuredToken;
    this.configured = {
      url: this.settings.runtimeUrl,
      model: this.settings.runtimeModel,
    };
    this.retryMs = dependencies.retryMs ?? 60_000;
  }
  // A `/connect` credential replaces the runtime the options name, and removing it brings that one back. Another
  // runtime starts unknown: what the previous one listed and its breaker do not carry over. Says whether it changed.
  connect(target: Target | undefined): boolean {
    const next = target ?? this.configured;
    const token = target ? () => target.token : this.configuredToken;
    const source = target ? "connect" : "options";
    const changed =
      next.url !== this.settings.runtimeUrl ||
      next.model !== this.settings.runtimeModel ||
      source !== this.state.source ||
      token() !== this.token();
    this.settings.runtimeUrl = next.url;
    this.settings.runtimeModel = next.model;
    this.token = token;
    if (changed)
      this.state = { reachable: null, retryAt: 0, models: [], source };
    return changed;
  }
  status(): RuntimeStatus {
    return structuredClone(this.state);
  }
  url(): string {
    return this.settings.runtimeUrl;
  }
  // The decision model asked: the configured one, else the first the runtime listed, as the Claude Code plugin's
  // calibrate names it. Unknown until a listing is read.
  model(): string | undefined {
    return this.settings.runtimeModel || this.state.models[0]?.id;
  }
  // The model each request names: the configured one, or none, so a runtime with one model answers with it.
  named(): string {
    return this.settings.runtimeModel;
  }
  close(): void {
    this.closed.abort();
  }
  private async request(
    url: string,
    body: unknown | undefined,
    timeoutMs: number,
    signal?: AbortSignal,
    queued = true,
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
    // The listener's own APIs answer beside the decision model, so they do not wait in its queue.
    const previous = queued ? this.tail : Promise.resolve();
    let release: () => void = () => {};
    if (queued)
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
  // The model asked: the configured one, or the runtime's only one; unknown until a listing is read.
  private selected(): ModelInfo | undefined {
    return this.state.models.find(
      (model) =>
        model.id === this.settings.runtimeModel ||
        (this.settings.runtimeModel === "" && this.state.models.length === 1),
    );
  }
  // Whether the model asked lists a span preset or an input modality. An unread listing says no, so a question only
  // some models answer is never sent to one that would refuse the request it rides in.
  supports(feature: { preset: string } | { modality: string }): boolean {
    const model = this.selected();
    return "preset" in feature
      ? (model?.presets ?? []).includes(feature.preset)
      : (model?.modalities ?? []).includes(feature.modality);
  }
  // `images`: base64 data URLs every question sees, for a model that reads images.
  async decide(
    state: string | Record<string, string>,
    questions: Record<string, unknown>,
    timeoutMs: number,
    signal?: AbortSignal,
    images?: readonly string[],
  ): Promise<Result<Reply>> {
    const selected = this.selected();
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
        ...(images?.length ? { images } : {}),
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
    const base = this.base();
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
    const models = this.adopt(result.value);
    if (!models) return failure("invalid-response", this.now() - started);
    return {
      ok: true,
      value: structuredClone(models),
      elapsedMs: this.now() - started,
    };
  }
  private base(): string {
    return this.settings.runtimeUrl.replace(/\/v1\/decisions$/, "");
  }
  // Keeps a listing's models for the question-type check when it names the configured one.
  private adopt(listing: unknown): ModelInfo[] | undefined {
    const models = modelsOf(listing);
    if (
      !models?.length ||
      (this.settings.runtimeModel &&
        !models.some((model) => model.id === this.settings.runtimeModel))
    )
      return undefined;
    this.state.models = models;
    return models;
  }
  // Names the listener and the key sent to it, never the key itself: another key may be granted other rules.
  listenerKey(): string {
    let hash = 0x811c9dc5;
    for (const char of this.token() ?? "")
      hash = Math.imul(hash ^ char.charCodeAt(0), 0x01000193) >>> 0;
    return `vsr.rules:${this.base()}/v1/systemone/models#${hash.toString(16)}`;
  }
  // A Router listener's System One listing: its models, and the `vsr` profile and rules it sets for the plugins. A bare
  // runtime has none (404) and lists its models at /v1/models, read here too so what they read (images, halu spans)
  // is known from the first prompt, not only after a runtime check.
  async listing(
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<Result<unknown>> {
    const started = this.now();
    const result = await this.request(
      `${this.base()}/v1/systemone/models`,
      undefined,
      timeoutMs,
      signal,
      false,
    );
    if (result.ok) this.adopt(result.value);
    else if (result.code === "http" && result.status === 404) {
      const models = await this.request(
        `${this.base()}/v1/models`,
        undefined,
        timeoutMs - (this.now() - started),
        signal,
        false,
      );
      if (models.ok) this.adopt(models.value);
    }
    return result;
  }
  // One call to the listener's rules for one app about each text.
  async rules(
    app: "guard" | "redact",
    parts: { id: string; text: string }[],
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<Result<unknown>> {
    return await this.request(
      `${this.base()}/v1/systemone/rules`,
      { app, parts },
      timeoutMs,
      signal,
      false,
    );
  }
}
