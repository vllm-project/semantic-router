// The gateway's side of the extension: its model list, its Chat Completions
// stream, and its routing headers. This module only imports vscode's types, so
// the tests run it with plain stand-ins for VS Code's part classes.
import { randomUUID } from 'node:crypto';
import type * as vscode from 'vscode';

/** VS Code's part classes, handed in by extension.ts. */
export interface Parts {
  Text: typeof vscode.LanguageModelTextPart;
  ToolCall: typeof vscode.LanguageModelToolCallPart;
  ToolResult: typeof vscode.LanguageModelToolResultPart;
}

export interface Gateway {
  url: string;
  apiKey?: string;
  sessionId?: string;
  /** How long to wait for the gateway to answer, in ms, instead of the defaults below. */
  timeoutMs?: number;
}

// How long the model list may take, and how long a reply may take to start.
// A reply that has started streams for as long as it takes.
const LIST_TIMEOUT_MS = 30_000;
const REPLY_TIMEOUT_MS = 120_000;

/** One entry of the gateway's `GET /v1/models`. */
export interface GatewayModel {
  id: string;
  description?: string;
  routing?: { resolution?: string; recipe?: string };
}

/** What the gateway said about the reply it served. */
export interface Route {
  requested: string;
  model?: string;
  decision?: string;
  recipe?: string;
  path?: string;
  replyModel?: string;
}

interface WireToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export type WireMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: WireToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string };

export interface ChatRequest {
  model: string;
  messages: WireMessage[];
  tools?: readonly vscode.LanguageModelChatTool[];
  toolRequired?: boolean;
}

type Report = (part: vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart) => void;

// A message can carry role 3, the System role of VS Code's proposed
// languageModelSystem API; the stable API names only User (1) and Assistant (2).
const ROLES: Record<number, 'user' | 'assistant' | 'system'> = { 1: 'user', 2: 'assistant', 3: 'system' };

/** Converts VS Code's chat messages into Chat Completions messages. */
export function toWireMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[],
  parts: Parts,
): WireMessage[] {
  const wire: WireMessage[] = [];
  for (const message of messages) {
    const role = ROLES[message.role] ?? 'user';
    let text = '';
    const calls: WireToolCall[] = [];
    for (const part of message.content) {
      if (part instanceof parts.Text) {
        text += part.value;
      } else if (part instanceof parts.ToolCall) {
        calls.push({ id: part.callId, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input) } });
      } else if (part instanceof parts.ToolResult) {
        // A tool message must follow the assistant message that called it,
        // so results go out before the rest of their user message.
        wire.push({ role: 'tool', tool_call_id: part.callId, content: resultText(part.content, parts) });
      }
      // Images, cache hints and other parts are not sent.
    }
    if (role === 'assistant' && (text || calls.length)) {
      wire.push({ role, content: text || null, ...(calls.length ? { tool_calls: calls } : {}) });
    } else if (role !== 'assistant' && text) {
      wire.push({ role, content: text });
    }
  }
  return wire;
}

const decoder = new TextDecoder();

function resultText(content: readonly unknown[], parts: Parts): string {
  return content.map((part) => (part instanceof parts.Text ? part.value : isTextData(part) ? decoder.decode(part.data) : '')).join('');
}

// A LanguageModelDataPart that holds text or JSON. Newer VS Code releases
// put these in tool results; the class is not in the API this extension pins.
function isTextData(part: unknown): part is { mimeType: string; data: Uint8Array } {
  const { mimeType, data } = (part ?? {}) as { mimeType?: unknown; data?: unknown };
  return typeof mimeType === 'string' && data instanceof Uint8Array && (mimeType.startsWith('text/') || mimeType === 'application/json');
}

/** A rough token count, about four characters a token, as VS Code's guide suggests. */
export function estimateTokens(input: string | vscode.LanguageModelChatRequestMessage, parts: Parts): number {
  const text = typeof input === 'string' ? input : JSON.stringify(toWireMessages([input], parts));
  return Math.ceil(text.length / 4);
}

/**
 * The models to offer: the ids in `wanted` the gateway lists, in that order,
 * or with none wanted, every entrypoint (virtual model) it lists.
 */
export function listModels(gateway: Gateway, wanted: readonly string[], signal?: AbortSignal): Promise<GatewayModel[]> {
  return redacted(gateway, () => readModels(gateway, wanted, signal));
}

async function readModels(gateway: Gateway, wanted: readonly string[], signal?: AbortSignal): Promise<GatewayModel[]> {
  const { signal: timed } = deadline(gateway, gateway.timeoutMs ?? LIST_TIMEOUT_MS, signal);
  const response = await fetch(`${baseUrl(gateway.url)}/v1/models`, { headers: requestHeaders(gateway), signal: timed });
  if (!response.ok) {
    throw await gatewayError(response, gateway.apiKey);
  }
  const models = ((await response.json()) as { data?: GatewayModel[] }).data ?? [];
  if (!wanted.length) {
    return models.filter((model) => model.routing?.resolution === 'virtual');
  }
  return wanted.flatMap((id) => models.filter((model) => model.id === id));
}

/** Streams one reply into `report` and returns what the gateway said about its route. */
export function chat(gateway: Gateway, request: ChatRequest, parts: Parts, report: Report, signal?: AbortSignal): Promise<Route> {
  return redacted(gateway, () => streamReply(gateway, request, parts, report, signal));
}

async function streamReply(gateway: Gateway, request: ChatRequest, parts: Parts, report: Report, signal?: AbortSignal): Promise<Route> {
  const tools = (request.tools ?? []).map((tool) => ({
    type: 'function',
    function: { name: tool.name, description: tool.description, parameters: tool.inputSchema ?? { type: 'object', properties: {} } },
  }));
  const body = {
    model: request.model,
    messages: request.messages,
    stream: true,
    ...(tools.length ? { tools, ...(request.toolRequired ? { tool_choice: 'required' } : {}) } : {}),
  };
  const wait = deadline(gateway, gateway.timeoutMs ?? REPLY_TIMEOUT_MS, signal);
  const response = await fetch(`${baseUrl(gateway.url)}/v1/chat/completions`, {
    method: 'POST',
    headers: requestHeaders(gateway),
    body: JSON.stringify(body),
    signal: wait.signal,
  });
  wait.answered();
  if (!response.ok) {
    throw await gatewayError(response, gateway.apiKey);
  }
  const header = (name: string) => response.headers.get(name) || undefined;
  const route: Route = {
    requested: request.model,
    model: header('x-vsr-selected-model'),
    decision: header('x-vsr-selected-decision'),
    recipe: header('x-vsr-selected-recipe'),
    path: header('x-vsr-response-path'),
  };
  const reply = new Reply(parts, report);
  if (!(response.headers.get('content-type') ?? '').includes('text/event-stream')) {
    // Answered whole, not streamed.
    reply.add((await response.json()) as Chunk);
    reply.end(true);
  } else if (response.body) {
    let done = false;
    for await (const data of serverSentData(response.body)) {
      if (data === '[DONE]') {
        done = true;
        break;
      }
      reply.add(JSON.parse(data) as Chunk);
    }
    reply.end(done);
  } else {
    throw new Error('The gateway sent an empty reply.');
  }
  route.replyModel = reply.model;
  return route;
}

interface Delta {
  content?: string | null;
  tool_calls?: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[];
}

interface Chunk {
  model?: string;
  error?: { message?: string };
  choices?: { delta?: Delta; message?: Delta; finish_reason?: string | null }[];
}

// Collects one reply: text goes out as it arrives, tool calls once their
// arguments are complete.
class Reply {
  model?: string;
  private finished = false;
  private readonly calls = new Map<number, { id?: string; name: string; args: string }>();

  constructor(
    private readonly parts: Parts,
    private readonly report: Report,
  ) {}

  add(chunk: Chunk): void {
    if (chunk.error) {
      throw new Error(`The gateway failed the reply: ${chunk.error.message ?? JSON.stringify(chunk.error)}`);
    }
    this.model ??= chunk.model || undefined;
    const choice = chunk.choices?.[0];
    const delta = choice?.delta ?? choice?.message;
    if (delta?.content) {
      this.report(new this.parts.Text(delta.content));
    }
    delta?.tool_calls?.forEach((call, position) => {
      const index = call.index ?? position;
      const entry = this.calls.get(index) ?? { name: '', args: '' };
      entry.id ||= call.id;
      entry.name ||= call.function?.name ?? '';
      entry.args += call.function?.arguments ?? '';
      this.calls.set(index, entry);
    });
    if (choice?.finish_reason) {
      this.finished = true;
    }
  }

  end(done: boolean): void {
    if (!done && !this.finished) {
      throw new Error('The gateway closed the stream before the reply finished.');
    }
    for (const [, call] of [...this.calls].sort(([a], [b]) => a - b)) {
      // A call id must stay unique in the conversation, so make one up when
      // the backend sent none.
      this.report(new this.parts.ToolCall(call.id ?? `call_${randomUUID()}`, call.name, toolInput(call.name, call.args)));
    }
  }
}

function toolInput(name: string, args: string): object {
  let input: unknown;
  try {
    input = args.trim() ? JSON.parse(args) : {};
  } catch {
    input = undefined;
  }
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    throw new Error(`The model called ${name} with arguments that are not a JSON object.`);
  }
  return input;
}

/** The `data:` payloads of a server-sent event stream. */
export async function* serverSentData(body: AsyncIterable<Uint8Array>): AsyncGenerator<string> {
  const text = new TextDecoder();
  let buffer = '';
  for await (const bytes of body) {
    buffer += text.decode(bytes, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (line.startsWith('data:')) {
        yield line.slice(5).trim();
      }
    }
  }
  buffer += text.decode();
  if (buffer.startsWith('data:')) {
    yield buffer.slice(5).trim();
  }
}

/** The status bar's words for a route: the model that answered, then the details. */
export function describeRoute(route: Route): { model: string; lines: string[] } {
  const model = route.model ?? route.replyModel ?? route.requested;
  const source = route.model
    ? ''
    : route.replyModel
      ? ' (named by the reply: the gateway sent no x-vsr-selected-model)'
      : ' (asked for: the gateway did not say which model answered)';
  const lines = [`Entrypoint: ${route.requested}`, `Model: ${model}${source}`, `Decision: ${route.decision ?? 'none'}`];
  if (route.recipe) {
    lines.push(`Recipe: ${route.recipe}`);
  }
  if (route.path && route.path !== 'upstream') {
    lines.push(`Response path: ${route.path}`);
  }
  return { model, lines };
}

function baseUrl(url: string): string {
  return url.trim().replace(/\/+$/, '').replace(/\/v1$/, '');
}

function requestHeaders(gateway: Gateway): Record<string, string> {
  return {
    'content-type': 'application/json',
    ...(gateway.apiKey ? { authorization: `Bearer ${gateway.apiKey}` } : {}),
    ...(gateway.sessionId ? { 'x-session-id': gateway.sessionId } : {}),
  };
}

// `signal`, which also aborts with a readable error when the gateway has not
// answered within `ms`, unless `answered` is called first.
function deadline(gateway: Gateway, ms: number, signal?: AbortSignal) {
  const clock = new AbortController();
  const late = Object.assign(new Error(`The gateway at ${baseUrl(gateway.url)} did not answer within ${ms / 1000} s.`), { name: 'TimeoutError' });
  const timer = setTimeout(() => clock.abort(late), ms).unref();
  return { signal: signal ? AbortSignal.any([signal, clock.signal]) : clock.signal, answered: () => clearTimeout(timer) };
}

async function gatewayError(response: Response, apiKey?: string): Promise<Error> {
  // Redacted before it is cut, so no part of the key is left at the cut.
  const text = redact(await response.text().catch(() => ''), apiKey);
  let message = text.slice(0, 500);
  try {
    message = (JSON.parse(text) as { error?: { message?: string } }).error?.message ?? message;
  } catch {
    // Not JSON: keep the text.
  }
  const hint = response.status === 401 ? ' Run "vLLM Semantic Router: Set Gateway API Key".' : '';
  const sentence = (message.trim() || response.statusText).replace(/\.?$/, '.');
  return new Error(`The gateway answered ${response.status}: ${sentence}${hint}`);
}

// A gateway, or a proxy in front of it, can echo the API key in what it
// answers, and an error's message reaches the log, the status bar and the
// chat. So every error leaves this module with the key and any bearer token
// taken out, and a parse error, which quotes some of the text it could not
// read, says only what failed.
async function redacted<T>(gateway: Gateway, work: () => Promise<T>): Promise<T> {
  try {
    return await work();
  } catch (error) {
    const text =
      error instanceof SyntaxError
        ? 'The gateway sent a reply that is not JSON.'
        : redact(reasonOf(gateway, error), gateway.apiKey);
    throw Object.assign(new Error(text), { name: error instanceof Error ? error.name : 'Error' });
  }
}

// fetch says only "fetch failed" (or "terminated" mid-reply) when the network
// fails, and keeps why, with the code and the address, in `cause`.
function reasonOf(gateway: Gateway, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const { code, message: why } = ((error as { cause?: unknown })?.cause ?? {}) as { code?: unknown; message?: string };
  if (typeof code !== 'string') {
    return message;
  }
  const reason = why?.includes(code) ? why : `${why || message}, ${code}`;
  return code === 'ECONNREFUSED'
    ? `The gateway at ${baseUrl(gateway.url)} refused the connection (${reason}).`
    : `The connection to the gateway at ${baseUrl(gateway.url)} failed (${reason}).`;
}

/**
 * `text` with `apiKey` and every bearer token in it (RFC 6750 token
 * characters, so a full stop after one stays) replaced by `***`.
 */
function redact(text: string, apiKey?: string): string {
  const withoutKey = apiKey ? text.split(apiKey).join('***') : text;
  return withoutKey.replace(/\bBearer\s+[\w~+/-]+(?:\.[\w~+/-]+)*=*/gi, 'Bearer ***');
}
