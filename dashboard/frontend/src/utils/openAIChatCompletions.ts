export type OpenAIChatContent =
  | string
  | Array<
      | { type: 'text'; text: string }
      | { type: 'image_url'; image_url: { url: string; detail?: 'auto' | 'low' | 'high' } }
    >

export interface OpenAIChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: OpenAIChatContent
}

export interface OpenAIChatUsage {
  promptTokens: number
  completionTokens: number
  totalTokens: number
}

export interface OpenAIChatStreamResult {
  finishReason?: string
  headers: Record<string, string>
  model?: string
  responseId?: string
  usage?: OpenAIChatUsage
}

interface StreamChatCompletionOptions {
  accessToken: string
  endpoint: string
  messages: OpenAIChatMessage[]
  model: string
  onDelta: (delta: string) => void
  sessionId?: string
  signal: AbortSignal
  tools?: unknown[]
}

export interface ParsedChatCompletionChunk {
  deltas: string[]
  error?: string
  finishReason?: string
  model?: string
  responseId?: string
  usage?: OpenAIChatUsage
}

const MAX_REQUEST_BYTES = 10 * 1024 * 1024
const DEFAULT_MAX_COMPLETION_TOKENS = 2048

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function asNonNegativeInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function textContent(value: unknown): string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  return value
    .map((part) => {
      if (typeof part === 'string') return part
      const record = asRecord(part)
      if (!record) return ''
      return typeof record.text === 'string'
        ? record.text
        : typeof record.output_text === 'string'
          ? record.output_text
          : ''
    })
    .join('')
}

function parseUsage(value: unknown): OpenAIChatUsage | undefined {
  const usage = asRecord(value)
  if (!usage) return undefined
  const promptTokens = asNonNegativeInteger(usage.prompt_tokens)
  const completionTokens = asNonNegativeInteger(usage.completion_tokens)
  const totalTokens = asNonNegativeInteger(usage.total_tokens)
  if (promptTokens === undefined || completionTokens === undefined || totalTokens === undefined) {
    return undefined
  }
  return { promptTokens, completionTokens, totalTokens }
}

export function parseOpenAIChatCompletionChunk(value: unknown): ParsedChatCompletionChunk {
  const payload = asRecord(value)
  if (!payload) return { deltas: [], error: 'Router returned a malformed stream event.' }

  const error = asRecord(payload.error)
  if (error) {
    return {
      deltas: [],
      error:
        typeof error.message === 'string' && error.message.trim()
          ? error.message.trim()
          : 'Router returned an inference error.',
    }
  }

  const deltas: string[] = []
  let finishReason: string | undefined
  const choices = Array.isArray(payload.choices) ? payload.choices : []
  for (const candidate of choices) {
    const choice = asRecord(candidate)
    if (!choice) continue
    const delta = asRecord(choice.delta)
    const message = asRecord(choice.message)
    const content = textContent(delta?.content ?? message?.content ?? choice.content)
    if (content) deltas.push(content)
    if (!finishReason && typeof choice.finish_reason === 'string' && choice.finish_reason) {
      finishReason = choice.finish_reason
    }
  }
  const usage = parseUsage(payload.usage)

  return {
    deltas,
    ...(finishReason ? { finishReason } : {}),
    ...(typeof payload.id === 'string' && payload.id ? { responseId: payload.id } : {}),
    ...(typeof payload.model === 'string' && payload.model ? { model: payload.model } : {}),
    ...(usage ? { usage } : {}),
  }
}

const SAFE_RESPONSE_HEADER =
  /^(?:x-vsr-|x-request-id$|request-id$|openai-request-id$|x-ratelimit-)/i

export function collectOpenAIResponseHeaders(headers: Headers): Record<string, string> {
  const collected: Record<string, string> = {}
  headers.forEach((value, name) => {
    const key = name.toLocaleLowerCase()
    if (SAFE_RESPONSE_HEADER.test(key) && value.trim()) collected[key] = value.trim()
  })
  return collected
}

async function responseError(response: Response): Promise<string> {
  const fallback = `Router returned status ${response.status}.`
  try {
    const body = (await response.json()) as unknown
    const root = asRecord(body)
    const error = asRecord(root?.error)
    if (typeof error?.message === 'string' && error.message.trim()) return error.message.trim()
    if (typeof root?.message === 'string' && root.message.trim()) return root.message.trim()
  } catch {
    return fallback
  }
  return fallback
}

async function consumeOpenAIEventStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (value: unknown) => void,
): Promise<void> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const consumeFrame = (frame: string): boolean => {
    const data = frame
      .split(/\r\n|\r|\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
      .trim()
    if (!data) return false
    if (data === '[DONE]') return true
    let value: unknown
    try {
      value = JSON.parse(data)
    } catch {
      throw new Error('Router returned malformed OpenAI stream data.')
    }
    onEvent(value)
    return false
  }

  const nextBoundary = (): { index: number; length: number } | null => {
    const match = /\r\n\r\n|\n\n|\r\r/.exec(buffer)
    return match ? { index: match.index, length: match[0].length } : null
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
      let boundary = nextBoundary()
      while (boundary) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary.length)
        if (consumeFrame(frame)) {
          await reader.cancel()
          return
        }
        boundary = nextBoundary()
      }
      if (done) break
    }
    if (buffer.trim() && consumeFrame(buffer)) return
    throw new Error('Router stream ended before completion.')
  } finally {
    reader.releaseLock()
  }
}

export async function streamOpenAIChatCompletion({
  accessToken,
  endpoint,
  messages,
  model,
  onDelta,
  sessionId,
  signal,
  tools = [],
}: StreamChatCompletionOptions): Promise<OpenAIChatStreamResult> {
  const request: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    max_completion_tokens: DEFAULT_MAX_COMPLETION_TOKENS,
    ...(tools.length ? { tools, tool_choice: 'auto' } : {}),
  }
  if (new TextEncoder().encode(JSON.stringify(request)).byteLength > MAX_REQUEST_BYTES) {
    throw new Error('Playground request exceeds the 10 MB request limit.')
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    cache: 'no-store',
    credentials: 'omit',
    headers: {
      Accept: 'text/event-stream',
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
      'x-vsr-debug': 'true',
      ...(sessionId ? { 'x-session-id': sessionId } : {}),
    },
    body: JSON.stringify(request),
    signal,
  })

  if (!response.ok) throw new Error(await responseError(response))
  if (!response.headers.get('content-type')?.toLocaleLowerCase().includes('text/event-stream')) {
    throw new Error('Router did not return an OpenAI-compatible event stream.')
  }
  if (!response.body) throw new Error('Router returned an empty OpenAI-compatible event stream.')

  const result: OpenAIChatStreamResult = {
    headers: collectOpenAIResponseHeaders(response.headers),
  }
  await consumeOpenAIEventStream(response.body, (value) => {
    const chunk = parseOpenAIChatCompletionChunk(value)
    if (chunk.error) throw new Error(chunk.error)
    chunk.deltas.forEach(onDelta)
    if (chunk.finishReason) result.finishReason = chunk.finishReason
    if (chunk.model) result.model = chunk.model
    if (chunk.responseId) result.responseId = chunk.responseId
    if (chunk.usage) result.usage = chunk.usage
  })
  return result
}
