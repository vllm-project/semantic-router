// Optional input passed through to the Router's routing/preview request.
export interface PreviewEnvelope {
  messages?: Record<string, unknown>[]
  metadata?: Record<string, string>
  tools?: Record<string, unknown>[]
}

export interface PreviewOptions extends PreviewEnvelope {
  recipe?: string
}

export function parsePreviewEnvelope(value: string): PreviewEnvelope {
  if (!value.trim()) return {}
  const parsed: unknown = JSON.parse(value)
  if (!isObject(parsed)) throw new Error('Request context must be a JSON object.')
  for (const key of Object.keys(parsed)) {
    if (!['messages', 'metadata', 'tools'].includes(key)) {
      throw new Error(`Unsupported request context field: ${key}. Use messages, metadata or tools.`)
    }
  }
  for (const key of ['messages', 'tools']) {
    const items = parsed[key]
    if (items !== undefined && (!Array.isArray(items) || !items.every(isObject))) {
      throw new Error(`${key} must be an array of objects.`)
    }
  }
  if (
    parsed.metadata !== undefined &&
    (!isObject(parsed.metadata) ||
      Object.values(parsed.metadata).some((value) => typeof value !== 'string'))
  ) {
    throw new Error('metadata must be an object with string values.')
  }
  return parsed as PreviewEnvelope
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
