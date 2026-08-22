export const normalizedPrefix = (value: string) => {
  const trimmed = value.trim()
  return trimmed && !trimmed.endsWith('/') ? `${trimmed}/` : trimmed
}

export const parseList = (value: string) =>
  value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean)

export const optionalNumber = (value: string): number | undefined => {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

export const parseHeaders = (value: string): Record<string, string> => {
  const trimmed = value.trim()
  if (!trimmed) return {}
  const parsed: unknown = JSON.parse(trimmed)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Extra headers must be a JSON object.')
  }
  const entries = Object.entries(parsed as Record<string, unknown>)
  if (entries.some(([key, item]) => !key.trim() || typeof item !== 'string')) {
    throw new Error('Every extra header needs a text key and value.')
  }
  return Object.fromEntries(entries.map(([key, item]) => [key.trim(), String(item).trim()]))
}
