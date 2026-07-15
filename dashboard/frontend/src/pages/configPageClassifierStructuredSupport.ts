import type { MCPCategoryModel } from './configPageSupport'

function parseLegacyJson(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  if (!trimmed) return undefined

  try {
    return JSON.parse(trimmed)
  } catch {
    return value
  }
}

export function normalizeMcpArguments(value: unknown): string[] {
  const parsed = parseLegacyJson(value)
  if (parsed === undefined || parsed === null) return []
  if (!Array.isArray(parsed)) {
    throw new Error('MCP arguments must be a list of text values.')
  }

  return parsed.map((item, index) => {
    if (typeof item !== 'string' || !item.trim()) {
      throw new Error(`MCP argument ${index + 1} must be a non-empty text value.`)
    }
    return item.trim()
  })
}

export function normalizeMcpEnvironment(value: unknown): Record<string, string> {
  const parsed = parseLegacyJson(value)
  if (parsed === undefined || parsed === null) return {}
  if (typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('MCP environment variables must be text key/value pairs.')
  }

  const normalized: Record<string, string> = {}
  for (const [rawKey, rawValue] of Object.entries(parsed as Record<string, unknown>)) {
    const key = rawKey.trim()
    if (!key || typeof rawValue !== 'string') {
      throw new Error('MCP environment variables must use non-empty text keys and text values.')
    }
    if (Object.prototype.hasOwnProperty.call(normalized, key)) {
      throw new Error(`MCP environment variable ${key} is duplicated.`)
    }
    normalized[key] = rawValue
  }
  return normalized
}

export function normalizeMcpCategoryModel(data: MCPCategoryModel): MCPCategoryModel {
  const record = data as MCPCategoryModel & Record<string, unknown>
  const normalized: MCPCategoryModel = { ...data }

  if (record.args !== undefined) normalized.args = normalizeMcpArguments(record.args)
  if (record.env !== undefined) normalized.env = normalizeMcpEnvironment(record.env)

  return normalized
}
