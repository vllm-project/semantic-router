import type { ToolCall } from '../tools'

const TOOL_LABELS: Record<string, string> = {
  calculate: 'Calculator',
  current_time: 'Current Time',
  get_weather: 'Weather',
  open_web: 'Web Page',
  search_web: 'Web Search',
}

const STATUS_LABELS: Record<ToolCall['status'], string> = {
  pending: 'Queued',
  running: 'Running',
  completed: 'Done',
  failed: 'Failed',
  skipped: 'Not executed',
}

function readStringField(args: Record<string, unknown> | null, key: string) {
  const value = args?.[key]
  return typeof value === 'string' ? value.trim() : ''
}

export function getToolDisplayName(toolName: string) {
  return TOOL_LABELS[toolName] || toolName
}

export function getToolStatusLabel(status: ToolCall['status']) {
  return STATUS_LABELS[status]
}

export function getToolSummary(args: Record<string, unknown> | null) {
  const name = readStringField(args, 'name')
  const query = readStringField(args, 'query')
  const url = readStringField(args, 'url')
  const location = readStringField(args, 'location')
  const expression = readStringField(args, 'expression')
  const timezone = readStringField(args, 'timezone')

  if (query) {
    return `"${query}"`
  }

  if (expression) {
    return expression
  }

  if (location) {
    return location
  }

  if (url) {
    try {
      return new URL(url).hostname
    } catch {
      return url
    }
  }

  if (timezone) {
    return timezone
  }

  if (name) {
    return name
  }

  return 'Tool execution'
}
