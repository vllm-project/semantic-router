import type { ToolResult } from './types'

const DEFAULT_MAX_TOOL_RESULT_LENGTH = 15000
const TOOL_EXECUTION_FAILURE_PREFIX = 'Tool execution failed:'

const truncateToolResultText = (content: string, maxLength: number): string => {
  if (content.length <= maxLength) {
    return content
  }

  return `${content.substring(0, maxLength)}\n\n...[Content truncated due to length]`
}

const stringifyToolResultContent = (content: unknown): string => {
  if (typeof content === 'string') {
    return content
  }

  try {
    return JSON.stringify(content ?? null)
  } catch {
    return String(content ?? '')
  }
}

export const serializeToolResultForModel = (
  toolResult: ToolResult,
  maxLength = DEFAULT_MAX_TOOL_RESULT_LENGTH,
): string => {
  const content = toolResult.error
    ? `${TOOL_EXECUTION_FAILURE_PREFIX} ${toolResult.error}`
    : stringifyToolResultContent(toolResult.content)

  return truncateToolResultText(content, maxLength)
}
