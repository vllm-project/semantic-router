import type { CSSProperties, ReactNode } from 'react'

import MarkdownRenderer from '../components/MarkdownRenderer'
import type { ViewField } from '../components/ViewModal'

import type { InsightsRecord, ToolTrace, ToolTraceStep } from './insightsPageTypes'
import { hasMeaningfulToolResultText } from '../tools/toolResultSupport'
import styles from './InsightsPage.module.css'

const TOOL_TRACE_STYLES: Record<string, { tint: string; accent: string }> = {
  user_input: {
    tint: 'rgba(118, 185, 0, 0.12)',
    accent: 'rgba(163, 230, 53, 0.9)',
  },
  assistant_tool_call: {
    tint: 'rgba(255, 255, 255, 0.06)',
    accent: 'rgba(255, 255, 255, 0.9)',
  },
  client_tool_result: {
    tint: 'rgba(118, 185, 0, 0.08)',
    accent: 'rgba(220, 252, 231, 0.92)',
  },
  assistant_final_response: {
    tint: 'rgba(255, 255, 255, 0.08)',
    accent: 'rgba(255, 255, 255, 0.96)',
  },
}

export function renderToolNamesCell(record: InsightsRecord): ReactNode {
  const toolNames = getTraceToolNames(record.tool_trace)
  if (toolNames.length === 0) {
    return <span>-</span>
  }

  const visibleToolNames = toolNames.slice(0, 2)
  const hiddenCount = toolNames.length - visibleToolNames.length
  const summary = hiddenCount > 0
    ? `${visibleToolNames.join(' · ')} · +${hiddenCount}`
    : visibleToolNames.join(' · ')

  return (
    <span className={styles.tableSummaryText} title={toolNames.join(', ')}>
      {summary}
    </span>
  )
}

export function buildToolTraceFields(
  record: InsightsRecord,
  options: { canViewFlowDetails: boolean },
): ViewField[] {
  const trace = record.tool_trace
  if (!trace || !hasToolTraceContent(trace)) {
    return []
  }

  return buildToolTraceFieldList(trace, options.canViewFlowDetails)
}

function buildToolTraceFieldList(trace: ToolTrace, canViewFlowDetails: boolean): ViewField[] {
  const fields: ViewField[] = [
    {
      label: 'Overview',
      value: renderToolTraceOverview(trace),
      fullWidth: true,
    },
  ]

  fields.push({
    label: 'Flow',
    value: renderToolTraceFlow(trace, canViewFlowDetails),
    fullWidth: true,
  })

  return fields
}

function renderToolTraceOverview(trace: ToolTrace) {
  const steps = trace.steps ?? []
  const summary = summarizeToolTrace(trace)
  const successRate =
    summary.totalToolCalls > 0
      ? `${Math.round((summary.successfulToolCalls / summary.totalToolCalls) * 100)}%`
      : '-'

  return (
    <div className={styles.toolTraceSummaryGrid}>
      <article className={styles.toolTraceSummaryCard}>
        <span className={styles.toolTraceSummaryLabel}>Total Tool Calls</span>
        <strong className={styles.toolTraceSummaryValue}>{summary.totalToolCalls}</strong>
        <span className={styles.toolTraceSummaryMeta}>
          {summary.totalToolCalls > 0
            ? `${summary.successfulToolCalls} succeeded, ${summary.failedToolCalls} failed`
            : 'No tool calls recorded'}
        </span>
      </article>

      <article className={styles.toolTraceSummaryCard}>
        <span className={styles.toolTraceSummaryLabel}>Tools Used</span>
        {summary.toolNames.length > 0 ? (
          <div className={styles.toolTraceSummaryTools}>
            {summary.toolNames.map((toolName) => (
              <span key={toolName} className={styles.signalPillCompact}>
                {toolName}
              </span>
            ))}
          </div>
        ) : (
          <strong className={styles.toolTraceSummaryValue}>-</strong>
        )}
        <span className={styles.toolTraceSummaryMeta}>
          {summary.toolNames.length > 0
            ? `${summary.toolNames.length} distinct tool${summary.toolNames.length > 1 ? 's' : ''}`
            : 'No tools used'}
        </span>
      </article>

      <article className={styles.toolTraceSummaryCard}>
        <span className={styles.toolTraceSummaryLabel}>Tool Call Success Rate</span>
        <strong className={styles.toolTraceSummaryValue}>{successRate}</strong>
        <span className={styles.toolTraceSummaryMeta}>
          {summary.totalToolCalls > 0
            ? `${summary.successfulToolCalls} / ${summary.totalToolCalls} succeeded`
            : 'Waiting for first tool call'}
        </span>
      </article>

      <article className={styles.toolTraceSummaryCard}>
        <span className={styles.toolTraceSummaryLabel}>Current Stage</span>
        <strong className={styles.toolTraceSummaryValue}>
          {formatToolTraceStage(trace.stage, steps[steps.length - 1]?.type)}
        </strong>
        <span className={styles.toolTraceSummaryMeta}>
          {steps.length > 0 ? `Latest step ${steps.length}` : 'No detailed steps captured'}
        </span>
      </article>
    </div>
  )
}

function renderToolTraceFlow(trace: ToolTrace, canViewFlowDetails: boolean) {
  const steps = trace.steps ?? []
  if (steps.length === 0) {
    return (
      <div className={styles.pluginStack}>
        <span className={styles.costSubtle}>No detailed tool steps were captured for this request.</span>
        {trace.flow ? <pre className={styles.toolTraceBlock}>{trace.flow}</pre> : null}
      </div>
    )
  }

  return (
    <div className={styles.toolTraceFlowShell}>
      <div className={styles.toolTraceLegend}>
        {([
          'user_input',
          'assistant_tool_call',
          'client_tool_result',
          'assistant_final_response',
        ] as const).map((stepType) => (
          <span
            key={stepType}
            className={styles.toolTraceLegendItem}
            style={toolTraceTintStyle(stepType)}
          >
            <span className={styles.toolTraceLegendIcon}>{renderToolTraceStepIcon(stepType)}</span>
            <span className={styles.toolTraceLegendLabel}>{formatToolTraceStepLabel(stepType)}</span>
          </span>
        ))}
      </div>

      <div className={styles.toolTraceTimeline}>
      {steps.map((step, index) => (
        <div key={buildToolTraceStepKey(step, index)} className={styles.toolTraceStepRow}>
          <div className={styles.toolTraceMarker}>
            <span className={styles.toolTraceMarkerDot} style={toolTraceTintStyle(step.type)}>
              {index + 1}
            </span>
            {index < steps.length - 1 ? <span className={styles.toolTraceMarkerLine} /> : null}
          </div>

          <article className={styles.toolTraceStepCard} style={toolTraceTintStyle(step.type)}>
            <div className={styles.toolTraceHeader}>
              <span className={styles.toolTraceRolePill} style={toolTraceTintStyle(step.type)}>
                <span className={styles.toolTraceRoleIcon}>{renderToolTraceStepIcon(step.type)}</span>
                {formatToolTraceStepLabel(step.type)}
              </span>
              {step.tool_name ? (
                <span className={styles.signalPillCompact}>{step.tool_name}</span>
              ) : null}
              {formatToolTraceSource(step) ? (
                <span className={styles.costSubtle}>Source: {formatToolTraceSource(step)}</span>
              ) : null}
            </div>

            <div className={styles.toolTraceTitleRow}>
              <strong className={styles.toolTraceTitle}>{formatToolTraceHeadline(step)}</strong>
              {step.type === 'client_tool_result' ? (
                <span
                  className={`${styles.toolTraceStatusBadge} ${
                    isSuccessfulToolResult(step)
                      ? styles.toolTraceStatusBadgeSuccess
                      : styles.toolTraceStatusBadgeFailed
                  }`}
                >
                  {isSuccessfulToolResult(step) ? 'Completed' : 'Failed'}
                </span>
              ) : null}
            </div>

            {step.tool_call_id ? (
              <span className={styles.toolTraceMeta}>Call ID: {step.tool_call_id}</span>
            ) : null}
            {step.type === 'client_tool_result' && !isSuccessfulToolResult(step) ? (
              <span className={styles.toolTraceMeta}>
                Null or empty tool result returned to the model
              </span>
            ) : null}
            {canViewFlowDetails && step.arguments ? (
              <pre className={styles.toolTraceBlock}>{formatTraceBlock(step.arguments)}</pre>
            ) : null}
            {canViewFlowDetails && step.text ? (
              renderToolTraceContent(step)
            ) : null}
            {!canViewFlowDetails && step.content_redacted ? (
              <span className={styles.toolTraceMeta}>Inputs and outputs are hidden for your role</span>
            ) : null}
          </article>
        </div>
      ))}
      </div>
    </div>
  )
}

function buildToolTraceStepKey(step: ToolTraceStep, index: number) {
  return [
    step.type,
    step.tool_call_id || '',
    step.tool_name || '',
    String(index),
  ].join(':')
}

function formatToolTraceStepLabel(stepType: string) {
  switch (stepType) {
    case 'user_input':
      return 'User Query'
    case 'assistant_tool_call':
      return 'Tool Calling'
    case 'client_tool_result':
      return 'Tool Execute'
    case 'assistant_final_response':
      return 'LLM Answer'
    default:
      return stepType
  }
}

function formatToolTraceHeadline(step: ToolTraceStep) {
  switch (step.type) {
    case 'assistant_tool_call':
      return step.tool_name ? `Tool Calling (${step.tool_name})` : 'Tool Calling'
    case 'client_tool_result':
      return step.tool_name ? `Tool Execute (${step.tool_name})` : 'Tool Execute'
    case 'assistant_final_response':
      return 'LLM Answer'
    case 'user_input':
      return 'User Query'
    default:
      return formatToolTraceStepLabel(step.type)
  }
}

function summarizeToolTrace(trace: ToolTrace) {
  const steps = trace.steps ?? []
  const toolCalls = steps.filter((step) => step.type === 'assistant_tool_call')
  const toolResults = steps.filter((step) => step.type === 'client_tool_result')
  const usedToolResults = new Set<number>()
  let successfulToolCalls = 0

  for (const toolCall of toolCalls) {
    const resultIndex = findMatchingToolResultIndex(toolCall, toolResults, usedToolResults)
    if (resultIndex < 0) {
      continue
    }
    usedToolResults.add(resultIndex)
    if (isSuccessfulToolResult(toolResults[resultIndex])) {
      successfulToolCalls += 1
    }
  }

  const totalToolCalls = toolCalls.length

  return {
    totalToolCalls,
    successfulToolCalls,
    failedToolCalls: Math.max(totalToolCalls - successfulToolCalls, 0),
    toolNames: getTraceToolNames(trace),
  }
}

function findMatchingToolResultIndex(
  toolCall: ToolTraceStep,
  toolResults: ToolTraceStep[],
  usedToolResults: Set<number>,
) {
  if (toolCall.tool_call_id) {
    const matchByCallId = toolResults.findIndex(
      (result, index) =>
        !usedToolResults.has(index) && result.tool_call_id === toolCall.tool_call_id,
    )
    if (matchByCallId >= 0) {
      return matchByCallId
    }
  }

  if (toolCall.tool_name) {
    const matchByToolName = toolResults.findIndex(
      (result, index) =>
        !usedToolResults.has(index) &&
        !result.tool_call_id &&
        result.tool_name === toolCall.tool_name,
    )
    if (matchByToolName >= 0) {
      return matchByToolName
    }
  }

  return toolResults.findIndex((_, index) => !usedToolResults.has(index))
}

function isSuccessfulToolResult(step: ToolTraceStep) {
  return getToolResultStatus(step) === 'succeeded'
}

function getToolResultStatus(step: ToolTraceStep) {
  if (step.type !== 'client_tool_result') {
    return null
  }

  const normalizedStatus = step.status?.trim().toLowerCase()
  if (normalizedStatus === 'succeeded' || normalizedStatus === 'failed') {
    return normalizedStatus
  }

  return hasMeaningfulToolResultText(step.text) ? 'succeeded' : 'failed'
}

function getTraceToolNames(trace?: ToolTrace) {
  if (!trace) {
    return []
  }

  const toolNames = new Set<string>(trace.tool_names ?? [])
  for (const step of trace.steps ?? []) {
    if (step.tool_name) {
      toolNames.add(step.tool_name)
    }
  }
  return [...toolNames]
}

function formatToolTraceStage(stage?: string, fallbackStepType?: string) {
  if (stage) {
    const normalized = stage.trim().toLowerCase()
    if (normalized === 'llm tool call' || normalized === 'assistant_tool_call') {
      return 'Tool Calling'
    }
    if (normalized === 'client tool result' || normalized === 'client_tool_result') {
      return 'Tool Execute'
    }
    if (normalized === 'llm final response' || normalized === 'assistant_final_response') {
      return 'LLM Answer'
    }
    if (normalized === 'user query' || normalized === 'user_input') {
      return 'User Query'
    }
    return stage
  }

  if (fallbackStepType) {
    return formatToolTraceStepLabel(fallbackStepType)
  }
  return '-'
}

function hasToolTraceContent(trace?: ToolTrace) {
  return Boolean(trace?.steps?.length || trace?.tool_names?.length || trace?.stage || trace?.flow)
}

function formatToolTraceSource(step: ToolTraceStep) {
  switch (step.type) {
    case 'user_input':
      return 'User'
    case 'assistant_tool_call':
    case 'assistant_final_response':
      return 'LLM'
    case 'client_tool_result':
      return 'Agent'
    default:
      return formatToolTraceSourceFallback(step.source)
  }
}

function formatToolTraceSourceFallback(source?: string) {
  const normalized = source?.trim().toLowerCase()
  if (!normalized) {
    return null
  }

  switch (normalized) {
    case 'request':
      return 'Request'
    case 'response':
      return 'Response'
    case 'stream':
      return 'Stream'
    default:
      return source
  }
}

function toolTraceTintStyle(stepType: string): CSSProperties {
  const visualStyle = TOOL_TRACE_STYLES[stepType]
  return {
    '--tool-trace-tint': visualStyle?.tint || 'rgba(255, 255, 255, 0.08)',
    '--tool-trace-accent': visualStyle?.accent || 'rgba(255, 255, 255, 0.9)',
  } as CSSProperties
}

function formatTraceBlock(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

function renderToolTraceContent(step: ToolTraceStep) {
  if (step.type === 'assistant_final_response' && step.text) {
    return (
      <div className={styles.toolTraceMarkdown}>
        <MarkdownRenderer content={step.text} />
      </div>
    )
  }

  return <pre className={styles.toolTraceBlock}>{formatTraceBlock(step.text || '')}</pre>
}

function renderToolTraceStepIcon(stepType: string) {
  switch (stepType) {
    case 'user_input':
      return (
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="M10 10.5a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5Z" />
          <path d="M4 16c1.2-2.35 3.18-3.52 6-3.52 2.82 0 4.8 1.17 6 3.52" strokeLinecap="round" />
        </svg>
      )
    case 'assistant_tool_call':
      return (
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="m7 6-3 4 3 4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="m13 6 3 4-3 4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M11 4 9 16" strokeLinecap="round" />
        </svg>
      )
    case 'client_tool_result':
      return (
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="M12.5 4.5a3 3 0 0 1 3 3c0 .53-.14 1.02-.38 1.44l-5.68 5.68a2 2 0 0 1-2.83 0l-1.25-1.25a2 2 0 0 1 0-2.83l5.68-5.68c.42-.24.91-.36 1.46-.36Z" />
          <path d="m10.4 6.9 2.7 2.7" strokeLinecap="round" />
        </svg>
      )
    case 'assistant_final_response':
      return (
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="M4.5 5.5h11a1.5 1.5 0 0 1 1.5 1.5v6a1.5 1.5 0 0 1-1.5 1.5H9l-3.5 2v-2H4.5A1.5 1.5 0 0 1 3 13V7a1.5 1.5 0 0 1 1.5-1.5Z" strokeLinejoin="round" />
          <path d="M6.5 9h7M6.5 11.5h5" strokeLinecap="round" />
        </svg>
      )
    default:
      return (
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <circle cx="10" cy="10" r="5.5" />
        </svg>
      )
  }
}
