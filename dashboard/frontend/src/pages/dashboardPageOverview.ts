import type { DecisionRule } from './dashboardPageTypes'
import { getDecisionCategory, SIGNAL_COLORS } from './dashboardPageStats'

export interface SignalBreakdownRow {
  type: string
  count: number
  percent: number
  color: string
}

export interface DecisionPreviewRow {
  key: string
  name: string
  title: string
  priorityLabel: number | string
  category: ReturnType<typeof getDecisionCategory>
  typeLabel: string
  modelNames: string
}

export function buildSignalBreakdownRows(byType: Record<string, number>): SignalBreakdownRow[] {
  const entries = Object.entries(byType).filter(([, count]) => count > 0)
  if (entries.length === 0) return []

  const maxCount = Math.max(...entries.map(([, count]) => count))
  return entries
    .sort((a, b) => b[1] - a[1])
    .map(([type, count]) => ({
      type,
      count,
      percent: Math.round((count / maxCount) * 100),
      color: SIGNAL_COLORS[type] || '#999',
    }))
}

export function buildDecisionPreviewRows(
  decisions: readonly DecisionRule[],
  limit = 10,
): DecisionPreviewRow[] {
  return decisions.slice(0, limit).map((decision, index) => {
    const category = getDecisionCategory(decision.priority)
    const name = decision.name || `Decision ${index + 1}`
    return {
      key: `${name}-${index}`,
      name,
      title: decision.description || decision.name || '',
      priorityLabel: decision.priority ?? '—',
      category,
      typeLabel: category === 'guardrail' ? 'Guard' : category === 'fallback' ? 'Default' : 'Route',
      modelNames: formatDecisionModelNames(decision.modelRefs),
    }
  })
}

function formatDecisionModelNames(modelRefs: unknown): string {
  if (!Array.isArray(modelRefs)) return '—'
  return modelRefs.map(modelRefName).filter(Boolean).join(', ')
}

function modelRefName(modelRef: unknown): string {
  if (!modelRef || typeof modelRef !== 'object') return ''
  const model = (modelRef as { model?: unknown }).model
  return typeof model === 'string' ? model : ''
}
