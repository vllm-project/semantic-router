import type { DecisionRule } from './dashboardPageTypes'
import { getDecisionCategory } from './dashboardPageStats'

export interface DecisionPreviewRow {
  key: string
  name: string
  title: string
  priorityLabel: number | string
  category: ReturnType<typeof getDecisionCategory>
  typeLabel: string
  modelNames: string
  scopeLabel?: string
}

export function buildDecisionPreviewRows(
  decisions: readonly DecisionRule[],
  limit = 10,
): DecisionPreviewRow[] {
  return decisions.slice(0, limit).map((decision, index) => {
    const category = getDecisionCategory(decision.priority)
    const name = decision.name || `Decision ${index + 1}`
    const scopeLabel = decision.routingEntrypoints?.[0] ?? decision.routingScope
    return {
      key: `${name}-${index}`,
      name,
      title: decision.description || decision.name || '',
      priorityLabel: decision.priority ?? '—',
      category,
      typeLabel: category === 'guardrail' ? 'Guard' : category === 'fallback' ? 'Default' : 'Route',
      modelNames: formatDecisionModelNames(decision.modelRefs),
      ...(scopeLabel ? { scopeLabel } : {}),
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
