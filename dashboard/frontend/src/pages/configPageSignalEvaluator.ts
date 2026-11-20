import { bindingDeployment, isSignalBindingKey } from './modelBindings'
import type { DecisionTasks } from './useDecisionTasks'

export interface SignalEvaluator {
  /** model: a deployment answers it; rule: the Router evaluates it itself. */
  kind: 'model' | 'rule'
  deployment?: string
  /** Where the deployment comes from. */
  source?: 'rule' | 'recipe binding' | 'global binding' | 'default binding' | 'decision model'
  /** How a keyword rule matches. */
  method?: string
}

export interface SignalEvaluatorContext {
  tasks?: DecisionTasks | null
  recipeBindings?: Record<string, unknown>
  globalBindings?: Record<string, unknown>
  decisionModel?: string
}

/** Which deployment answers a rule, following the Router's binding order. */
export function signalEvaluator(
  type: string,
  record: Record<string, unknown>,
  context: SignalEvaluatorContext,
): SignalEvaluator {
  if (type === 'keyword') {
    const method = typeof record.method === 'string' && record.method ? record.method : 'literal'
    return {
      kind: 'rule',
      method: record.fuzzy_match === true && method === 'literal' ? 'fuzzy' : method,
    }
  }
  const own = bindingDeployment(record.deployment)
  if (own) return { kind: 'model', deployment: own, source: 'rule' }
  if (type === 'systemone') {
    const deployment = context.decisionModel || context.tasks?.default_deployment
    return { kind: 'model', deployment, source: 'decision model' }
  }
  if (!isSignalBindingKey(type)) return { kind: 'rule' }
  const recipe = bindingDeployment(context.recipeBindings?.[type])
  if (recipe) return { kind: 'model', deployment: recipe, source: 'recipe binding' }
  const global = bindingDeployment(context.globalBindings?.[type])
  if (global) return { kind: 'model', deployment: global, source: 'global binding' }
  const fallback =
    bindingDeployment(context.tasks?.default_bindings?.[type]) || context.decisionModel
  return { kind: 'model', deployment: fallback, source: 'default binding' }
}

export function signalEvaluatorLabel(evaluator: SignalEvaluator): string {
  if (evaluator.kind === 'rule') return evaluator.method ?? 'built-in'
  return evaluator.deployment ?? 'decision model'
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** RE2 inline flags at the start of a pattern, which JavaScript only takes as RegExp flags. */
function splitInlineFlags(pattern: string): { body: string; flags: string } {
  const match = pattern.match(/^\(\?([imsU]+)\)/)
  if (!match) return { body: pattern, flags: '' }
  return { body: pattern.slice(match[0].length), flags: match[1].replace(/U/g, '') }
}

/**
 * Where a keyword rule's literal keywords or regular expressions occur in a
 * prompt, for the try-it preview. Scored methods (bm25, ngram, fuzzy) do not
 * locate matches, so they report null.
 */
export function keywordOccurrences(
  text: string,
  record: Record<string, unknown>,
): Array<{ start: number; end: number }> | null {
  const method = typeof record.method === 'string' ? record.method : ''
  if (method === 'bm25' || method === 'ngram' || method === 'fuzzy' || record.fuzzy_match === true)
    return null
  const keywords = Array.isArray(record.keywords)
    ? record.keywords.filter((item): item is string => typeof item === 'string' && item !== '')
    : []
  const ranges: Array<{ start: number; end: number }> = []
  for (const keyword of keywords) {
    const { body, flags } =
      method === 'regex'
        ? splitInlineFlags(keyword)
        : { body: `\\b${escapeRegExp(keyword)}\\b`, flags: '' }
    const flagSet = new Set(['g', ...flags])
    if (record.case_sensitive !== true) flagSet.add('i')
    let pattern: RegExp
    try {
      pattern = new RegExp(body, [...flagSet].join(''))
    } catch {
      continue
    }
    for (const match of text.matchAll(pattern)) {
      if (match[0].length > 0 && match.index !== undefined)
        ranges.push({ start: match.index, end: match.index + match[0].length })
    }
  }
  return ranges
}

/** The score threshold a rule matches at, when it states one. */
export function signalThreshold(record: Record<string, unknown>): number | undefined {
  const predicate = record.predicate
  if (predicate && typeof predicate === 'object') {
    const bounds = predicate as Record<string, unknown>
    const value = bounds.gte ?? bounds.gt
    if (typeof value === 'number') return value
  }
  return typeof record.threshold === 'number' ? record.threshold : undefined
}
