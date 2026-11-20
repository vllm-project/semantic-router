import type {
  DecisionConfig,
  DecisionCondition,
  DecisionFormState,
  DecisionPluginConfiguration,
  DecisionRuleSet,
} from './configPageSupport'

function validateCondition(condition: DecisionCondition, path: string): void {
  const hasChildren = Boolean(condition.operator || condition.conditions?.length)
  if (hasChildren) {
    if (!condition.operator || !['AND', 'OR', 'NOT'].includes(condition.operator)) {
      throw new Error(`${path} needs an AND, OR, or NOT operator.`)
    }
    const children = condition.conditions || []
    if (condition.operator === 'NOT' && children.length !== 1) {
      throw new Error(`${path} NOT group needs exactly one condition.`)
    }
    if (children.length === 0) throw new Error(`${path} needs at least one condition.`)
    children.forEach((child, index) => validateCondition(child, `${path}.${index + 1}`))
    return
  }

  if (!condition.type?.trim() || !condition.name?.trim()) {
    throw new Error(`${path} needs both type and name.`)
  }
}

function conditionUsesOnError(condition: DecisionCondition): boolean {
  return Boolean(
    condition.on_error || condition.conditions?.some((child) => conditionUsesOnError(child)),
  )
}

// The Router accepts one leaf condition as the whole rule tree: `rules: {type, name}`.
export const isSingleConditionRules = (rules: DecisionRuleSet | undefined): boolean =>
  Boolean(
    rules &&
      !rules.operator &&
      !rules.conditions?.length &&
      (rules.type || rules.name || rules.label || rules.predicate || rules.on_error),
  )

export const decisionRulesForSave = (
  value: DecisionFormState['rules'],
): DecisionConfig['rules'] => {
  const rules = JSON.parse(JSON.stringify(value || {})) as DecisionConfig['rules']
  const conditions = rules.conditions || []
  if (isSingleConditionRules(rules)) {
    validateCondition(rules, 'The root condition')
    if (rules.on_unknown && conditionUsesOnError(rules)) {
      throw new Error('Rules on_unknown cannot be combined with condition on_error.')
    }
    return rules
  }
  if (!rules.operator && conditions.length === 0) return {}
  if (!rules.operator || !['AND', 'OR', 'NOT'].includes(rules.operator)) {
    throw new Error('Rules need an AND, OR, or NOT root operator.')
  }
  if (rules.operator === 'NOT' && conditions.length !== 1) {
    throw new Error('The root NOT group needs exactly one condition.')
  }
  if (rules.operator !== 'AND' && conditions.length === 0) {
    throw new Error(`The root ${rules.operator} group needs at least one condition.`)
  }
  if (rules.on_error) throw new Error('on_error only applies to leaf conditions.')
  conditions.forEach((condition, index) => validateCondition(condition, `Condition #${index + 1}`))
  if (rules.on_unknown && conditions.some((condition) => conditionUsesOnError(condition))) {
    throw new Error('Rules on_unknown cannot be combined with condition on_error.')
  }
  return rules
}

export const decisionModelRefsForSave = (
  values: DecisionFormState['modelRefs'],
): DecisionConfig['modelRefs'] =>
  (values || [])
    .filter((value) => (value?.model || '').trim())
    .map((value, index) => {
      const model = (value?.model || '').trim()
      if (!model) {
        throw new Error(`Model reference #${index + 1} is missing a model name.`)
      }
      return {
        model,
        use_reasoning: !!value?.use_reasoning,
        ...optionalText('reasoning_description', value?.reasoning_description),
        ...optionalText('reasoning_mode', value?.reasoning_mode),
        ...optionalText('reasoning_effort', value?.reasoning_effort),
        ...optionalText('lora_name', value?.lora_name),
        ...(typeof value?.weight === 'number' && Number.isFinite(value.weight)
          ? { weight: value.weight }
          : {}),
      } as DecisionConfig['modelRefs'][number]
    })

const optionalText = <Key extends string>(
  key: Key,
  value?: string,
): Partial<Record<Key, string>> => {
  const normalized = (value || '').trim()
  return normalized ? ({ [key]: normalized } as Partial<Record<Key, string>>) : {}
}

export const decisionPluginsForSave = (
  values: DecisionFormState['plugins'],
): NonNullable<DecisionConfig['plugins']> =>
  (values || [])
    .filter((plugin) => {
      const hasType = (plugin?.type || '').trim()
      const hasConfigString =
        typeof plugin?.configuration === 'string' && plugin.configuration.trim()
      const hasConfigObject = plugin?.configuration && typeof plugin.configuration === 'object'
      return Boolean(hasType || hasConfigString || hasConfigObject)
    })
    .map((plugin, index) => ({
      type: requiredPluginType(plugin?.type, index),
      configuration: parsePluginConfiguration(plugin?.configuration, index),
    }))

const requiredPluginType = (value: string | undefined, index: number): string => {
  const type = (value || '').trim()
  if (!type) throw new Error(`Plugin #${index + 1} must include a type.`)
  return type
}

const parsePluginConfiguration = (
  value: DecisionFormState['plugins'][number]['configuration'],
  index: number,
): DecisionPluginConfiguration => {
  if (typeof value !== 'string') {
    return value && typeof value === 'object' ? (value as DecisionPluginConfiguration) : {}
  }
  const normalized = value.trim()
  if (!normalized) return {}
  try {
    return JSON.parse(normalized)
  } catch {
    throw new Error(`Plugin #${index + 1} configuration must be valid JSON.`)
  }
}

// Drawer editing works on a copy of the decision as written: fields the
// editor does not touch keep their exact values, and saving only normalizes
// what the old form normalized (trimmed names, dropped empty blocks).

export type DecisionTab = 'overview' | 'rules' | 'models' | 'algorithm' | 'plugins' | 'advanced'

export class DecisionDraftError extends Error {
  constructor(
    message: string,
    readonly tab: DecisionTab,
  ) {
    super(message)
  }
}

const OPTIONAL_BLOCKS = [
  'action',
  'adaptations',
  'reliability',
  'fallback',
  'output_contract_spec',
  'retention',
  'annotations',
] as const

const MODEL_REF_TEXT_FIELDS = [
  'lora_name',
  'reasoning_description',
  'reasoning_mode',
  'reasoning_effort',
] as const

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export function newDecisionDraft(): DecisionConfig {
  return { name: '', description: '', priority: 1, rules: {}, modelRefs: [] }
}

export function cloneDecision(decision: DecisionConfig): DecisionConfig {
  return JSON.parse(JSON.stringify(decision)) as DecisionConfig
}

/** Model references as saved: a model each, trimmed text, no empty optional values. */
export function decisionModelRefsFromDraft(
  refs: DecisionConfig['modelRefs'] | undefined,
): DecisionConfig['modelRefs'] {
  return (refs ?? []).flatMap((ref) => {
    const model = typeof ref?.model === 'string' ? ref.model.trim() : ''
    if (!model) return []
    const next: Record<string, unknown> = { ...ref, model }
    for (const key of MODEL_REF_TEXT_FIELDS) {
      const value = typeof next[key] === 'string' ? (next[key] as string).trim() : next[key]
      if (value === '' || value === undefined) delete next[key]
      else next[key] = value
    }
    if (
      next.weight !== undefined &&
      !(typeof next.weight === 'number' && Number.isFinite(next.weight))
    )
      delete next.weight
    return [next as unknown as DecisionConfig['modelRefs'][number]]
  })
}

/** The decision a draft saves as, or the first problem with the tab it belongs to. */
export function decisionFromDraft(
  draft: DecisionConfig,
  validate: {
    algorithmErrors?: (algorithm: Record<string, unknown>) => string[]
    actionErrors?: (action: Record<string, unknown>) => string[]
    declarativeErrors?: (decision: DecisionConfig) => string[]
  } = {},
): DecisionConfig {
  const next = cloneDecision(draft) as DecisionConfig & Record<string, unknown>
  next.name = (next.name || '').trim()
  if (!next.name) throw new DecisionDraftError('Name is required.', 'overview')
  try {
    next.rules = decisionRulesForSave(next.rules)
  } catch (error) {
    throw new DecisionDraftError(error instanceof Error ? error.message : String(error), 'rules')
  }
  next.modelRefs = decisionModelRefsFromDraft(next.modelRefs)
  if (next.plugins !== undefined) {
    try {
      next.plugins = decisionPluginsForSave(next.plugins as DecisionFormState['plugins'])
    } catch (error) {
      throw new DecisionDraftError(
        error instanceof Error ? error.message : String(error),
        'plugins',
      )
    }
  }
  for (const key of OPTIONAL_BLOCKS) {
    if (isPlainObject(next[key]) && Object.keys(next[key] as object).length === 0) delete next[key]
  }
  if (Array.isArray(next.candidateIterations) && next.candidateIterations.length === 0)
    delete next.candidateIterations
  if (typeof next.output_contract === 'string') {
    const contract = next.output_contract.trim()
    if (contract) next.output_contract = contract
    else delete next.output_contract
  }
  for (const key of ['priority', 'tier'] as const) {
    if (next[key] !== undefined && !(typeof next[key] === 'number' && Number.isFinite(next[key])))
      delete next[key]
  }
  if (next.output !== undefined && !isPlainObject(next.output)) delete next.output
  if (isPlainObject(next.output) && Object.keys(next.output).length === 0) delete next.output
  if (isPlainObject(next.action)) {
    const problem = validate.actionErrors?.(next.action)[0]
    if (problem) throw new DecisionDraftError(problem, 'models')
  }
  if (isPlainObject(next.algorithm)) {
    const problem = validate.algorithmErrors?.(next.algorithm)[0]
    if (problem) throw new DecisionDraftError(problem, 'algorithm')
  }
  const declarative = validate.declarativeErrors?.(next)[0]
  if (declarative) throw new DecisionDraftError(declarative, 'advanced')
  return next
}

/** Writes a decision into the profile's list in place; a new or renamed-onto-free name is appended. */
export function upsertDecision(
  decisions: readonly DecisionConfig[],
  decision: DecisionConfig,
  originalName?: string,
): DecisionConfig[] {
  const clash = decisions.find(
    (entry) => entry.name === decision.name && entry.name !== originalName,
  )
  if (clash)
    throw new DecisionDraftError(`A decision named “${decision.name}” already exists.`, 'overview')
  const index =
    originalName === undefined ? -1 : decisions.findIndex((entry) => entry.name === originalName)
  if (index < 0) return [...decisions, decision]
  return decisions.map((entry, position) => (position === index ? decision : entry))
}
