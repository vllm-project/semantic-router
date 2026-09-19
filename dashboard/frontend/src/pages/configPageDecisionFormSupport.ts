import type {
  DecisionConfig,
  DecisionCondition,
  DecisionFormState,
  DecisionPluginConfiguration,
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

export const decisionRulesForSave = (
  value: DecisionFormState['rules'],
): DecisionConfig['rules'] => {
  const rules = JSON.parse(JSON.stringify(value || {})) as DecisionConfig['rules']
  const conditions = rules.conditions || []
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
  conditions.forEach((condition, index) => validateCondition(condition, `Condition #${index + 1}`))
  if (rules.on_unknown && conditions.some((condition) => conditionUsesOnError(condition))) {
    throw new Error('Rules on_unknown cannot be combined with condition on_error.')
  }
  return rules
}

export const decisionModelRefsForForm = (
  values: DecisionConfig['modelRefs'] | undefined,
): DecisionFormState['modelRefs'] =>
  (values || []).map((ref) => ({
    model: ref.model || '',
    use_reasoning: !!ref.use_reasoning,
    reasoning_description: ref.reasoning_description || '',
    reasoning_mode: ref.reasoning_mode || '',
    reasoning_effort: ref.reasoning_effort || '',
    lora_name: ref.lora_name || '',
    ...(typeof ref.weight === 'number' && Number.isFinite(ref.weight) ? { weight: ref.weight } : {}),
    ...optionalLoadedNumber('max_completion_tokens', ref.max_completion_tokens),
  }))

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
        ...optionalPositiveInt(
          'max_completion_tokens',
          value?.max_completion_tokens,
          `Model reference #${index + 1}`,
        ),
      } as DecisionConfig['modelRefs'][number]
    })

const optionalText = <Key extends string>(
  key: Key,
  value?: string,
): Partial<Record<Key, string>> => {
  const normalized = (value || '').trim()
  return normalized ? ({ [key]: normalized } as Partial<Record<Key, string>>) : {}
}

const optionalLoadedNumber = <Key extends string>(
  key: Key,
  value: unknown,
): Partial<Record<Key, number>> =>
  typeof value === 'number' && Number.isFinite(value)
    ? ({ [key]: value } as Partial<Record<Key, number>>)
    : {}

const optionalPositiveInt = <Key extends string>(
  key: Key,
  value: unknown,
  path: string,
): Partial<Record<Key, number>> => {
  if (value === undefined || value === null || value === '') {
    return {}
  }
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${path} ${key} must be a finite integer >= 1.`)
  }
  return { [key]: value } as Partial<Record<Key, number>>
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
