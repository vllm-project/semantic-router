import type {
  DecisionConfig,
  DecisionFormState,
  DecisionPluginConfiguration,
} from './configPageSupport'

export const decisionConditionsForSave = (
  values: DecisionFormState['conditions'],
): NonNullable<DecisionConfig['rules']>['conditions'] =>
  (values || [])
    .filter((condition) => (condition?.type || '').trim() || (condition?.name || '').trim())
    .map((condition, index) => {
      const type = (condition?.type || '').trim()
      const name = (condition?.name || '').trim()
      if (!type || !name) {
        throw new Error(`Condition #${index + 1} needs both type and name.`)
      }
      return {
        type,
        name,
        ...(condition.label ? { label: condition.label } : {}),
        ...(condition.predicate ? { predicate: condition.predicate } : {}),
        ...(condition.on_error ? { on_error: condition.on_error } : {}),
      }
    })

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
