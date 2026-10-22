import { canonicalToRuleDocument, type RuleIssue } from '../lib/decisionRuleAst'
import {
  RAW_EXPRESSION_INCOMPLETE,
  blockingRuleIssues,
  formatRuleIssuePath,
  validateRuleDocument,
  type RuleSignalScope,
} from '../lib/decisionRuleValidation'
import type {
  DecisionConfig,
  DecisionFormState,
  DecisionPluginConfiguration,
  DecisionRuleSet,
} from './configPageSupport'

// The Router accepts one leaf condition as the whole rule tree: `rules: {type, name}`.
export const isSingleConditionRules = (rules: DecisionRuleSet | undefined): boolean =>
  Boolean(
    rules &&
      !rules.operator &&
      !rules.conditions?.length &&
      (rules.type || rules.name || rules.label || rules.predicate || rules.on_error),
  )

/**
 * Validates the rule tree against the selected routing scope's signals with
 * the same checks the Go config validator runs, and blocks the save on any
 * error. `editorIssues` carries what only the editor knows, such as an
 * incomplete raw expression that never reached the tree.
 */
export const decisionRulesForSave = (
  value: DecisionFormState['rules'],
  scope: RuleSignalScope,
  editorIssues: RuleIssue[] = [],
): DecisionConfig['rules'] => {
  const rules = JSON.parse(JSON.stringify(value || {})) as DecisionConfig['rules']
  const issues = [
    ...editorIssues.filter((issue) => issue.message.startsWith(RAW_EXPRESSION_INCOMPLETE)),
    ...validateRuleDocument(canonicalToRuleDocument(rules), scope),
  ]
  const messages = Array.from(
    new Set(
      blockingRuleIssues(issues).map((issue) =>
        issue.path.length > 0 ? `${formatRuleIssuePath(issue.path)}: ${issue.message}` : issue.message,
      ),
    ),
  )
  if (messages.length > 0) {
    throw new Error(`Fix the rule tree before saving: ${messages.join('; ')}`)
  }
  return rules
}

const MODELLED_MODEL_REF_KEYS = new Set([
  'model',
  'use_reasoning',
  'reasoning_description',
  'reasoning_mode',
  'reasoning_effort',
  'lora_name',
  'weight',
])

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
      // Keys the form does not model survive; the modelled optional text
      // fields are rebuilt so a cleared field is dropped, not saved empty.
      const unmodelled = Object.fromEntries(
        Object.entries(value).filter(([key]) => !MODELLED_MODEL_REF_KEYS.has(key)),
      )
      return {
        ...unmodelled,
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
