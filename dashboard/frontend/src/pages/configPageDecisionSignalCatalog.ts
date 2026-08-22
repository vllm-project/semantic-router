import type { SignalDescriptor } from '../components/ExpressionBuilderSupport'
import type {
  ConfigDecisionConditionType,
  ConfigProjections,
  ConfigSignals,
} from './configPageSupport'

// Every decision condition type that resolves to a concrete signal catalog. `conversation`
// and `event` are valid backend condition types but have no corresponding named-signal source
// in ConfigSignals yet, so they're intentionally left out of pickers until that catalog exists.
export const DECISION_CONDITION_TYPES: readonly ConfigDecisionConditionType[] = [
  'keyword',
  'domain',
  'preference',
  'user_feedback',
  'reask',
  'embedding',
  'fact_check',
  'language',
  'context',
  'structure',
  'complexity',
  'modality',
  'authz',
  'jailbreak',
  'pii',
  'kb',
  'metadata',
  'classifier',
  'projection',
]

export function getDecisionConditionNameOptions(
  type: ConfigDecisionConditionType | undefined,
  signals: ConfigSignals | undefined,
  projections?: ConfigProjections,
): string[] {
  switch (type) {
    case 'keyword':
      return signals?.keywords?.map((s) => s.name) || []
    case 'domain':
      return signals?.domains?.map((s) => s.name) || []
    case 'preference':
      return signals?.preferences?.map((s) => s.name) || []
    case 'user_feedback':
      return signals?.user_feedbacks?.map((s) => s.name) || []
    case 'reask':
      return signals?.reasks?.map((s) => s.name) || []
    case 'embedding':
      return signals?.embeddings?.map((s) => s.name) || []
    case 'fact_check':
      return signals?.fact_check?.map((s) => s.name) || []
    case 'language':
      return signals?.language?.map((s) => s.name) || []
    case 'context':
      return signals?.context?.map((s) => s.name) || []
    case 'structure':
      return signals?.structure?.map((s) => s.name) || []
    case 'complexity':
      return (signals?.complexity || []).flatMap((s) => [
        `${s.name}:easy`,
        `${s.name}:medium`,
        `${s.name}:hard`,
      ])
    case 'modality':
      return signals?.modality?.map((s) => s.name) || []
    case 'authz':
      return signals?.role_bindings?.map((s) => s.name) || []
    case 'jailbreak':
      return signals?.jailbreak?.map((s) => s.name) || []
    case 'pii':
      return signals?.pii?.map((s) => s.name) || []
    case 'kb':
      return signals?.kb?.map((s) => s.name) || []
    case 'metadata':
      return signals?.metadata?.map((s) => s.name) || []
    case 'classifier':
      return signals?.classifiers?.map((s) => s.name) || []
    case 'projection':
      return (projections?.mappings || []).flatMap((mapping) =>
        (mapping.outputs || []).map((output) => output.name),
      )
    default:
      return []
  }
}

// Flattens every configured signal into the {signalType, name} pairs ExpressionBuilder
// (and DecisionRuleEditor) use for autocomplete and reference validation.
export function buildAvailableSignals(
  signals: ConfigSignals | undefined,
  projections?: ConfigProjections,
): SignalDescriptor[] {
  return DECISION_CONDITION_TYPES.flatMap((type) =>
    getDecisionConditionNameOptions(type, signals, projections).map((name) => ({
      signalType: type,
      name,
    })),
  )
}
