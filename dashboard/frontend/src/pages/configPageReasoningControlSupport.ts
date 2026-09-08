import type { DecisionModelRef, NormalizedModel, ReasoningFamily } from './configPageSupport'

export function reasoningFamilyForModel(
  model: NormalizedModel | undefined,
  families: Record<string, ReasoningFamily>,
): ReasoningFamily | undefined {
  const inline = model?.reasoning
  const familyName = inline?.family || model?.reasoning_family || ''
  const family = familyName
    ? families[familyName]
    : inline?.type && inline.parameter
      ? ({ ...inline, type: inline.type, parameter: inline.parameter } satisfies ReasoningFamily)
      : undefined
  if (!family) return undefined
  return {
    ...family,
    ...(model?.reasoning_modes ? { modes: model.reasoning_modes } : {}),
    ...(model?.reasoning_efforts ? { levels: model.reasoning_efforts } : {}),
  }
}

export function reasoningFamilyCanDisable(family: ReasoningFamily | undefined): boolean {
  return Boolean(
    family &&
      (family.activation_parameter ||
        family.disabled ||
        family.modes?.includes('disabled')),
  )
}

export function reasoningFamilyIsAlwaysOn(family: ReasoningFamily | undefined): boolean {
  return Boolean(family) && !reasoningFamilyCanDisable(family)
}

export function defaultReasoningEnabled(family: ReasoningFamily | undefined): boolean {
  if (!family) return false
  if (family.default_mode) return family.default_mode !== 'disabled'
  return reasoningFamilyIsAlwaysOn(family)
}

export function modelSelectionReasoningState(
  family: ReasoningFamily | undefined,
): Pick<DecisionModelRef, 'use_reasoning' | 'reasoning_mode' | 'reasoning_effort'> {
  return {
    use_reasoning: defaultReasoningEnabled(family),
    reasoning_mode: '',
    reasoning_effort: '',
  }
}
