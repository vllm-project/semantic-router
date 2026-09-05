export interface EvaluationConditionModel {
  reasoning_family?: string
}

const publishedConditionLabels: Record<string, string> = {
  unspecified: 'Effort not reported',
  default: 'Published default',
  enabled: 'Reasoning enabled',
  disabled: 'Non-reasoning run',
  adaptive: 'Adaptive reasoning',
  none: 'No reasoning',
  no_think: 'No reasoning',
}

const readable = (value: string): string => value.replace(/_/g, ' ')

export function modelHubEvaluationConditionLabel(
  model: EvaluationConditionModel,
  condition: string,
): string {
  const readableCondition = readable(condition.trim())
  if (model.reasoning_family) return `${readableCondition} effort`
  const publishedLabel = publishedConditionLabels[condition.trim()]
  if (publishedLabel) return publishedLabel
  if (!readableCondition) return publishedConditionLabels.unspecified
  return `${readableCondition.charAt(0).toLocaleUpperCase()}${readableCondition.slice(1)} run`
}
