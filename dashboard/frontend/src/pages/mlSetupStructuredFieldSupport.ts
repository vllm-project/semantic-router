import type { DecisionEntry } from '../types/mlPipeline'

export interface HiddenLayerEntry {
  size?: number
}

function duplicateValue(values: readonly string[]): string | null {
  const seen = new Set<string>()
  for (const value of values) {
    const normalized = value.trim().toLocaleLowerCase()
    if (normalized && seen.has(normalized)) return value.trim()
    if (normalized) seen.add(normalized)
  }
  return null
}

export function parseMlpHiddenLayers(value: string): HiddenLayerEntry[] {
  if (!value.trim()) return []
  return value.split(',').map((item) => {
    const trimmed = item.trim()
    if (!trimmed) return {}
    const size = Number(trimmed)
    return { size: Number.isFinite(size) ? size : undefined }
  })
}

export function serializeMlpHiddenLayers(value: readonly HiddenLayerEntry[]): string {
  return value.map((item) => (item.size === undefined ? '' : String(item.size))).join(',')
}

export function getMlpHiddenLayersError(value: string): string | null {
  if (!value.trim()) return 'Add at least one hidden layer before training the MLP.'

  const entries = value.split(',').map((item) => item.trim())
  const invalidIndex = entries.findIndex((item) => {
    const size = Number(item)
    return !Number.isInteger(size) || size <= 0
  })
  return invalidIndex >= 0
    ? `Hidden layer ${invalidIndex + 1} must be a positive integer.`
    : null
}

export function getDecisionEntriesError(decisions: readonly DecisionEntry[]): string | null {
  if (decisions.length === 0) return 'Add at least one routing decision.'

  const names = decisions.map((decision) => decision.name.trim())
  const duplicateName = duplicateValue(names)
  if (duplicateName) return `Decision name ${duplicateName} is duplicated.`

  for (const [index, decision] of decisions.entries()) {
    const label = decision.name.trim() || `Decision ${index + 1}`
    if (!decision.name.trim()) return `Decision ${index + 1} requires a name.`
    if (!Array.isArray(decision.domains) || decision.domains.some((domain) => !domain.trim())) {
      return `${label} domains must be non-empty text values.`
    }
    const duplicateDomain = duplicateValue(decision.domains)
    if (duplicateDomain) return `${label} domain ${duplicateDomain} is duplicated.`
    if (!Array.isArray(decision.model_names) || decision.model_names.length === 0) {
      return `${label} requires at least one model reference.`
    }
    if (decision.model_names.some((model) => !model.trim())) {
      return `${label} model references must be non-empty text values.`
    }
    const duplicateModel = duplicateValue(decision.model_names)
    if (duplicateModel) return `${label} model ${duplicateModel} is duplicated.`
  }
  return null
}
