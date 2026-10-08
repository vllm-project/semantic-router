import type { DecisionModelSignal } from '../types/config'
import runtimeCatalog from './decisionRuntimeCatalog.generated.json'

export type DecisionQuestionKind = DecisionModelSignal['question']['type']
export interface DecisionRuntimeCatalogEntry {
  id: string
  name: string
  provider: string
  family: string
  revision: string
  backbone: string
  minMemoryGiB: number
}

// Generated from model-runtime's canonical registry; never add release pins here.
export const DECISION_RUNTIME_CATALOG: readonly DecisionRuntimeCatalogEntry[] =
  runtimeCatalog.models
export const DECISION_RUNTIME_CAPABILITIES =
  runtimeCatalog.questionTypes as readonly DecisionQuestionKind[]

// Presentation metadata is deliberately independent of deployable model identity.
export const DECISION_PROVIDERS: Record<string, { name: string; logo: string }> = {
  'vllm-sr': { name: 'vLLM Semantic Router', logo: '/vllm.png' },
}

export function runtimeFamilyLabel(entry: DecisionRuntimeCatalogEntry): string {
  return (
    ({ decision2: 'Decision 2.0', decision1: 'Decision 1.0' } as Record<string, string>)[
      entry.family
    ] ?? entry.family
  )
}

export function runtimeBackboneLabel(entry: DecisionRuntimeCatalogEntry): string {
  return (
    (
      { modernbert: 'ModernBERT', qwen3: 'Qwen3', qwen3_5_text: 'Qwen3.5' } as Record<
        string,
        string
      >
    )[entry.backbone] ?? entry.backbone
  )
}

export function runtimeDeploymentName(entry: DecisionRuntimeCatalogEntry): string {
  return entry.name.toLowerCase().replace(/\./g, '-')
}
