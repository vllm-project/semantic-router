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
