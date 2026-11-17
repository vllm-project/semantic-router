import type { SystemOneSignal } from '../types/config'
import runtimeCatalog from './decisionRuntimeCatalog.generated.json'

export type DecisionQuestionKind = SystemOneSignal['question']['type']
export interface DecisionRuntimeCatalogEntry {
  id: string
  name: string
  provider: string
  /** The model runtime's registry family, such as vela3_d3. */
  family: string
  /** The Supported models pages' family label, such as Vela 3.0. */
  familyLabel: string
  /** Empty for an announced model the registry does not pin yet. */
  revision: string
  backbone: string
  /** Null when no registry table or announcement states it. */
  minMemoryGiB: number | null
  parameters: number | null
  /** What the Router's decision catalog says the model answers; empty when it serves none. */
  questionTypes: readonly string[]
  modalities: readonly string[]
  spanHeads: readonly string[]
  summary: string
  /** The pull request that adds runtime support for an announced model; empty once registered. */
  pending: string
}

// Generated from model-runtime's canonical registry; never add release pins here.
export const DECISION_RUNTIME_CATALOG: readonly DecisionRuntimeCatalogEntry[] =
  runtimeCatalog.models
export const DECISION_RUNTIME_CAPABILITIES =
  runtimeCatalog.questionTypes as readonly DecisionQuestionKind[]

// Presentation metadata is deliberately independent of deployable model identity.
export const DECISION_PROVIDERS: Record<string, { name: string; logo: string }> = {
  'vllm-sr': { name: 'vLLM Semantic Router', logo: 'vllm' },
}
