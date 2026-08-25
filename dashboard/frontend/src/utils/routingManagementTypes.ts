export type RoutingStatus = 'draft' | 'active' | 'disabled'
export type DispatchCardinality = 'single' | 'multi'
export type FallbackTrigger = 'unavailable' | 'overloaded' | 'timeout'

export interface RoutingDecision {
  id: string
  name: string
  dispatchCardinality: DispatchCardinality
}

export interface RoutingAssignmentReasoning {
  enabled: boolean
  effort?: string
  description?: string
}

export interface RoutingAssignmentModel {
  modelId: string
  modelRevision: number
  priority: number
  weight: string
  loraName?: string
  reasoning?: RoutingAssignmentReasoning
}

export interface RoutingFallbackPolicy {
  strategy: 'priority'
  on: FallbackTrigger[]
}

export interface RoutingAssignmentSet {
  models: RoutingAssignmentModel[]
  fallback?: RoutingFallbackPolicy
}

export interface RoutingAssignmentModelWrite {
  modelId: string
  priority?: number
  weight?: string
  loraName?: string
  reasoning?: RoutingAssignmentReasoning
}

export interface RoutingAssignmentSetWrite {
  models: RoutingAssignmentModelWrite[]
  fallback?: RoutingFallbackPolicy
}

export interface RoutingMatcher {
  claim?: { name: string; value: RoutingClaimValue }
  exactPath?: string
  pathPrefix?: string
}

export type RoutingClaimValue =
  | { kind: 'string'; string: string }
  | { kind: 'boolean'; boolean: boolean }
  | { kind: 'integer'; integer: number }

export interface RoutingModel {
  id: string
  name: string
  status: RoutingStatus
  revision: number
  modelRevision: number
  catalogRevision: string
  aliases: string[]
  capabilities: string[]
  reasoning?: { type?: string; efforts?: string[] }
  loras: string[]
  control: RoutingModelControl
  pricing: Record<string, string | null>
  backends: Array<{
    providerId: string
    providerModelId: string
    credentialConfigured: boolean
    weight: string
  }>
  createdAt: string
  updatedAt: string
}

export interface RoutingModelCard {
  aliases: string[]
  paramSize?: string
  contextWindowSize?: number
  description?: string
  capabilities: string[]
  reasoning?: { type?: string; efforts?: string[] }
  loras: string[]
  qualityScore?: number
  modality?: string
  tags: string[]
}

export interface RoutingModelCardView {
  id: string
  name: string
  card: RoutingModelCard
}

export interface RoutingModelControl {
  retry: {
    count: number
    on: FallbackTrigger[]
  }
  timeout: {
    request: string
    stream: string
  }
}

export interface RoutingPricing {
  inputCostPerMillionTokens: string | null
  outputCostPerMillionTokens: string | null
  cacheReadCostPerMillionTokens: string | null
  cacheWriteCostPerMillionTokens: string | null
}

export interface RoutingModelBackendWrite {
  providerId: string
  interfaceId?: string
  providerModelId: string
  credentialId?: string
  baseUrl?: string
  connectionFields?: Record<string, unknown>
  weight?: string
}

export interface RoutingModelWrite {
  id?: string
  name: string
  aliases?: string[]
  capabilities?: string[]
  reasoning?: { type?: string; efforts?: string[] }
  loras?: string[]
  control: RoutingModelControl
  pricing: RoutingPricing
  backends: RoutingModelBackendWrite[]
}

export interface RoutingModelPatch {
  name?: string
  aliases?: string[]
  capabilities?: string[]
  reasoning?: { type?: string; efforts?: string[] }
  loras?: string[]
  control?: RoutingModelControl
  pricing?: RoutingPricing
  backends?: RoutingModelBackendWrite[]
}

export interface RoutingBulkModelSelection {
  catalogItemId: string
  id?: string
  name: string
  aliases?: string[]
  capabilities?: string[]
  reasoning?: { type?: string; efforts?: string[] }
  loras?: string[]
  control: RoutingModelControl
  pricing: RoutingPricing
}

export interface RoutingBulkImportRequest {
  providerId: string
  interfaceId?: string
  catalogRevision: string
  discoveryClaim: string
  credentialId?: string
  baseUrl?: string
  connectionFields?: Record<string, unknown>
  weight?: string
  selections: RoutingBulkModelSelection[]
}

export interface RoutingRecipe {
  id: string
  name: string
  description?: string
  status: RoutingStatus
  revision: number
  recipeRevision: number
  origin: 'custom' | 'distribution'
  immutable: boolean
  provenance?: RoutingRecipeProvenance
  decisions: RoutingDecision[]
  document: Record<string, unknown>
  createdAt: string
  updatedAt: string
}

export interface RoutingRecipeProvenance {
  distributionId: string
  distributionVersion: string
  assetDigest: string
  sourceRecipeId: string
  sourceRevision: number
  recipeDigest: string
  installedAt: string
}

export interface RoutingEntrypointRule {
  id: string
  name: string
  matchers?: RoutingMatcher[]
  recipeId: string
  recipeRevision: number
  assignments: Record<string, RoutingAssignmentSet>
}

export interface RoutingEntrypoint {
  id: string
  name: string
  status: RoutingStatus
  revision: number
  entrypointRevision: number
  aliases: string[]
  ruleCount: number
  assignedModelCount: number
  rules?: RoutingEntrypointRule[]
  createdAt: string
  updatedAt: string
}

export interface RoutingEntrypointWrite {
  id?: string
  name: string
  aliases: string[]
  rules: Array<{
    id?: string
    name: string
    matchers?: RoutingMatcher[]
    recipeId: string
    assignments: Record<string, RoutingAssignmentSetWrite>
  }>
}

export interface RoutingRecipeWrite {
  id?: string
  name: string
  description?: string
  document: Record<string, unknown>
}

export interface RoutingMutationReceipt {
  resource?: { kind: string; id: string; revision: number }
  operation?: { operationId: string; desiredRevision?: number }
  idempotency?: { replayed: boolean; originalRequestId?: string }
}

export interface RoutingResolveResponse {
  outcome: 'matched' | 'claimed_no_match' | 'unclaimed'
  entrypoint?: {
    id: string
    revision: number
    name: string
    aliases: string[]
  }
  rule?: RoutingEntrypointRule
  recipe?: {
    id: string
    revision: number
    name: string
    decisions: RoutingDecision[]
    document: Record<string, unknown>
  }
}

export interface RoutingProbeResponse {
  reachable: boolean
  latencyMilliseconds: number
  checkedAt: string
}

export interface ManagementOperation {
  operationId: string
  kind: string
  state: 'pending' | 'running' | 'succeeded' | 'partially_succeeded' | 'failed' | 'cancelled'
  itemErrors?: Array<{ itemId?: string; code: string; reason: string }>
}

export interface RoutingPage<T> {
  data: T[]
  page: { nextCursor?: string; hasMore: boolean; pageSize: number }
}

export interface RoutingListParams {
  search?: string
  cursor?: string
  pageSize?: number
  status?: RoutingStatus
  signal?: AbortSignal
}
