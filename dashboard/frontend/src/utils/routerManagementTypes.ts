export type ResourceStatus = 'active' | 'disabled' | 'deleted'
export type SubjectType = 'user' | 'team' | 'api_key'

export interface PageInfo {
  nextCursor?: string
  hasMore: boolean
  pageSize: number
}

export interface ManagementPage<T> {
  data: T[]
  page: PageInfo
}

export interface ResourceDetail<T> {
  data: T
}

export interface ResourceReference {
  kind: string
  id: string
  revision: number
}

export interface MutationReceipt {
  resource?: ResourceReference
  operation?: { operationId: string; desiredRevision?: number }
  idempotency?: { replayed: boolean; originalRequestId?: string }
}

export interface ManagementUser {
  userId: string
  email: string
  displayName: string
  status: ResourceStatus
  revision: number
  createdAt: string
  updatedAt: string
  deletedAt?: string
}

export interface ManagementTeam {
  teamId: string
  name: string
  description: string
  status: ResourceStatus
  revision: number
  createdAt: string
  updatedAt: string
  deletedAt?: string
}

export interface ManagementMembership {
  teamId: string
  userId: string
  role: 'admin' | 'member'
  status: ResourceStatus
  revision: number
  createdAt: string
  updatedAt: string
  displayName?: string
  userStatus?: ResourceStatus
  teamName?: string
  teamStatus?: ResourceStatus
}

export interface ManagementAPIKeyOwner {
  type: 'user' | 'team'
  id: string
}

export interface ManagementAPIKey {
  keyId: string
  name: string
  owner: ManagementAPIKeyOwner
  contextTeamId?: string
  status: ResourceStatus
  expiresAt?: string
  lastUsedAt?: string
  revision: number
  createdAt: string
  updatedAt: string
  deletedAt?: string
}

export interface ManagementCredential {
  credentialId: string
  keyId: string
  kid: string
  status: ResourceStatus
  revealable: boolean
  notBefore: string
  expiresAt?: string
  revokedAt?: string
  createdAt: string
}

export interface IssuedAPIKeySecret {
  data: ManagementAPIKey
  credential: ManagementCredential
  secret: string
  accessPolicyBindings?: Array<{ policyId: string; bindingId: string }>
  rateLimitOverride?: { policyId: string; bindingId: string; created: boolean }
  deliveryExpiresAt: string
}

export interface PolicySubject {
  type: SubjectType
  id: string
}

export interface AccessPolicyGrant {
  resourceType: 'entrypoint' | 'model'
  resourceId: string
  permission: 'discover' | 'invoke'
  effect: 'allow' | 'deny'
}

export interface EffectiveAccessGrant {
  resourceType: 'entrypoint' | 'model'
  resourceId: string
  permissions: Array<'discover' | 'invoke'>
  effect: 'allow' | 'deny'
  source: {
    subjectType: SubjectType
    subjectId: string
    bindingId: string
  }
}

export interface ManagementEffectivePolicy {
  subject: PolicySubject
  revision: number
  appliedRevision: number
  access: { grants: EffectiveAccessGrant[] }
  quota: ManagementEffectiveQuota
}

export interface ManagementAccessPolicy {
  policyId: string
  name: string
  description: string
  status: ResourceStatus
  revision: number
  grants: AccessPolicyGrant[]
  createdAt: string
  updatedAt: string
}

export interface RateLimitRule {
  ruleId?: string
  metric:
    | 'requests'
    | 'input_tokens'
    | 'output_tokens'
    | 'total_tokens'
    | 'concurrent_requests'
    | 'served_input_tokens'
    | 'served_output_tokens'
    | 'served_total_tokens'
    | 'cost'
  algorithm: 'sliding_log' | 'calendar_window' | 'token_bucket' | 'gcra' | 'concurrency'
  limit?: string
  window?: string
  period?: 'day' | 'month'
  timezone?: string
  capacity?: string
  refillAmount?: string
  refillPeriod?: string
  emissionInterval?: string
  burstTolerance?: number
  accounting: 'request' | 'response_actual'
  enforcement: 'enforce' | 'shadow'
  ordinal?: number
}

export interface ManagementRateLimitPolicy {
  policyId: string
  name: string
  description: string
  status: ResourceStatus
  revision: number
  rules: RateLimitRule[]
  createdAt: string
  updatedAt: string
}

export interface ManagementPolicyBinding {
  bindingId: string
  policyId: string
  subject: PolicySubject
  mode?: 'allocation' | 'hard_cap'
  quotaPartitionId?: string
  status: ResourceStatus
  revision: number
  createdAt: string
  updatedAt: string
}

export interface ManagementQuotaMeter {
  policyId: string
  ruleId: string
  bindingId: string
  source: { subjectType: SubjectType; subjectId: string; bindingId: string }
  counterOwner: string
  metric: string
  algorithm: string
  accounting: string
  enforcement: string
  window?: string
  currency?: string
  limit: string
  used: string
  remaining: string | null
  overage?: string
  resetAt?: string
  completeness: string
  knownDispatches: string
  incompleteDispatches: string
  capacityState: string
  activeFenceIds: string[]
  freshness: { source: string; asOf: string }
}

export interface ManagementEffectiveQuota {
  meters: ManagementQuotaMeter[]
  limitingRuleId?: string
  unknownUsageFences: string[]
  asOf: string
}

export interface ManagementPrincipalSummary {
  principalId: string
  displayName: string
  kind: string
  status: string
}

export interface ManagementSessionSummary {
  sessionId: string
  authenticatedAt: string
  expiresAt: string
  evidenceKind: string
}

export interface ManagementNamespaceSummary {
  namespaceId: string
  name: string
  status: string
  desiredRevision: number
  appliedRevision: number
}

export interface ManagementMeUser {
  userId: string
  email: string
  displayName: string
  status: string
}

export interface ManagementMeTeam {
  teamId: string
  name: string
  role: 'admin' | 'member'
  status: string
}

export interface ManagementSelfServicePolicy {
  maxKeysPerUser: number
  maxDelegatedSessions: number
  delegatedSessionTtlSeconds: number
  allowTeamKeyDelegation: boolean
  automaticFirstKey: boolean
  revision: number
}

export interface ManagementMeNamespace {
  namespace: ManagementNamespaceSummary
  permissions: string[]
  roleBindings: unknown[]
  user?: ManagementMeUser
  teams: ManagementMeTeam[]
  selfServicePolicy: ManagementSelfServicePolicy
}

export interface ManagementMe {
  principal: ManagementPrincipalSummary
  session: ManagementSessionSummary
  clusterPermissions: string[]
  namespaces: ManagementMeNamespace[]
}

export interface SelfInferenceKey {
  keyId: string
  name: string
  owner: ManagementAPIKeyOwner
  contextTeamId?: string
  expiresAt?: string
}

export interface DelegatedInferenceSession {
  sessionId: string
  publicId: string
  keyId: string
  userId: string
  teamId?: string
  audience: string
  status: string
  notBefore: string
  expiresAt: string
  createdAt: string
}

export interface SecretEnvelope {
  resourceId: string
  kind: 'delegated_inference_credential'
  secret: string
  expiresAt?: string
}

export interface ManagementCostSummary {
  currency: string
  knownAmount: string
  completeness: 'complete' | 'partial' | 'unknown'
  knownDispatches: string
  incompleteDispatches: string
}

export interface ManagementTimingSummary {
  sampleCount: string
  totalMilliseconds: string
  averageMilliseconds: number
  p50Milliseconds: number
  p95Milliseconds: number
  p99Milliseconds: number
  percentilesAreEstimated: boolean
}

export interface ManagementUsageTotals {
  requests: string
  successfulRequests: string
  inputTokens: string
  outputTokens: string
  totalTokens: string
  incompleteDispatches: string
  completeness: 'complete' | 'partial' | 'unknown'
  costs: ManagementCostSummary[]
  latency: ManagementTimingSummary
  ttft: ManagementTimingSummary
}

export interface ManagementUsageSummary {
  totals: ManagementUsageTotals
  grain: 'minute' | 'hour' | 'day'
  asOf?: string
  ledgerWatermark?: string
  ingestionLag?: number
  final: boolean
}

/** Permission-projected control-plane counts. Omitted fields are unavailable. */
export interface ManagementAccessStatistics {
  asOf: string
  expiringBefore: string
  users?: string
  teams?: string
  activeApiKeys?: string
  expiringApiKeys?: string
  accessPolicies?: string
  activeRatePolicies?: string
}

export interface ManagementUsageSeriesPoint {
  bucketStart: string
  totals: ManagementUsageTotals
}

export interface ManagementUsageSeries {
  points: ManagementUsageSeriesPoint[]
  grain: 'minute' | 'hour' | 'day'
  asOf?: string
  ledgerWatermark?: string
  final: boolean
}

export interface ManagementUsageBreakdownRow {
  value: string
  totals: ManagementUsageTotals
}

export interface ManagementUsageBreakdown {
  dimension: string
  rows: ManagementUsageBreakdownRow[]
  grain: 'minute' | 'hour' | 'day'
}

export interface ManagementRequestLog {
  admissionId: string
  eventId: string
  occurredAt: string
  completedAt: string
  protocol: string
  path: string
  statusCode: number
  errorCode?: string
  usageState: string
  inputTokens: string
  outputTokens: string
  latencyMilliseconds: number
  ttftMilliseconds?: number
  stream: boolean
  toolCall: boolean
  apiKeyId?: string
  userId?: string
  teamId?: string
  entrypointId?: string
  recipeId?: string
  metadata?: Record<string, string>
  costs: ManagementCostSummary[]
}

/** Immutable evidence returned by the scoped request-log detail endpoint. */
export interface ManagementRequestLogDetail {
  request: ManagementRequestLog
  routing: unknown
  quotaReceipts: unknown[]
  dispatches: unknown[]
}
