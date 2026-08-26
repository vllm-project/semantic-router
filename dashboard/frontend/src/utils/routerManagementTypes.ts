import type {
  APIKey,
  APIKeyCredential,
  APIKeyIssuedSecret,
  APIKeyOwner,
  AccessPolicy,
  AccessPolicyBinding,
  AccessPolicyGrant as WireAccessPolicyGrant,
  AccessStatistics,
  CostSummary,
  DelegatedInferenceSession as WireDelegatedInferenceSession,
  EffectiveGrant,
  EffectivePolicy,
  EffectiveQuota,
  EligibleInferenceKey,
  Me,
  MeNamespace,
  MeNamespaceScope,
  MePrincipal,
  MeSelfServicePolicy,
  MeSession,
  MeTeamMembership,
  MeUser,
  Membership,
  MutationReceipt as WireMutationReceipt,
  Page,
  PageInfo as WirePageInfo,
  PolicySubject as WirePolicySubject,
  QuotaMeter as WireQuotaMeter,
  RateLimitBinding,
  RateLimitPolicy,
  RateLimitRule as WireRateLimitRule,
  RateLimitRuleInput,
  RequestDispatch,
  RequestLog,
  RequestLogDetailData,
  RequestModel,
  ResourceReference as WireResourceReference,
  SecretEnvelope as WireSecretEnvelope,
  Team,
  TeamMember,
  TimingSummary,
  UsageBreakdown,
  UsageBreakdownRow,
  UsageSeries,
  UsageSeriesPoint,
  UsageSummary,
  UsageTotals,
  User,
  UserDetail,
  UserMembership,
} from '../generated/managementApiContract'

// This module gives product-facing names to the generated Router wire contract.
// Resource shapes stay generated; only generic envelope specialization lives here.
export type ResourceStatus = User['status']
export type SubjectType = WirePolicySubject['type']

export type PageInfo = WirePageInfo
export type ManagementPage<T> = Omit<Page, 'data'> & { data: T[] }
export type ResourceDetail<T> = Omit<UserDetail, 'data'> & { data: T }

export type ResourceReference = WireResourceReference
export type MutationReceipt = WireMutationReceipt
export type ManagementUser = User
export type ManagementTeam = Team
export type ManagementMembership = Membership | TeamMember | UserMembership
export type ManagementAPIKeyOwner = APIKeyOwner
export type ManagementAPIKey = APIKey
export type ManagementCredential = APIKeyCredential
export type IssuedAPIKeySecret = APIKeyIssuedSecret
export type PolicySubject = WirePolicySubject
export type AccessPolicyGrant = WireAccessPolicyGrant
export type EffectiveAccessGrant = EffectiveGrant
export type ManagementEffectivePolicy = EffectivePolicy
export type ManagementAccessPolicy = AccessPolicy
type AnyRateLimitRule = WireRateLimitRule | RateLimitRuleInput

/** Product editor model; normalized to a generated RateLimitRuleInput before transport. */
export interface RateLimitRule {
  ruleId?: string
  metric: AnyRateLimitRule['metric']
  algorithm: AnyRateLimitRule['algorithm']
  limit?: string
  window?: string
  period?: 'day' | 'month'
  timezone?: string
  capacity?: string
  refillAmount?: string
  refillPeriod?: string
  emissionInterval?: string
  burstTolerance?: number
  accounting: AnyRateLimitRule['accounting']
  enforcement: AnyRateLimitRule['enforcement']
  ordinal?: number
}

export type RateLimitRuleWrite = RateLimitRuleInput
export type ManagementRateLimitPolicy = RateLimitPolicy
export type ManagementPolicyBinding = AccessPolicyBinding | RateLimitBinding
export type ManagementAccessPolicyBinding = AccessPolicyBinding
export type ManagementRateLimitBinding = RateLimitBinding
export type ManagementQuotaMeter = WireQuotaMeter
export type ManagementEffectiveQuota = EffectiveQuota
export type ManagementPrincipalSummary = MePrincipal
export type ManagementSessionSummary = MeSession
export type ManagementNamespaceSummary = MeNamespace
export type ManagementMeUser = MeUser
export type ManagementMeTeam = MeTeamMembership
export type ManagementSelfServicePolicy = MeSelfServicePolicy
export type ManagementMeNamespace = MeNamespaceScope
export type ManagementMe = Me
export type SelfInferenceKey = EligibleInferenceKey
export type DelegatedInferenceSession = WireDelegatedInferenceSession
export type SecretEnvelope = WireSecretEnvelope
export type ManagementCostSummary = CostSummary
export type ManagementTimingSummary = TimingSummary
export type ManagementUsageTotals = UsageTotals
export type ManagementUsageSummary = UsageSummary
export type ManagementAccessStatistics = AccessStatistics
export type ManagementUsageSeriesPoint = UsageSeriesPoint
export type ManagementUsageSeries = UsageSeries
export type ManagementUsageBreakdownRow = UsageBreakdownRow
export type ManagementUsageBreakdown = UsageBreakdown
export type ManagementRequestLog = RequestLog
export type ManagementRequestDispatch = RequestDispatch
export type ManagementRequestModel = RequestModel
export type ManagementRequestLogDetail = RequestLogDetailData
