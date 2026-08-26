import { getManagementNamespace } from './managementApiContract'
import { query, request, viewPage } from './inferenceAccessTransport'
import type {
  AccessOverview,
  AccessPage,
  AccessUsageEvent,
  UsageFilter,
  UsageReadCapabilities,
  UsageSlice,
  UsageSummary,
} from './inferenceAccessTypes'
import { ManagementApiError } from './managementApiContract'
import type {
  ManagementAccessStatistics,
  ManagementPage,
  ManagementRequestLog,
  ManagementRequestLogDetail,
  ManagementUsageBreakdown,
  ManagementUsageSeries,
  ManagementUsageSummary,
  ResourceDetail,
} from './routerManagementTypes'

const observabilityQuery = (filter: UsageFilter) =>
  query({
    userId: filter.userId,
    teamId: filter.teamId,
    apiKeyId: filter.keyId,
    logicalModelId: filter.model,
    start: filter.from,
    end: filter.to,
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    cursor: filter.cursor,
    pageSize: filter.limit,
  })

const usageQuery = (filter: UsageFilter) => {
  const params = observabilityQuery(filter)
  params.set('grain', filter.granularity ?? 'auto')
  return params
}

const usageBreakdownQuery = (filter: UsageFilter, dimension: string) => {
  const params = usageQuery(filter)
  params.set('dimension', dimension)
  return params
}

const emptyUsage = (): UsageSummary => ({
  final: true,
  completeness: 'complete',
  granularity: 'hour',
  requests: 0,
  successful: 0,
  failed: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  activeKeys: 0,
  averageLatencyMs: 0,
  p95LatencyMs: 0,
  averageTtftMs: 0,
  p95TtftMs: 0,
  costs: [],
  series: [],
  byModel: [],
  byEntrypoint: [],
  byRecipe: [],
  byDecision: [],
  byUser: [],
  byTeam: [],
  byKey: [],
})

const emptyBreakdown = (dimension: string, grain: ManagementUsageBreakdown['grain']) =>
  ({
    dimension,
    rows: [],
    grain,
    final: true,
  }) as ManagementUsageBreakdown

const requestedGrain = (filter: UsageFilter): ManagementUsageBreakdown['grain'] =>
  filter.granularity && filter.granularity !== 'auto' ? filter.granularity : 'hour'

async function optionalBreakdown(
  filter: UsageFilter,
  dimension: string,
): Promise<ManagementUsageBreakdown> {
  try {
    return await request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, dimension),
    })
  } catch (error) {
    if (error instanceof ManagementApiError && error.status === 403) {
      return emptyBreakdown(dimension, requestedGrain(filter))
    }
    throw error
  }
}

async function loadUsage(
  filter: UsageFilter,
  summaryOperationId:
    | 'getUsage'
    | 'getApiKeysByKeyIdUsage'
    | 'getUsersByUserIdUsage'
    | 'getTeamsByTeamIdUsage',
  pathParameters: Record<string, string> | undefined,
  summaryFilter: UsageFilter = filter,
  capabilities: UsageReadCapabilities = {},
): Promise<UsageSummary> {
  const publicFilter = capabilities.internalDimensions ? filter : { ...filter, model: undefined }
  const publicSummaryFilter = capabilities.internalDimensions
    ? summaryFilter
    : { ...summaryFilter, model: undefined }
  const [summary, series, byModel, byEntrypoint, byRecipe, byDecision, byUser, byTeam, byKey] =
    await Promise.all([
      request<ManagementUsageSummary>(summaryOperationId, {
        pathParameters,
        query: usageQuery(publicSummaryFilter),
      }),
      request<ManagementUsageSeries>('getUsageSeries', { query: usageQuery(publicFilter) }),
      capabilities.internalDimensions
        ? optionalBreakdown(publicFilter, 'logical_model')
        : Promise.resolve(emptyBreakdown('logical_model', requestedGrain(publicFilter))),
      optionalBreakdown(publicFilter, 'entrypoint'),
      optionalBreakdown(publicFilter, 'recipe'),
      optionalBreakdown(publicFilter, 'decision'),
      optionalBreakdown(publicFilter, 'user'),
      optionalBreakdown(publicFilter, 'team'),
      optionalBreakdown(publicFilter, 'api_key'),
    ])
  const totals = summary.totals
  const requests = Number(totals.requests)
  const successful = Number(totals.successfulRequests)
  const slice = (item: ManagementUsageBreakdown['rows'][number]): UsageSlice => ({
    id: item.value,
    requests: Number(item.totals.requests),
    successful: Number(item.totals.successfulRequests),
    failed: Math.max(0, Number(item.totals.requests) - Number(item.totals.successfulRequests)),
    promptTokens: Number(item.totals.inputTokens),
    completionTokens: Number(item.totals.outputTokens),
    totalTokens: Number(item.totals.totalTokens),
    averageLatencyMs: item.totals.latency.averageMilliseconds,
    p95LatencyMs: item.totals.latency.p95Milliseconds,
    costs: item.totals.costs,
  })
  return {
    ...emptyUsage(),
    final: summary.final,
    completeness: totals.completeness,
    ...(summary.asOf ? { asOf: summary.asOf } : {}),
    granularity: summary.grain,
    requests,
    successful,
    failed: Math.max(0, requests - successful),
    promptTokens: Number(totals.inputTokens),
    completionTokens: Number(totals.outputTokens),
    totalTokens: Number(totals.totalTokens),
    averageLatencyMs: totals.latency.averageMilliseconds,
    p95LatencyMs: totals.latency.p95Milliseconds,
    averageTtftMs: totals.ttft.averageMilliseconds,
    p95TtftMs: totals.ttft.p95Milliseconds,
    costs: totals.costs,
    series: series.points.map((point) => ({
      bucket: point.bucketStart,
      requests: Number(point.totals.requests),
      successful: Number(point.totals.successfulRequests),
      failed: Math.max(0, Number(point.totals.requests) - Number(point.totals.successfulRequests)),
      promptTokens: Number(point.totals.inputTokens),
      completionTokens: Number(point.totals.outputTokens),
      totalTokens: Number(point.totals.totalTokens),
      averageLatencyMs: point.totals.latency.averageMilliseconds,
      p95LatencyMs: point.totals.latency.p95Milliseconds,
      averageTtftMs: point.totals.ttft.averageMilliseconds,
      p95TtftMs: point.totals.ttft.p95Milliseconds,
      costs: point.totals.costs,
    })),
    byModel: byModel.rows.map(slice),
    byEntrypoint: byEntrypoint.rows.map(slice),
    byRecipe: byRecipe.rows.map(slice),
    byDecision: byDecision.rows.map(slice),
    byUser: byUser.rows.map(slice),
    byTeam: byTeam.rows.map(slice),
    byKey: byKey.rows.map(slice),
  }
}

export async function usage(
  filter: UsageFilter = {},
  capabilities: UsageReadCapabilities = {},
): Promise<UsageSummary> {
  return loadUsage(filter, 'getUsage', undefined, filter, capabilities)
}

export async function keyUsage(
  keyId: string,
  filter: UsageFilter = {},
  capabilities: UsageReadCapabilities = {},
): Promise<UsageSummary> {
  const exactFilter = { ...filter, keyId }
  const summaryFilter = { ...filter }
  delete summaryFilter.keyId
  return loadUsage(exactFilter, 'getApiKeysByKeyIdUsage', { keyId }, summaryFilter, capabilities)
}

export async function userUsage(
  userId: string,
  filter: UsageFilter = {},
  capabilities: UsageReadCapabilities = {},
): Promise<UsageSummary> {
  const exactFilter = { ...filter, userId }
  const summaryFilter = { ...filter }
  delete summaryFilter.userId
  return loadUsage(exactFilter, 'getUsersByUserIdUsage', { userId }, summaryFilter, capabilities)
}

export async function teamUsage(
  teamId: string,
  filter: UsageFilter = {},
  capabilities: UsageReadCapabilities = {},
): Promise<UsageSummary> {
  const exactFilter = { ...filter, teamId }
  const summaryFilter = { ...filter }
  delete summaryFilter.teamId
  return loadUsage(exactFilter, 'getTeamsByTeamIdUsage', { teamId }, summaryFilter, capabilities)
}

function mapRequestLog(item: ManagementRequestLog): AccessUsageEvent {
  const inputTokens = Number(item.inputTokens)
  const outputTokens = Number(item.outputTokens)
  const namespaceID = getManagementNamespace()
  const externalRequestID = item.externalRequestId
  return {
    id: namespaceID ? `${namespaceID}:${item.admissionId}` : item.admissionId,
    requestId:
      typeof externalRequestID === 'string' && externalRequestID ? externalRequestID : item.eventId,
    admissionId: item.admissionId,
    keyId: item.apiKeyId ?? '',
    userId: item.userId,
    teamId: item.teamId,
    namespaceId: namespaceID || undefined,
    model: item.models?.[0]?.name || item.entrypointId || item.recipeId || '',
    models: item.models ?? [],
    entrypointId: item.entrypointId,
    recipeId: item.recipeId,
    decisionId: item.decisionId,
    decisionName: item.decisionName,
    decisionTier: item.decisionTier,
    completedAt: item.completedAt,
    streaming: item.stream,
    toolCall: item.toolCall,
    costs: item.costs,
    statusCode: item.statusCode,
    promptTokens: inputTokens,
    completionTokens: outputTokens,
    totalTokens: inputTokens + outputTokens,
    latencyMs: item.latencyMilliseconds,
    ttftMs: item.ttftMilliseconds,
    errorCode: item.errorCode,
    metadata: item.metadata,
    createdAt: item.occurredAt,
  }
}

export async function overview(): Promise<AccessOverview> {
  const [statistics, currentUsage] = await Promise.all([
    request<ManagementAccessStatistics>('getStatistics'),
    usage(),
  ])
  return {
    users: statistics.users ?? null,
    teams: statistics.teams ?? null,
    activeKeys: statistics.activeApiKeys ?? null,
    expiringKeys: statistics.expiringApiKeys ?? null,
    accessGroups: statistics.accessPolicies ?? null,
    enabledBudgets: statistics.activeRatePolicies ?? null,
    requestsToday: currentUsage.requests,
    successfulToday: currentUsage.successful,
    tokensToday: currentUsage.totalTokens,
    p95LatencyMs: currentUsage.p95LatencyMs,
  }
}

export async function requestLogs(filter: UsageFilter = {}): Promise<AccessPage<AccessUsageEvent>> {
  const requestQuery = observabilityQuery(filter)
  if (filter.q?.trim()) requestQuery.set('requestId', filter.q.trim())
  const page = await request<ManagementPage<ManagementRequestLog>>('getRequestLogs', {
    query: requestQuery,
  })
  return viewPage(page, mapRequestLog, filter.q?.trim())
}

export async function requestLog(id: string): Promise<AccessUsageEvent> {
  if (!id.includes(':'))
    throw new Error('Request log identity must include namespace and admission id.')
  const [namespaceID, admissionID] = id.split(':', 2)
  const detail = await request<ResourceDetail<ManagementRequestLogDetail>>(
    'getNamespacesByNamespaceIdRequestLogsByAdmissionId',
    { pathParameters: { namespaceId: namespaceID, admissionId: admissionID } },
  )
  return {
    ...mapRequestLog(detail.data.request),
    routing: detail.data.routing,
    quotaReceipts: detail.data.quotaReceipts,
    dispatches: detail.data.dispatches,
  }
}
