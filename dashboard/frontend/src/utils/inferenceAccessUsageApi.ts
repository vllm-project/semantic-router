import { getManagementNamespace } from './managementApiContract'
import { query, request, viewPage } from './inferenceAccessTransport'
import type {
  AccessOverview,
  AccessPage,
  AccessUsageEvent,
  UsageFilter,
  UsageSlice,
  UsageSummary,
} from './inferenceAccessTypes'
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

const usageQuery = (filter: UsageFilter) =>
  query({
    userId: filter.userId,
    teamId: filter.teamId,
    apiKeyId: filter.keyId,
    logicalModelId: filter.model,
    start: filter.from,
    end: filter.to,
    grain: filter.granularity ?? 'auto',
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    cursor: filter.cursor,
    pageSize: filter.limit,
  })

const usageBreakdownQuery = (filter: UsageFilter, dimension: string) => {
  const params = usageQuery(filter)
  params.set('dimension', dimension)
  return params
}

const emptyUsage = (): UsageSummary => ({
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
  byUser: [],
  byTeam: [],
  byKey: [],
})

async function loadUsage(
  filter: UsageFilter,
  summaryOperationId:
    | 'getUsage'
    | 'getApiKeysByKeyIdUsage'
    | 'getUsersByUserIdUsage'
    | 'getTeamsByTeamIdUsage',
  pathParameters: Record<string, string> | undefined,
  summaryFilter: UsageFilter = filter,
): Promise<UsageSummary> {
  const [summary, series, byModel, byUser, byTeam, byKey] = await Promise.all([
    request<ManagementUsageSummary>(summaryOperationId, {
      pathParameters,
      query: usageQuery(summaryFilter),
    }),
    request<ManagementUsageSeries>('getUsageSeries', { query: usageQuery(filter) }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'logical_model'),
    }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'user'),
    }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'team'),
    }),
    request<ManagementUsageBreakdown>('getUsageBreakdowns', {
      query: usageBreakdownQuery(filter, 'api_key'),
    }),
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
    byUser: byUser.rows.map(slice),
    byTeam: byTeam.rows.map(slice),
    byKey: byKey.rows.map(slice),
  }
}

export async function usage(filter: UsageFilter = {}): Promise<UsageSummary> {
  return loadUsage(filter, 'getUsage', undefined)
}

export async function keyUsage(keyId: string, filter: UsageFilter = {}): Promise<UsageSummary> {
  const exactFilter = { ...filter, keyId }
  const summaryFilter = { ...filter }
  delete summaryFilter.keyId
  return loadUsage(exactFilter, 'getApiKeysByKeyIdUsage', { keyId }, summaryFilter)
}

export async function userUsage(userId: string, filter: UsageFilter = {}): Promise<UsageSummary> {
  const exactFilter = { ...filter, userId }
  const summaryFilter = { ...filter }
  delete summaryFilter.userId
  return loadUsage(exactFilter, 'getUsersByUserIdUsage', { userId }, summaryFilter)
}

export async function teamUsage(teamId: string, filter: UsageFilter = {}): Promise<UsageSummary> {
  const exactFilter = { ...filter, teamId }
  const summaryFilter = { ...filter }
  delete summaryFilter.teamId
  return loadUsage(exactFilter, 'getTeamsByTeamIdUsage', { teamId }, summaryFilter)
}

function mapRequestLog(item: ManagementRequestLog): AccessUsageEvent {
  const inputTokens = Number(item.inputTokens)
  const outputTokens = Number(item.outputTokens)
  const namespaceID = getManagementNamespace()
  return {
    id: namespaceID ? `${namespaceID}:${item.admissionId}` : item.admissionId,
    requestId: item.metadata?.externalRequestId || item.eventId,
    admissionId: item.admissionId,
    keyId: item.apiKeyId ?? '',
    userId: item.userId,
    teamId: item.teamId,
    namespaceId: namespaceID || undefined,
    model: item.entrypointId || item.recipeId || '',
    entrypointId: item.entrypointId,
    recipeId: item.recipeId,
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
  const page = await request<ManagementPage<ManagementRequestLog>>('getRequestLogs', {
    query: usageQuery(filter),
  })
  return viewPage(page, mapRequestLog, filter.q)
}

export async function requestLog(id: string): Promise<AccessUsageEvent> {
  if (!id.includes(':'))
    throw new Error('Request log identity must include namespace and admission id.')
  const [namespaceID, admissionID] = id.split(':', 2)
  const detail = await request<ResourceDetail<ManagementRequestLogDetail>>(
    'getNamespacesByNamespaceIdRequestLogsByAdmissionId',
    { pathParameters: { namespaceId: namespaceID, admissionId: admissionID } },
  )
  return mapRequestLog(detail.data.request)
}
