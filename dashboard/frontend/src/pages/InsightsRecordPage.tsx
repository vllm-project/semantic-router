import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  Bot,
  Clock,
  Copy,
  FileJson,
  Layers,
  Link2,
  ListTree,
  MoreHorizontal,
  SearchX,
} from 'lucide-react'

import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CopyButton,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  IconTile,
  InlineCode,
  MetaItem,
  MiniStat,
  PageContainer,
  PageHeader,
  Skeleton,
  toast,
} from '../ui'
import { canAccessReplayFlowDetails } from '../utils/accessControl'
import { formatDateTime } from '../utils/dateTime'
import { formatInsightsCost } from '../utils/insightsCost'

import { compactIdentifier, displaySessionId } from './insightsAgentIdentity'
import InsightsAgentContext from './InsightsAgentContext'
import { useInsightsConfig } from './insightsConfigFacts'
import { formatLatency, formatTokenCount } from './insightsDisplayFormat'
import {
  InsightsLoadError,
  InsightsUnavailable,
  LifecycleStatus,
  RefreshButton,
} from './InsightsDisplay'
import { HarnessName } from './InsightsHarness'
import { InsightsDataProviders, modelLabel, useModelLabels } from './insightsModelLabelSupport'
import {
  fetchInsightsJSON,
  fetchInsightsRecord,
  fetchInsightsSessionReviews,
  fetchInsightsTrajectory,
  InsightsRequestError,
  insightsUnavailableReason,
  type InsightsUnavailableReason,
} from './insightsPageApi'
import {
  buildInsightsRecordSections,
  formatInsightsDecisionName,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type { InsightsRecord, InsightsTrajectory } from './insightsPageTypes'
import InsightsRecordSection from './InsightsRecordSection'
import InsightsRecordTrace from './InsightsRecordTrace'
import InsightsRoutingTrace from './InsightsRoutingTrace'
import InsightsSessionRoutes, { InsightsSessionPosition } from './InsightsSessionRoutes'
import { getRunPath } from './insightsRunSupport'
import type { RunResponse } from './insightsRunTypes'

// One record, its models read by their labels.
export default function InsightsRecordPage() {
  return (
    <InsightsDataProviders>
      <RecordInsight />
    </InsightsDataProviders>
  )
}

function BackLink() {
  return (
    <Link
      to="/observe/requests"
      className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-3.5" />
      Requests
    </Link>
  )
}

function RecordSkeleton() {
  return (
    <div className="flex flex-col gap-4" role="status" aria-label="Loading the request">
      <div className="flex items-start gap-3.5">
        <Skeleton className="size-11 rounded-xl" />
        <div className="flex flex-col gap-2">
          <Skeleton className="h-6 w-72" />
          <Skeleton className="h-4 w-96" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {Array.from({ length: 5 }, (_, index) => (
          <Skeleton key={index} className="h-[68px] rounded-lg" />
        ))}
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Skeleton className="h-96 rounded-xl" />
        <Skeleton className="h-72 rounded-xl" />
      </div>
    </div>
  )
}

function exportRecord(record: InsightsRecord) {
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${record.id}.json`
  link.click()
  URL.revokeObjectURL(url)
  toast.success('Exported the replay record', { description: `${record.id}.json` })
}

function copy(value: string, label: string) {
  void navigator.clipboard?.writeText(value).catch(() => undefined)
  toast.success(`${label} copied`, { description: value })
}

function RecordInsight() {
  const navigate = useNavigate()
  const { recordId } = useParams<{ recordId: string }>()
  const { isReadonly } = useReadonly()
  const { user } = useAuth()
  const canViewBodies = canAccessReplayFlowDetails(user)
  const modelLabels = useModelLabels()
  const { agentRules } = useInsightsConfig()
  const [record, setRecord] = useState<InsightsRecord | null>(null)
  const [trajectory, setTrajectory] = useState<InsightsTrajectory | null>(null)
  const [run, setRun] = useState<RunResponse | null>(null)
  const [sessionReviews, setSessionReviews] =
    useState<Awaited<ReturnType<typeof fetchInsightsSessionReviews>>>()
  const [traceError, setTraceError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notFound, setNotFound] = useState(false)
  const [unavailable, setUnavailable] = useState<InsightsUnavailableReason>()

  const loadRecord = useCallback(async () => {
    if (!recordId) {
      setRecord(null)
      setNotFound(true)
      setLoading(false)
      return
    }

    setLoading(true)
    setTraceError(null)

    try {
      const nextRecord = await fetchInsightsRecord(recordId)
      setRecord(nextRecord)
      setError(null)
      setNotFound(false)
      setUnavailable(undefined)
      setTrajectory(null)
      setRun(null)
      setSessionReviews(undefined)
      if (nextRecord.session_id) {
        const session = nextRecord.session_id
        // Reviews only mark the session's history; a failed lookup leaves it unmarked.
        void fetchInsightsSessionReviews(session)
          .then(setSessionReviews)
          .catch(() => setSessionReviews(undefined))
        // The run names the record's agent and the session's title; without it they stay unnamed.
        void fetchInsightsJSON<RunResponse>(
          `/api/router/api/v1/observability/replays/run?${new URLSearchParams({ session_id: session })}`,
          'agent run',
        )
          .then(setRun)
          .catch(() => setRun(null))
        try {
          setTrajectory(await fetchInsightsTrajectory(session, nextRecord.recipe || ''))
        } catch (traceCause) {
          setTraceError(traceCause instanceof Error ? traceCause.message : 'Unknown error')
        }
      }
    } catch (err) {
      setRecord(null)
      const reason = insightsUnavailableReason(err)
      const missing = err instanceof InsightsRequestError && err.status === 404
      setNotFound(missing)
      setUnavailable(missing ? undefined : reason)
      setError(missing || reason ? null : err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [recordId])

  useEffect(() => {
    void loadRecord()
  }, [loadRecord])

  const sections = useMemo(
    () =>
      record ? buildInsightsRecordSections(record, { isReadonly, modelLabels, canViewBodies }) : [],
    [canViewBodies, isReadonly, modelLabels, record],
  )

  if (notFound) {
    return (
      <PageContainer>
        <BackLink />
        <Card className="border-dashed shadow-none">
          <EmptyState
            icon={SearchX}
            tone="blue"
            title="Request not found"
            description={
              <>
                Nothing in the replay store matches <InlineCode>{recordId}</InlineCode>. Records
                expire with the store&apos;s retention, Replay may be off, or the id may be
                mistyped.
              </>
            }
            actions={
              <Button onClick={() => navigate('/observe/requests')}>Back to requests</Button>
            }
            footer="Record ids come from the x-vsr-replay-id response header."
          />
        </Card>
      </PageContainer>
    )
  }

  if (unavailable) {
    return (
      <PageContainer>
        <BackLink />
        <Card className="border-dashed shadow-none">
          <InsightsUnavailable
            reason={unavailable}
            noun="requests"
            onRetry={() => void loadRecord()}
          />
        </Card>
      </PageContainer>
    )
  }

  if (!record) {
    return (
      <PageContainer>
        <BackLink />
        {error ? (
          <InsightsLoadError
            title="The request could not load"
            message={error}
            onRetry={() => void loadRecord()}
          />
        ) : null}
        {loading ? <RecordSkeleton /> : null}
      </PageContainer>
    )
  }

  const reviewSection = sections.find((section) => section.title === 'Approval Review')
  const usageSection = sections.find((section) => section.title === 'Usage & Cost')
  const cacheSection = sections.find((section) => section.title === 'Provider Prompt Cache')
  const bodySection = sections.find((section) => section.title === 'Request / Response')
  const indexOf = (title: string) => sections.findIndex((section) => section.title === title)
  const decision = record.decision ? formatInsightsDecisionName(record.decision) : undefined
  const selected = modelLabel(modelLabels, record.selected_model)
  const hasCost = typeof record.actual_cost === 'number' && Boolean(record.currency)
  const saved = typeof record.cost_savings === 'number' && record.currency

  return (
    <PageContainer key={record.id}>
      <BackLink />
      <PageHeader
        icon={<IconTile icon={ListTree} tone="blue" size="lg" />}
        title={
          <span className="inline-flex flex-wrap items-center gap-2">
            <span className="font-mono tracking-[-0.02em]" title={record.id}>
              {compactIdentifier(record.request_id || record.id, 12, 6)}
            </span>
            <CopyButton value={record.id} label="Copy record ID" />
            <LifecycleStatus record={record} size="default" />
            {record.from_cache ? (
              <Badge tone="info" size="sm">
                cache hit
              </Badge>
            ) : null}
          </span>
        }
        description={
          <>
            {selected ? (
              <>
                Routed to <span className="font-medium text-foreground">{selected}</span>
              </>
            ) : (
              'No backend answered'
            )}
            {decision ? (
              <>
                {' '}
                by <span className="font-medium text-foreground">{decision}</span>
              </>
            ) : null}
            {record.original_model ? (
              <>
                {' '}
                for <span className="font-mono text-[12.5px]">{record.original_model}</span>
              </>
            ) : null}
            .
          </>
        }
        meta={
          <>
            <MetaItem icon={<Clock />}>{formatDateTime(record.timestamp)}</MetaItem>
            <MetaItem icon={<Layers />} label="Recipe">
              {record.recipe || 'default'}
            </MetaItem>
            <MetaItem>
              <HarnessName harness={record.agent?.harness} size={13} />
            </MetaItem>
            {record.session_id ? (
              <MetaItem icon={<Bot />} label="Session">
                <Link
                  to={getRunPath(record.session_id, record.id)}
                  className="font-mono hover:text-primary hover:underline"
                  title={record.session_id}
                >
                  {compactIdentifier(displaySessionId(record.session_id), 10, 6)}
                </Link>
              </MetaItem>
            ) : null}
            <InsightsSessionPosition trajectory={trajectory} current={record.id} />
          </>
        }
        actions={
          <>
            {record.session_id ? (
              <Button
                variant="outline"
                onClick={() => navigate(getRunPath(record.session_id ?? '', record.id))}
              >
                <Bot />
                Open agent run
              </Button>
            ) : null}
            <RefreshButton onClick={() => void loadRecord()} loading={loading} />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" aria-label="More actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  onSelect={() =>
                    copy(`${window.location.origin}${getInsightsRecordPath(record.id)}`, 'Link')
                  }
                >
                  <Link2 />
                  Copy link
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => copy(record.id, 'Record ID')}>
                  <Copy />
                  Copy record ID
                </DropdownMenuItem>
                {record.request_id ? (
                  <DropdownMenuItem onSelect={() => copy(record.request_id ?? '', 'Request ID')}>
                    <Copy />
                    Copy request ID
                  </DropdownMenuItem>
                ) : null}
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => exportRecord(record)}>
                  <FileJson />
                  Export JSON
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />

      {error ? (
        <InsightsLoadError
          title="The request could not refresh"
          message={error}
          onRetry={() => void loadRecord()}
        />
      ) : null}

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <MiniStat
          label="Latency"
          value={formatLatency(record.duration_ms)}
          hint={record.streaming ? 'Streamed' : undefined}
        />
        <MiniStat
          label="Tokens"
          value={
            <span>
              {formatTokenCount(record.prompt_tokens)}
              <span className="text-subtle-foreground"> / </span>
              {formatTokenCount(record.completion_tokens)}
            </span>
          }
          hint={
            typeof record.cached_prompt_tokens === 'number' && record.cached_prompt_tokens > 0
              ? `Input / output · ${formatTokenCount(record.cached_prompt_tokens)} cached`
              : 'Input / output'
          }
        />
        <MiniStat
          label="Cost"
          value={hasCost ? formatInsightsCost(record.actual_cost, record.currency) : '—'}
          hint={hasCost ? 'Configured-rate estimate' : 'No price recorded'}
        />
        <MiniStat
          label="Saved"
          value={saved ? formatInsightsCost(record.cost_savings, record.currency) : '—'}
          hint={
            record.baseline_model
              ? `vs. ${modelLabel(modelLabels, record.baseline_model)}`
              : 'No baseline recorded'
          }
        />
        <MiniStat
          label="Turn"
          value={record.turn_index + 1}
          hint={
            typeof trajectory?.turn_count === 'number'
              ? `of ${trajectory.turn_count} in the session`
              : undefined
          }
        />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_380px] xl:grid-rows-[auto_1fr]">
        <div className="min-w-0 xl:col-start-2 xl:row-start-1">
          <InsightsAgentContext
            record={record}
            run={run?.run}
            turnCount={trajectory?.turn_count}
            rules={agentRules}
          />
        </div>
        <div className="flex min-w-0 flex-col gap-4 xl:col-start-1 xl:row-span-2 xl:row-start-1">
          {reviewSection ? (
            <InsightsRecordSection
              section={reviewSection}
              sectionIndex={indexOf('Approval Review')}
            />
          ) : null}
          <Card>
            <CardHeader>
              <CardTitle>Routing trace</CardTitle>
              <CardDescription>What the router did with this request, step by step</CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <InsightsRoutingTrace record={record} sections={sections} />
            </CardContent>
          </Card>
          <InsightsSessionRoutes
            trajectory={trajectory}
            current={record.id}
            reviews={sessionReviews?.reviews}
            reviewsTruncated={sessionReviews?.truncated}
          />
          {usageSection ? (
            <InsightsRecordSection section={usageSection} sectionIndex={indexOf('Usage & Cost')} />
          ) : null}
          {cacheSection ? (
            <InsightsRecordSection
              section={cacheSection}
              sectionIndex={indexOf('Provider Prompt Cache')}
            />
          ) : null}
          {bodySection ? (
            <InsightsRecordSection
              section={bodySection}
              sectionIndex={indexOf('Request / Response')}
            />
          ) : null}
        </div>
        <div className="min-w-0 xl:col-start-2 xl:row-start-2">
          <InsightsRecordTrace record={record} trajectory={trajectory} error={traceError} />
        </div>
      </div>
    </PageContainer>
  )
}
