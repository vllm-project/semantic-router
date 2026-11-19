import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  ArrowLeft,
  Bot,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  Clock,
  Copy,
  FileText,
  Link2,
  ListTree,
  MessagesSquare,
  MoreHorizontal,
  SearchX,
  ShieldCheck,
  Timer,
  Zap,
} from 'lucide-react'

import {
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CopyButton,
  DetailSheet,
  DetailSheetBody,
  DetailSheetHeader,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  MetaItem,
  PageContainer,
  PageHeader,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  toast,
} from '../ui'
import { formatDateTime } from '../utils/dateTime'

import { compactIdentifier, displaySessionId } from './insightsAgentIdentity'
import { formatTokenCount } from './insightsDisplayFormat'
import {
  InsightsLoadError,
  InsightsUnavailable,
  RefreshButton,
  SummaryStrip,
  SummaryTile,
} from './InsightsDisplay'
import { HarnessLogo, HarnessName } from './InsightsHarness'
import { DEFAULT_PAGE_SIZE, pageHolding } from './insightsListData'
import { InsightsDataProviders, modelLabel, useModelLabels } from './insightsModelLabelSupport'
import {
  fetchInsightsJSON,
  InsightsRequestError,
  insightsUnavailableReason,
  type InsightsUnavailableReason,
} from './insightsPageApi'
import { getReviewsPath, getSessionRequestsPath } from './insightsReviewSupport'
import { RunDiagram, RunLegend, RunSegmentPanel } from './InsightsRunDiagram'
import { RunAgentsPanel, RunCallsPanel, RunOutcomesPanel } from './InsightsRunPanels'
import { RunEvents, RunStatus } from './InsightsRunsView'
import {
  countText,
  filterRunCalls,
  formatRunCost,
  formatRunDuration,
  runHeadline,
  type CallsView,
} from './insightsRunSupport'
import {
  buildRunTimeline,
  TIMELINE_SIZE,
  TIMELINE_SIZES,
  type TimelineBlock,
} from './insightsRunTimeline'
import type { RunCall, RunResponse } from './insightsRunTypes'

const RUNS_TAB = '/observe/requests?tab=runs'

function useRun(sessionId: string) {
  const [state, setState] = useState<{
    response: RunResponse | null
    error: string | null
    unavailable?: InsightsUnavailableReason
    notFound: boolean
    loading: boolean
  }>({ response: null, error: null, notFound: false, loading: true })
  const load = useCallback(async () => {
    if (!sessionId) {
      setState({ response: null, error: null, notFound: true, loading: false })
      return
    }
    setState((current) => ({ ...current, loading: true }))
    try {
      const query = new URLSearchParams({ session_id: sessionId })
      const response = await fetchInsightsJSON<RunResponse>(
        `/api/router/api/v1/observability/replays/run?${query.toString()}`,
        'agent run',
      )
      setState({ response, error: null, notFound: false, loading: false })
    } catch (error) {
      const reason = insightsUnavailableReason(error)
      const notFound = error instanceof InsightsRequestError && error.status === 404
      setState((current) => ({
        // a failed refresh keeps the run on screen under its error
        response: notFound ? null : current.response,
        error: reason ? null : error instanceof Error ? error.message : 'Unknown error',
        unavailable: notFound ? undefined : reason,
        notFound,
        loading: false,
      }))
    }
  }, [sessionId])
  useEffect(() => {
    void load()
  }, [load])
  return { ...state, load }
}

// One agent run, its models read by their labels.
export default function InsightsRunPage() {
  return (
    <InsightsDataProviders>
      <AgentRun />
    </InsightsDataProviders>
  )
}

function BackLink() {
  return (
    <Link
      to={RUNS_TAB}
      className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-3.5" />
      Agent runs
    </Link>
  )
}

function RunSkeleton() {
  return (
    <div className="flex flex-col gap-4" role="status" aria-label="Loading the run">
      <div className="flex items-start gap-3.5">
        <Skeleton className="size-[38px] rounded-lg" />
        <div className="flex flex-col gap-2">
          <Skeleton className="h-6 w-80" />
          <Skeleton className="h-4 w-96" />
        </div>
      </div>
      <Skeleton className="h-[86px] rounded-xl" />
      <Skeleton className="h-64 rounded-xl" />
      <Skeleton className="h-72 rounded-xl" />
    </div>
  )
}

/** The task a run was given, in full, beside the page. */
function TaskSheet({
  open,
  onOpenChange,
  title,
  task,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  task: string
}) {
  return (
    <DetailSheet open={open} onOpenChange={onOpenChange} size="lg">
      <DetailSheetHeader
        title="Task"
        description={
          <>
            <span className="min-w-0 truncate">{title}</span>
            <span aria-hidden>·</span>
            <span className="shrink-0">{task.length.toLocaleString()} characters</span>
          </>
        }
        actions={<CopyButton value={task} label="Copy the task" />}
      />
      <DetailSheetBody className="px-5 py-4">
        <p className="mb-2 text-xs text-subtle-foreground">
          The first prompt the main agent was given, as Replay kept it.
        </p>
        <div className="whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-subtle px-4 py-3 text-[13px] leading-relaxed text-foreground">
          {task}
        </div>
      </DetailSheetBody>
    </DetailSheet>
  )
}

const SIZE_ALL = 'all'

// One agent run: a compact header, its figures, its calls as lanes on a time
// axis that pages through long runs and folds idle time, and its calls,
// agents and outcomes as tables a page at a time.
function AgentRun() {
  const navigate = useNavigate()
  const labels = useModelLabels()
  const [searchParams] = useSearchParams()
  const sessionId = searchParams.get('session') ?? ''
  const step = searchParams.get('step') ?? undefined
  const { response, error, unavailable, notFound, loading, load } = useRun(sessionId)
  const calls = useMemo(() => response?.calls ?? [], [response])
  const total = calls.length
  const numbers = useMemo(
    () => new Map(calls.map((entry, index) => [entry.record_id, index + 1])),
    [calls],
  )

  const [taskOpen, setTaskOpen] = useState(false)
  const [tab, setTab] = useState('calls')
  const [blockKey, setBlockKey] = useState<string>()
  const [callId, setCallId] = useState<string | undefined>(step)
  const [windowSize, setWindowSize] = useState<number>()
  const [windowPage, setWindowPage] = useState<number>()
  const [view, setView] = useState<Omit<CallsView, 'page'> & { page?: number }>({
    agents: [],
    markers: [],
    size: DEFAULT_PAGE_SIZE,
  })

  // The timeline opens on the window that holds the step a link names, else the first.
  const size = Math.min(windowSize ?? (total > TIMELINE_SIZE ? TIMELINE_SIZE : total), total) || 1
  const pages = Math.max(1, Math.ceil(total / size))
  const stepIndex = step ? (numbers.get(step) ?? 0) - 1 : -1
  const windowAt = Math.min(windowPage ?? pageHolding(stepIndex, size), pages)
  const from = (windowAt - 1) * size
  const to = Math.min(total, from + size)
  const timeline = useMemo(
    () => (response ? buildRunTimeline(response, from, to) : null),
    [from, response, to],
  )
  const blocks = useMemo(() => timeline?.lanes.flatMap((lane) => lane.blocks) ?? [], [timeline])
  const block =
    blocks.find((entry) => entry.key === blockKey) ??
    blocks.find((entry) => entry.calls.some((one) => one.record_id === callId))
  const call =
    block?.calls.find((entry) => entry.record_id === callId) ??
    (block?.calls.length === 1 ? block.calls[0] : undefined)

  // The calls table opens on the page that holds the step a link names.
  const filtered = useMemo(() => filterRunCalls(calls, view), [calls, view])
  const callsPage =
    view.page ??
    pageHolding(
      filtered.findIndex((entry) => entry.record_id === step),
      view.size,
    )

  const selectBlock = (next: TimelineBlock) => {
    setBlockKey(next.key)
    setCallId(next.calls.length === 1 ? next.calls[0].record_id : undefined)
  }
  const selectCall = (next: RunCall) => {
    setBlockKey(blocks.find((entry) => entry.calls.includes(next))?.key)
    setCallId(next.record_id)
  }
  const showInTimeline = (next: RunCall) => {
    const index = (numbers.get(next.record_id) ?? 1) - 1
    setWindowPage(pageHolding(index, size))
    setBlockKey(undefined)
    setCallId(next.record_id)
    document.getElementById('run-timeline')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  const toggleAgent = (label: string) =>
    setView((current) => ({
      ...current,
      page: 1,
      agents: current.agents.includes(label)
        ? current.agents.filter((agent) => agent !== label)
        : [...current.agents, label],
    }))
  const followAgent = (label: string) => {
    setView((current) => ({ ...current, page: 1, agents: [label] }))
    setTab('calls')
  }

  if (!sessionId || notFound) {
    return (
      <PageContainer>
        <BackLink />
        <Card className="border-dashed shadow-none">
          <EmptyState
            icon={SearchX}
            tone="blue"
            title={sessionId ? 'No run under this session' : 'Pick an agent run'}
            description={
              sessionId
                ? `Replay holds no requests for session ${displaySessionId(sessionId)}. Its records may have expired, or Replay was off when it ran.`
                : 'Open a run from the Agent runs list to see its agents, their calls and what they cost.'
            }
            actions={<Button onClick={() => navigate(RUNS_TAB)}>Back to agent runs</Button>}
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
          <InsightsUnavailable reason={unavailable} noun="agent runs" onRetry={() => void load()} />
        </Card>
      </PageContainer>
    )
  }

  if (!response || !timeline) {
    return (
      <PageContainer>
        <BackLink />
        {error ? (
          <InsightsLoadError
            title="The run could not load"
            message={error}
            onRetry={() => void load()}
          />
        ) : null}
        {loading ? <RunSkeleton /> : null}
      </PageContainer>
    )
  }

  const run = response.run
  const headline = runHeadline(run)
  const task = run.prompt?.trim() ?? ''
  const subagents = run.agents.filter((entry) => entry.label !== 'main').length
  const topModel = run.models[0]
  const failures = (run.events?.failure ?? 0) + (run.events?.block ?? 0)
  const shownSession = displaySessionId(run.session_id)
  const reportedCount = run.outcome.reported?.length ?? 0

  return (
    <PageContainer key={run.session_id}>
      <BackLink />
      <PageHeader
        className="mb-5"
        icon={<HarnessLogo harness={run.harness} size={24} framed className="mt-0.5" />}
        title={
          <span className="flex min-w-0 items-center gap-3">
            <span className="min-w-0 truncate" title={run.session_name || undefined}>
              {headline}
            </span>
            <span className="shrink-0 text-[13px] font-normal tracking-normal">
              <RunStatus run={run} />
            </span>
          </span>
        }
        meta={
          <>
            <MetaItem>
              <HarnessName harness={run.harness} size={13} />
            </MetaItem>
            <MetaItem label="Session">
              <span className="inline-flex items-center gap-1">
                <span className="font-mono" title={run.session_id}>
                  {compactIdentifier(shownSession, 10, 8)}
                </span>
                <CopyButton value={run.session_id} label="Copy session ID" className="size-6" />
              </span>
            </MetaItem>
            <MetaItem icon={<Clock />}>{formatDateTime(run.started_at)}</MetaItem>
            {run.events ? <RunEvents events={run.events} /> : null}
            {task ? (
              <Button
                variant="ghost"
                size="xs"
                className="-ml-1.5 h-6 text-muted-foreground"
                onClick={() => setTaskOpen(true)}
              >
                <FileText />
                Read the task
              </Button>
            ) : null}
          </>
        }
        actions={
          <>
            <Button
              variant="outline"
              onClick={() => navigate(getSessionRequestsPath(run.session_id))}
            >
              <ListTree />
              View requests
            </Button>
            <RefreshButton onClick={() => void load()} loading={loading} />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" aria-label="More actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onSelect={() => navigate(getReviewsPath(run.session_id))}>
                  <ShieldCheck />
                  Approval reviews
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={() => {
                    void navigator.clipboard?.writeText(window.location.href).catch(() => undefined)
                    toast.success('Link copied')
                  }}
                >
                  <Link2 />
                  Copy link
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    void navigator.clipboard?.writeText(run.session_id).catch(() => undefined)
                    toast.success('Session ID copied', { description: run.session_id })
                  }}
                >
                  <Copy />
                  Copy session ID
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />
      {task ? (
        <TaskSheet open={taskOpen} onOpenChange={setTaskOpen} title={headline} task={task} />
      ) : null}
      {error ? (
        <InsightsLoadError
          title="The run could not refresh"
          message={error}
          onRetry={() => void load()}
        />
      ) : null}

      <SummaryStrip label="Run figures" className="mb-5 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
        <SummaryTile
          label="Duration"
          icon={<Timer className="text-chart-1" />}
          value={formatRunDuration(run.duration_ms)}
          hint="First call to last"
        />
        <SummaryTile
          label="Calls"
          icon={<Zap className="text-chart-2" />}
          value={run.calls.toLocaleString()}
          hint={failures ? `${failures.toLocaleString()} failed or blocked` : 'None failed'}
          tone={failures ? 'negative' : undefined}
        />
        <SummaryTile
          label="Agents"
          icon={<Bot className="text-chart-3" />}
          value={run.agents.length.toLocaleString()}
          hint={subagents ? `Main and ${countText(subagents, 'subagent')}` : 'Main agent only'}
        />
        <SummaryTile
          label="Models"
          icon={<MessagesSquare className="text-chart-4" />}
          value={run.models.length.toLocaleString()}
          hint={
            topModel
              ? `Mostly ${modelLabel(labels, topModel.name) || topModel.name}`
              : 'No model answered'
          }
        />
        <SummaryTile
          label="Cost"
          icon={<CircleDollarSign className="text-chart-6" />}
          value={formatRunCost(run.cost)}
          hint={
            run.cost.by_currency?.some((entry) => entry.router_call_spend > 0)
              ? 'Router calls included'
              : 'Configured-rate estimate'
          }
        />
        <SummaryTile
          label="Tokens"
          icon={<MessagesSquare className="text-chart-5" />}
          value={`${formatTokenCount(run.cost.prompt_tokens)} / ${formatTokenCount(run.cost.completion_tokens)}`}
          hint="Input / output"
        />
      </SummaryStrip>

      <Card id="run-timeline" className="mb-5 scroll-mt-6">
        <CardHeader>
          <CardTitle>Timeline</CardTitle>
          <CardDescription className="col-start-1">
            One lane per agent; a block is a stretch of calls on one model. Click a block for its
            calls, an agent to follow its calls below.
          </CardDescription>
          {total > TIMELINE_SIZES[0] ? (
            <CardAction className="gap-2 max-sm:col-start-1 max-sm:row-span-1 max-sm:row-start-3 max-sm:justify-self-start">
              <span className="whitespace-nowrap text-xs text-subtle-foreground tabular">
                Calls {(from + 1).toLocaleString()}–{to.toLocaleString()} of{' '}
                {total.toLocaleString()}
              </span>
              <Button
                variant="outline"
                size="icon-xs"
                aria-label="Earlier calls"
                disabled={windowAt <= 1}
                onClick={() => setWindowPage(windowAt - 1)}
              >
                <ChevronLeft />
              </Button>
              <Button
                variant="outline"
                size="icon-xs"
                aria-label="Later calls"
                disabled={windowAt >= pages}
                onClick={() => setWindowPage(windowAt + 1)}
              >
                <ChevronRight />
              </Button>
              <Select
                value={size >= total ? SIZE_ALL : String(size)}
                onValueChange={(value) => {
                  const next = value === SIZE_ALL ? total : Number(value)
                  setWindowSize(next)
                  setWindowPage(pageHolding(from, next))
                }}
              >
                <SelectTrigger size="sm" aria-label="Calls in view" className="h-7 px-2 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent align="end">
                  {TIMELINE_SIZES.filter((option) => option < total).map((option) => (
                    <SelectItem key={option} value={String(option)}>
                      {option} calls
                    </SelectItem>
                  ))}
                  <SelectItem value={SIZE_ALL}>All {total.toLocaleString()} calls</SelectItem>
                </SelectContent>
              </Select>
            </CardAction>
          ) : null}
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <RunDiagram
            run={run}
            timeline={timeline}
            selected={block?.key}
            agents={view.agents}
            onSelect={selectBlock}
            onAgent={toggleAgent}
          />
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5">
            <RunLegend run={run} />
            <span className="text-xs text-subtle-foreground tabular">
              {formatRunDuration(timeline.activeMs)} of activity in view
              {timeline.breaks.length === 1 ? ' · 1 idle stretch folded' : ''}
              {timeline.breaks.length > 1
                ? ` · ${timeline.breaks.length} idle stretches folded`
                : ''}
            </span>
          </div>
          {block ? (
            <div className="border-t border-border pt-4">
              <RunSegmentPanel
                segment={block}
                call={call}
                numbers={numbers}
                onSelectCall={selectCall}
              />
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Tabs value={tab} onValueChange={setTab} variant="line">
        <TabsList aria-label="Run details">
          <TabsTrigger value="calls">
            Calls
            <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
              {total.toLocaleString()}
            </span>
          </TabsTrigger>
          <TabsTrigger value="agents">
            Agents
            <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
              {run.agents.length.toLocaleString()}
            </span>
          </TabsTrigger>
          <TabsTrigger value="outcomes">
            Outcomes
            {reportedCount ? (
              <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
                {reportedCount.toLocaleString()}
              </span>
            ) : null}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="calls">
          <RunCallsPanel
            run={run}
            calls={calls}
            numbers={numbers}
            view={{ ...view, page: callsPage }}
            onView={(patch) => setView((current) => ({ ...current, page: callsPage, ...patch }))}
            activeId={call?.record_id ?? step}
            onShow={showInTimeline}
          />
        </TabsContent>
        <TabsContent value="agents">
          <RunAgentsPanel run={run} onFollow={followAgent} />
        </TabsContent>
        <TabsContent value="outcomes">
          <RunOutcomesPanel run={run} calls={calls} numbers={numbers} />
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}
