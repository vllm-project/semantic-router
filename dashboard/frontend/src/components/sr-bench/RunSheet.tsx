import { useState } from 'react'
import {
  ArrowLeft,
  ArrowUpRight,
  Copy,
  FlaskConical,
  GitCompareArrows,
  MoreHorizontal,
  RefreshCw,
  RotateCw,
  Search,
  Split,
} from 'lucide-react'
import {
  Badge,
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  IconTile,
  InputGroup,
  KeyValueGrid,
  Progress,
  Skeleton,
  toast,
  useRetainedValue,
} from '../../ui'
import { benchApi, SR_BENCH_API } from './api'
import { active, money, number, seconds } from './model'
import type { CaseResult, Run } from './types'
import { targetName } from './targetPresentation'
import { canReuseBaseline } from './baselineReuse'
import { useRunEvidence } from './useRunEvidence'
import RunArtifacts from './RunArtifacts'
import CallEvidence from './CallEvidence'
import RunEvents from './RunEvents'
import BenchPagination from './BenchPagination'
import PreviewEvidence from './PreviewEvidence'
import RunRecovery from './RunRecovery'
import RunLineage from './RunLineage'
import RunElapsed from './RunElapsed'
import RunActivity from './RunActivity'
import RecipeEvidence from './RecipeEvidence'
import RunResults from './RunResults'
import { BenchmarkChips, TargetChip } from './evalShared'
import { profileTitle } from './datasetPresentation'
import { experimentQuestionCount } from './experimentEvidencePresentation'
import {
  RUN_SECTIONS,
  bestScore,
  runModeLabel,
  runProgressShare,
  runStatusLabel,
  runStatusTone,
  type RunSection,
} from './runPresentation'
import { BenchLoading, Disclosure, Notice, SummaryStat, TableFrame } from './benchUi'
import {
  benchTable,
  benchText,
  codeBlockClass,
  lineTab,
  lineTabList,
  linkButtonClass,
} from './benchStyles'

const SECTION_LABELS: Record<RunSection, string> = {
  results: 'Results',
  questions: 'Questions',
  calls: 'Calls',
  activity: 'Activity',
  protocol: 'Protocol',
}

const progressTone = {
  success: 'success',
  danger: 'danger',
  warning: 'warning',
  info: 'brand',
  neutral: 'neutral',
} as const

function questionText(run: Run, caseID: string): string {
  const source = run.manifest.cases?.find(
    (value) => value && typeof value === 'object' && (value as { id?: string }).id === caseID,
  ) as { messages?: Array<{ role?: string; content?: unknown }> } | undefined
  return (
    source?.messages
      ?.filter((message) => message.role === 'user')
      .map((message) =>
        typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
      )
      .join('\n\n') || 'Question text is unavailable in this saved manifest.'
  )
}

const usd = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value)
    ? `$${value.toFixed(value !== 0 && Math.abs(value) < 0.01 ? 4 : 2)}`
    : '—'

export interface RunSheetProps {
  id: string | null
  onClose: () => void
  section: RunSection
  onSectionChange: (section: RunSection) => void
  actorID: string
  canRun: boolean
  onChanged: () => void
  onRecovered: (run: Run) => void
  onCandidate: (id: string, mode: 'live' | 'preview') => void
  onCompare: (run: Run) => void
  onRerun: (run: Run) => void
}

export default function RunSheet(props: RunSheetProps) {
  const retained = useRetainedValue(props.id)
  return (
    <DetailSheet open={!!props.id} onOpenChange={(open) => !open && props.onClose()} size="xl">
      {retained ? (
        <RunSheetContent key={`${props.actorID}:${retained}`} {...props} id={retained} />
      ) : null}
    </DetailSheet>
  )
}

function RunSheetContent({
  id,
  onClose,
  section,
  onSectionChange,
  actorID,
  canRun,
  onChanged,
  onRecovered,
  onCandidate,
  onCompare,
  onRerun,
}: RunSheetProps & { id: string }) {
  const [pending, setPending] = useState(false)
  const [filter, setFilter] = useState('')
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<CaseResult | null>(null)
  const [revision, setRevision] = useState(0)
  const [actionError, setActionError] = useState('')
  const {
    run,
    readAt,
    report,
    reportRead,
    results,
    events,
    eventsPage,
    loadMoreEvents,
    calls,
    error: readError,
    resultsPage,
    callsPage,
    loadMoreResults,
    loadMoreCalls,
  } = useRunEvidence(id, revision)
  const error = actionError || readError

  function reloadEvidence() {
    setPage(0)
    setSelected(null)
    setRevision((value) => value + 1)
  }

  async function cancel() {
    setPending(true)
    setActionError('')
    try {
      await benchApi.cancel(id)
      reloadEvidence()
      onChanged()
    } catch (cause) {
      setActionError(
        cause instanceof Error
          ? cause.message
          : 'Cancellation failed. Refresh this run before taking another action.',
      )
    } finally {
      setPending(false)
    }
  }

  const visible = results.filter((result) =>
    `${result.case_id} ${result.target_id} ${targetName(run?.manifest, result.target_id)} ${result.benchmark} ${result.status}`
      .toLowerCase()
      .includes(filter.toLowerCase()),
  )
  const currentPage = Math.min(page, Math.max(0, Math.ceil(visible.length / 25) - 1))
  const metrics = report?.summary.targets ?? run?.summary?.targets ?? []
  const best = run?.manifest.mode === 'live' ? bestScore(metrics, run.manifest) : null
  const share = run ? Math.round(runProgressShare(run) * 100) : 0
  const reusable = !!run && canRun && canReuseBaseline(run)
  const sections = RUN_SECTIONS

  const header = (
    <DetailSheetHeader
      icon={<IconTile icon={FlaskConical} tone="orange" size="lg" />}
      title={run?.manifest.name ?? 'Evaluation run'}
      badges={
        run ? (
          <>
            <Badge tone={runStatusTone(run.status)} dot>
              {runStatusLabel(run.status)}
            </Badge>
            <Badge tone="outline">{profileTitle(run.manifest.profile)}</Badge>
            {run.manifest.mode !== 'live' ? (
              <Badge tone="violet">{runModeLabel(run.manifest.mode)}</Badge>
            ) : null}
          </>
        ) : null
      }
      description={
        run ? (
          <>
            <span className="font-mono">{run.id}</span>
            <span className="text-subtle-foreground">·</span>
            <span>Created {new Date(run.created_at).toLocaleString()}</span>
            <span className="text-subtle-foreground">·</span>
            <span>
              {run.manifest.targets.length}{' '}
              {run.manifest.targets.length === 1 ? 'target' : 'targets'}
            </span>
          </>
        ) : undefined
      }
      actions={
        run ? (
          <>
            {run.manifest.mode === 'live' && (
              <Button variant="outline" size="sm" onClick={() => onCompare(run)}>
                <GitCompareArrows aria-hidden="true" />
                Compare with…
              </Button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label="More run actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                {canRun && run.manifest.mode !== 'replay' ? (
                  <DropdownMenuItem onSelect={() => onRerun(run)}>
                    <RotateCw aria-hidden="true" /> Re-run
                  </DropdownMenuItem>
                ) : null}
                {reusable ? (
                  <DropdownMenuItem onSelect={() => onCandidate(run.id, 'preview')}>
                    <Split aria-hidden="true" /> Preview candidate
                  </DropdownMenuItem>
                ) : null}
                <DropdownMenuItem onSelect={reloadEvidence}>
                  <RefreshCw aria-hidden="true" /> Reload evidence
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    void navigator.clipboard?.writeText(run.id).catch(() => undefined)
                    toast.success('Run ID copied')
                  }}
                >
                  <Copy aria-hidden="true" /> Copy run ID
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <a
                    href={`${SR_BENCH_API}/runs/${encodeURIComponent(id)}/report`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <ArrowUpRight aria-hidden="true" /> Report JSON
                  </a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        ) : null
      }
    />
  )

  if (!run)
    return (
      <>
        {header}
        <DetailSheetBody className="px-5 py-5">
          {error && (
            <Notice tone="danger" as="p" role="alert" className="mt-0">
              {error}
            </Notice>
          )}
          {readError ? (
            <div role="status" className="flex flex-col items-start gap-2">
              <p className={benchText.muted}>Run details are unavailable.</p>
              <Button variant="outline" size="sm" onClick={reloadEvidence}>
                Try again
              </Button>
            </div>
          ) : (
            <BenchLoading label="Loading run…" lines={4} />
          )}
        </DetailSheetBody>
      </>
    )

  return (
    <>
      {header}
      <DetailSheetBody>
        <div className="flex flex-col gap-3 px-5 pt-4">
          {error && (
            <Notice tone="danger" as="p" role="alert" className="my-0">
              {error}
            </Notice>
          )}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <SummaryStat
              label="Progress"
              value={`${share}%`}
              hint={`${number(run.progress.completed)} of ${number(run.progress.total)} attempts · ${number(run.progress.failed)} failed`}
            >
              <Progress
                className="mt-2 h-1"
                value={share}
                tone={progressTone[runStatusTone(run.status)]}
                aria-label="Evaluation progress"
              />
            </SummaryStat>
            <SummaryStat
              label="Best score"
              value={
                best
                  ? best.score.toFixed(1)
                  : run.manifest.mode !== 'live'
                    ? 'No score'
                    : active(run.status)
                      ? 'Pending'
                      : '—'
              }
              hint={
                best
                  ? best.target
                  : run.manifest.mode !== 'live'
                    ? `${runModeLabel(run.manifest.mode)} runs are not scored`
                    : 'Waits for every planned outcome'
              }
            />
            <SummaryStat
              aria-label="Total recorded cost"
              label="Cost"
              value={usd(report?.summary.total_spend_usd)}
              hint="Recorded, with evaluation overhead"
            />
            <RunElapsed
              run={run}
              savedSeconds={report?.summary.wall_time_s ?? run.summary?.wall_time_s}
            />
          </div>
          {run.status !== 'completed' && (
            <Notice tone={active(run.status) ? 'info' : 'warning'} as="p" className="my-0">
              This run is {run.status}. {number(run.progress.failed)} failed results remain part of
              the saved evidence. Comparisons require an explicit terminal outcome for every planned
              result; missing or ungraded results are not eligible.
            </Notice>
          )}
          {(run.error || report?.failure) && (
            <Notice tone="danger" role="alert" className="my-0">
              <p className="font-medium text-foreground">{run.error || report?.failure?.reason}</p>
              {!run.error && report?.failure && (
                <p>
                  Model {targetName(run.manifest, report.failure.target_id)} · case{' '}
                  {report.failure.case_id}
                  {report.failure.inferred_from_saved_results ? ' · from saved case evidence' : ''}
                </p>
              )}
            </Notice>
          )}
        </div>
        <nav
          className={`${lineTabList} sticky top-0 z-10 mb-0 mt-4 bg-card px-5`}
          role="tablist"
          aria-label="Run sections"
        >
          {sections.map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              className={lineTab}
              id={`run-tab-${value}`}
              tabIndex={section === value ? 0 : -1}
              onKeyDown={(event) => {
                if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return
                event.preventDefault()
                const tabs = Array.from(
                  event.currentTarget
                    .closest('[role=tablist]')!
                    .querySelectorAll<HTMLButtonElement>('[role=tab]'),
                )
                const index = tabs.indexOf(event.currentTarget)
                const next =
                  event.key === 'Home'
                    ? 0
                    : event.key === 'End'
                      ? tabs.length - 1
                      : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length
                tabs[next].focus()
                tabs[next].click()
              }}
              aria-selected={section === value}
              aria-controls={`run-panel-${value}`}
              onClick={() => onSectionChange(value)}
            >
              {SECTION_LABELS[value]}
            </button>
          ))}
        </nav>
        <div className="px-5 pb-6 pt-4">
          <div
            role="tabpanel"
            id="run-panel-results"
            aria-labelledby="run-tab-results"
            hidden={section !== 'results'}
            className="min-w-0"
          >
            <RunResults
              run={run}
              report={report}
              reportRead={reportRead}
              visible={section === 'results'}
              onRetry={reloadEvidence}
              recovery={
                run.manifest.mode === 'live' &&
                !active(run.status) &&
                run.status !== 'completed' && (
                  <RunRecovery
                    key={`${actorID}:${run.id}`}
                    run={run}
                    actorID={actorID}
                    canRun={canRun}
                    onRecovered={onRecovered}
                  />
                )
              }
            />
          </div>
          <div
            role="tabpanel"
            id="run-panel-questions"
            aria-labelledby="run-tab-questions"
            hidden={section !== 'questions'}
            className="min-w-0"
          >
            <div hidden={!!selected}>
              <div className="mb-1 flex flex-wrap items-end justify-between gap-3">
                <h3 id="run-cases" className={benchText.h3}>
                  Case results
                </h3>
                <InputGroup
                  type="search"
                  aria-label="Filter loaded results"
                  value={filter}
                  leading={<Search aria-hidden="true" />}
                  containerClassName="w-full sm:w-72"
                  className="h-8"
                  onChange={(event) => {
                    setFilter(event.target.value)
                    setPage(0)
                  }}
                  placeholder="Target, case, benchmark or status"
                />
              </div>
              {resultsPage.total !== null && (
                <p className={benchText.muted}>
                  Loaded {number(results.length)} of {number(resultsPage.total)} persisted results.
                  Filtering applies to loaded results. Scores and costs in Results use the full
                  report.
                </p>
              )}
              {resultsPage.loading && resultsPage.total === null && (
                <BenchLoading label="Loading persisted case results…" lines={3} />
              )}
              {report?.provenance.accounting_correction && (
                <Notice tone="info" as="p">
                  Original receipt accounting is shown in case and call details. See the report
                  summary for reconciled tokens and costs; original evidence has not been
                  overwritten.
                </Notice>
              )}
              <TableFrame>
                <table className={benchTable}>
                  <thead>
                    <tr>
                      <th>Case</th>
                      <th>Benchmark</th>
                      <th>Target</th>
                      <th>Status</th>
                      <th>Score</th>
                      <th>
                        {report?.provenance.accounting_correction ? 'Cost (original)' : 'Cost'}
                      </th>
                      <th>Latency / TTFT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.slice(currentPage * 25, (currentPage + 1) * 25).map((result, i) => (
                      <tr key={`${result.case_id}-${result.target_id}-${i}`}>
                        <td>
                          <button
                            className={`${linkButtonClass} font-mono text-xs`}
                            onClick={() => setSelected(result)}
                          >
                            {result.case_id}
                          </button>
                        </td>
                        <td>{result.benchmark}</td>
                        <td>{targetName(run.manifest, result.target_id)}</td>
                        <td>{result.status}</td>
                        <td>{number(result.score, 3)}</td>
                        <td>{money(result.cost_usd)}</td>
                        <td>
                          {seconds(result.latency_s)} / {seconds(result.ttft_s)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableFrame>
              {!visible.length &&
                !resultsPage.loading &&
                !resultsPage.error &&
                resultsPage.total !== null && (
                  <p className={benchText.muted}>
                    {resultsPage.total === 0
                      ? 'No persisted case results yet.'
                      : 'No matching persisted results.'}
                  </p>
                )}
              <BenchPagination
                label="Results"
                total={visible.length}
                page={currentPage}
                pageSize={25}
                onChange={setPage}
              />
              {resultsPage.error && (
                <Notice tone="danger" as="p" role="alert">
                  {resultsPage.error}
                </Notice>
              )}
              {resultsPage.nextCursor !== null && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={resultsPage.loading}
                  onClick={() => void loadMoreResults()}
                >
                  {resultsPage.loading ? 'Loading results…' : 'Load more results'}
                </Button>
              )}
            </div>
            {selected && (
              <div className="min-w-0">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <h3 className={benchText.h3}>
                    {selected.case_id} · {targetName(run.manifest, selected.target_id)}
                  </h3>
                  <Button variant="outline" size="sm" onClick={() => setSelected(null)}>
                    <ArrowLeft aria-hidden="true" />
                    Back to questions
                  </Button>
                </div>
                <h4 className={`${benchText.h4} mt-3`}>Question</h4>
                <pre className={codeBlockClass}>{questionText(run, selected.case_id)}</pre>
                {selected.error && (
                  <Notice tone="danger" as="p">
                    {selected.error}
                  </Notice>
                )}
                {run.manifest.mode === 'preview' ? (
                  <PreviewEvidence routing={selected.details?.routing} />
                ) : (
                  <>
                    <h4 className={`${benchText.h4} mt-4`}>
                      {['mmlu-pro', 'gpqa-diamond'].includes(selected.benchmark)
                        ? 'Parsed answer'
                        : 'Recorded answer'}
                    </h4>
                    <pre className={codeBlockClass}>
                      {selected.answer == null || selected.answer === ''
                        ? ['mmlu-pro', 'gpqa-diamond'].includes(selected.benchmark)
                          ? 'No answer parsed.'
                          : 'No answer recorded for this case.'
                        : typeof selected.answer === 'string'
                          ? selected.answer
                          : JSON.stringify(selected.answer, null, 2)}
                    </pre>
                    {selected.details?.quality_failure === 'output_limit' ? (
                      <Notice tone="warning" as="p">
                        The output limit was reached. This case counts as incorrect under the frozen
                        protocol.
                      </Notice>
                    ) : selected.details?.strict_format === false ? (
                      <Notice tone="warning" as="p">
                        The final text did not match the required answer format. Inspect the
                        original call for the full response.
                      </Notice>
                    ) : null}
                  </>
                )}
                <Disclosure summary="Usage, trace and grading evidence">
                  <pre className={codeBlockClass}>{JSON.stringify(selected, null, 2)}</pre>
                </Disclosure>
              </div>
            )}
          </div>
          <div
            role="tabpanel"
            id="run-panel-calls"
            aria-labelledby="run-tab-calls"
            hidden={section !== 'calls'}
            className="min-w-0"
          >
            <CallEvidence
              id={id}
              calls={calls}
              manifest={run.manifest}
              page={callsPage}
              loadMore={loadMoreCalls}
              accountingReconciled={!!report?.provenance.accounting_correction}
            />
          </div>
          <div
            role="tabpanel"
            id="run-panel-activity"
            aria-labelledby="run-tab-activity"
            hidden={section !== 'activity'}
            className="flex min-w-0 flex-col gap-1"
          >
            <RunActivity run={run} revision={revision} />
            <RunLineage report={report} />
            <RunEvents
              manifest={run.manifest}
              events={events}
              page={eventsPage}
              loadMore={loadMoreEvents}
            />
          </div>
          <div
            role="tabpanel"
            id="run-panel-protocol"
            aria-labelledby="run-tab-protocol"
            hidden={section !== 'protocol'}
            className="flex min-w-0 flex-col gap-1"
          >
            <h3 className={benchText.h3}>Frozen protocol</h3>
            <p className={benchText.muted}>
              Everything this run was measured under. A re-run or candidate reuses it, so results
              stay comparable.
            </p>
            <KeyValueGrid
              className="my-3"
              items={[
                { label: 'Size', value: profileTitle(run.manifest.profile) },
                {
                  label: 'Questions',
                  value: (() => {
                    const count = experimentQuestionCount(run)
                    return count === undefined ? '—' : number(count)
                  })(),
                },
                { label: 'Type', value: runModeLabel(run.manifest.mode) },
                {
                  label: 'Benchmarks',
                  value: <BenchmarkChips ids={run.manifest.dataset?.benchmarks ?? []} max={6} />,
                },
                {
                  label: 'Targets',
                  value: (
                    <span className="flex flex-wrap gap-1">
                      {run.manifest.targets.map((target) => (
                        <TargetChip key={target.id} target={target} />
                      ))}
                    </span>
                  ),
                },
                {
                  label: 'Limits',
                  value: [
                    run.manifest.cost_policy === 'capability_only'
                      ? 'No USD cap'
                      : `${usd(run.manifest.limits?.max_cost_usd)} budget`,
                    `${number(run.manifest.limits?.max_run_seconds)} s deadline`,
                    `${number(run.manifest.limits?.concurrency)} concurrent`,
                  ].join(' · '),
                },
                {
                  label: 'Sampling',
                  value: `temperature ${run.manifest.sampling?.temperature ?? '—'} · top_p ${run.manifest.sampling?.top_p ?? '—'} · seed ${run.manifest.sampling?.seed ?? run.manifest.seed ?? '—'}`,
                },
                {
                  label: 'Dataset',
                  value: run.manifest.dataset?.sha256 ?? '—',
                  mono: true,
                  span: true,
                },
              ]}
            />
            {run.manifest.targets.some((target) => target.kind === 'mom') &&
              (report ? (
                <RecipeEvidence report={report} targets={run.manifest.targets} />
              ) : reportRead.error ? (
                <Notice tone="danger" as="p">
                  Recipe evidence is unavailable.
                </Notice>
              ) : (
                <BenchLoading label="Loading recipe evidence…" lines={3} />
              ))}
            <Disclosure summary="Frozen manifest and provenance" summaryId="run-provenance">
              <pre className={codeBlockClass}>{JSON.stringify(run.manifest, null, 2)}</pre>
              <pre className={codeBlockClass}>
                {JSON.stringify(report?.provenance ?? {}, null, 2)}
              </pre>
            </Disclosure>
            {run.status === 'completed' && run.manifest.mode === 'live' && (
              <RunArtifacts id={id} canRun={canRun} />
            )}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {[
                [`${SR_BENCH_API}/runs/${encodeURIComponent(id)}/report`, 'Open report JSON'],
                [
                  `${SR_BENCH_API}/runs/${encodeURIComponent(id)}/results`,
                  'Open first results page',
                ],
                [
                  `${SR_BENCH_API}/runs/${encodeURIComponent(id)}/events?after=0`,
                  'Open event ledger',
                ],
              ].map(([href, label]) => (
                <Button key={href} variant="ghost" size="sm" asChild>
                  <a href={href} target="_blank" rel="noreferrer">
                    {label} <ArrowUpRight aria-hidden="true" />
                  </a>
                </Button>
              ))}
            </div>
          </div>
        </div>
      </DetailSheetBody>
      <DetailSheetFooter>
        <span className="min-w-0 truncate text-xs text-muted-foreground" role="status">
          {readAt ? `Synced ${readAt}` : <Skeleton className="h-3 w-24" />}
          {active(run.status) ? ' · updates while running' : ''}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {active(run.status) && (
            <Button
              variant="destructive-ghost"
              size="sm"
              disabled={!canRun || pending}
              onClick={() => void cancel()}
            >
              {pending ? 'Stopping…' : 'Cancel evaluation'}
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
          {reusable ? (
            <Button size="sm" onClick={() => onCandidate(run.id, 'live')}>
              <FlaskConical aria-hidden="true" /> Evaluate candidate
            </Button>
          ) : canRun && run.manifest.mode !== 'replay' && !active(run.status) ? (
            <Button size="sm" onClick={() => onRerun(run)}>
              <RotateCw aria-hidden="true" /> Re-run
            </Button>
          ) : null}
        </div>
      </DetailSheetFooter>
    </>
  )
}
