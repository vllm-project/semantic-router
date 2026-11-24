import { useMemo, useState } from 'react'
import { ArrowUpRight, ChartNoAxesColumn, Search, TriangleAlert } from 'lucide-react'

import { Badge, Collapsible, CollapsibleContent, CollapsibleTrigger, EmptyState, cn } from '../ui'
import {
  evaluationSourceLabel,
  filterEvaluationDetails,
  type EffectiveEvaluationGroup,
  type EffectiveEvaluationRecord,
} from './configPageEffectiveEvaluations'
import {
  modelHubBenchmarkRawValueLabel,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'

const readable = (value: string) => value.split('_').join(' ')

const statusTone = {
  available: 'success',
  missing: 'neutral',
  failed: 'danger',
  not_applicable: 'neutral',
  withheld: 'warning',
} as const

function SourceReference({ value, label }: { value: string; label?: string }) {
  return /^https?:\/\//i.test(value) ? (
    <a
      href={value}
      target="_blank"
      rel="noreferrer"
      className="inline-flex min-w-0 items-center gap-0.5 font-medium text-primary hover:underline"
    >
      <span className="truncate">{label ?? value}</span>
      <ArrowUpRight className="size-3 shrink-0" />
    </a>
  ) : (
    <span className="min-w-0 truncate font-mono">{value}</span>
  )
}

function MetricValue({
  name,
  value,
  record,
}: {
  name: string
  value: number
  record: EffectiveEvaluationRecord
}) {
  const definition = record.benchmark?.metrics.find((metric) => metric.id === name)
  const label = definition
    ? modelHubBenchmarkValueLabel(value, definition)
    : Number.isInteger(value)
      ? value.toLocaleString()
      : String(value)
  const raw = definition ? modelHubBenchmarkRawValueLabel(value, definition) : undefined
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <dt className="truncate text-[11px] text-subtle-foreground">{readable(name)}</dt>
      <dd
        className="font-mono text-[13px] font-medium tabular"
        title={raw ? `Raw: ${raw}` : undefined}
      >
        {label}
      </dd>
      <dd className="text-[11px] text-subtle-foreground">
        {definition
          ? `${definition.unit} · ${definition.range[0]}–${definition.range[1]}`
          : 'unit not defined'}
      </dd>
    </div>
  )
}

export function EvaluationEvidenceCard({ record }: { record: EffectiveEvaluationRecord }) {
  const { evaluation, benchmark, issues } = record
  const metrics = Object.entries(evaluation.metrics)
  const subject = Object.keys(evaluation.subject).length > 0 ? evaluation.subject : null
  return (
    <article className="flex flex-col gap-3 rounded-lg border border-border bg-card p-3.5">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-[13px] font-medium">
            {benchmark?.display_name || evaluation.benchmark}
          </span>
          <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
            {evaluation.benchmark}
          </span>
        </div>
        <span className="flex shrink-0 flex-wrap items-center gap-1.5">
          <Badge size="sm" tone={record.origin === 'built_in' ? 'outline' : 'brand'}>
            {evaluationSourceLabel(record)}
          </Badge>
          <Badge size="sm" tone={statusTone[evaluation.status]} dot className="capitalize">
            {readable(evaluation.status)}
          </Badge>
        </span>
      </header>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span>
          Profile{' '}
          <span className="text-foreground/85">
            {evaluation.benchmark_profile || 'not resolved'}
          </span>
        </span>
        <span>
          Effort <span className="text-foreground/85">{evaluation.reasoning_effort}</span>
        </span>
      </div>
      {metrics.length ? (
        <dl className="grid grid-cols-2 gap-3 rounded-md bg-surface-subtle/70 p-2.5 sm:grid-cols-3">
          {metrics.map(([name, value]) => (
            <MetricValue key={name} name={name} value={value} record={record} />
          ))}
        </dl>
      ) : (
        <p className="text-xs text-muted-foreground">No metric values recorded</p>
      )}
      {issues.map((issue) => (
        <p
          key={issue}
          className="flex items-start gap-1.5 rounded-md bg-warning-soft px-2.5 py-1.5 text-xs text-warning-foreground"
        >
          <TriangleAlert className="mt-px size-3.5 shrink-0" />
          {issue}
        </p>
      ))}
      <footer className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-t border-border pt-2.5 text-xs text-muted-foreground">
        <span>
          {readable(evaluation.evidence.provenance)} · {evaluation.evidence.verification}
          {evaluation.measured_at ? ` · measured ${evaluation.measured_at}` : ''}
          {evaluation.observed_at ? ` · observed ${evaluation.observed_at}` : ''}
        </span>
        {evaluation.evidence.source ? (
          <SourceReference value={evaluation.evidence.source} label="Source" />
        ) : (
          <span className="text-subtle-foreground">Source not recorded</span>
        )}
      </footer>
      {evaluation.evidence.artifact ? (
        <div className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
          Artifact <SourceReference value={evaluation.evidence.artifact} />
        </div>
      ) : null}
      {subject ? (
        <Collapsible>
          <CollapsibleTrigger className="text-xs font-medium text-muted-foreground hover:text-foreground">
            Measurement details
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-surface-sunken p-2.5 font-mono text-[11.5px] text-foreground/85">
              {JSON.stringify(subject, null, 2)}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      ) : null}
    </article>
  )
}

/** A model's benchmark evidence: catalog results and configured records, never ranked. */
export default function ConfigPageEvaluationDetails({
  group,
}: {
  group: EffectiveEvaluationGroup
}) {
  const [search, setSearch] = useState('')
  const records = useMemo(
    () => filterEvaluationDetails(group.records, search),
    [group.records, search],
  )
  if (group.records.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong">
        <EmptyState
          compact
          icon={ChartNoAxesColumn}
          tone="orange"
          title="No published evidence for this model"
          description="No built-in or configured evaluation evidence is available for this model."
        />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      {group.records.length > 3 ? (
        <label className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-subtle-foreground" />
          <span className="sr-only">Search evidence</span>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search benchmark, effort, status or source…"
            className={cn(
              'h-8 w-full rounded-md border border-input bg-surface pl-8 pr-3 text-[13px] shadow-xs outline-none placeholder:text-subtle-foreground',
              'focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40',
            )}
          />
        </label>
      ) : null}
      {records.map((record) => (
        <EvaluationEvidenceCard key={record.key} record={record} />
      ))}
      {records.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-muted-foreground">
          No evidence matches this search.
        </p>
      ) : null}
    </div>
  )
}
