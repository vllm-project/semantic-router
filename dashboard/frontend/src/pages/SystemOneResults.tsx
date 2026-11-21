import { Check, CircleDot } from 'lucide-react'
import { Badge, KeyValueGrid, SectionLabel, cn } from '../ui'
import {
  finiteNumber,
  probabilities,
  probabilityLabel,
  spanSegments,
  spanSource,
  type SystemOneAnswer,
  type SystemOneQuestion,
} from './systemOnePlayground'
import { QUESTION_ICONS } from './decisionQuestionTypes'
import type { SystemOneRun } from './useSystemOnePlayground'

function Probability({
  label,
  value,
  highlighted,
}: {
  label: string
  value: number
  highlighted?: boolean
}) {
  const percent = Math.min(100, Math.max(0, value * 100))
  return (
    <div className="grid grid-cols-[minmax(0,13rem)_minmax(4rem,1fr)_3.5rem] items-center gap-3">
      <span
        className={cn(
          'flex min-w-0 items-center gap-1.5 font-mono text-[12px]',
          highlighted ? 'font-medium text-foreground' : 'text-muted-foreground',
        )}
      >
        {highlighted ? (
          <Check className="size-3 shrink-0 text-success" strokeWidth={3} aria-label="Selected" />
        ) : (
          <span className="size-3 shrink-0" />
        )}
        <span className="truncate" title={label}>
          {label}
        </span>
      </span>
      <span
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
        className="relative block h-1.5 w-full overflow-hidden rounded-full bg-muted"
      >
        <span
          className={cn(
            'absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]',
            highlighted ? 'bg-chart-1' : 'bg-chart-8/60',
          )}
          style={{ width: `${percent}%` }}
        />
      </span>
      <span className="text-right font-mono text-xs tabular text-muted-foreground">
        {probabilityLabel(value)}
      </span>
    </div>
  )
}

function ProbabilityList({
  values,
  selected = [],
  legend,
}: {
  values: [string, number][]
  selected?: string[]
  legend?: Record<string, string>
}) {
  if (!values.length)
    return <p className="text-xs text-subtle-foreground">No probabilities reported.</p>
  return (
    <div className="flex flex-col gap-2">
      {values.map(([key, value]) => (
        <Probability
          key={key}
          label={legend?.[key] || key}
          value={value}
          highlighted={selected.includes(key)}
        />
      ))}
    </div>
  )
}

function AnswerView({
  name,
  question,
  answer,
  run,
}: {
  name: string
  question: SystemOneQuestion
  answer?: SystemOneAnswer
  run: SystemOneRun
}) {
  const response = run.response
  if (answer?.error)
    return (
      <div
        role="alert"
        className="rounded-lg border border-destructive/25 bg-destructive-soft px-3 py-2.5 text-[13px]"
      >
        <p className="font-medium capitalize">{answer.error.replace(/_/g, ' ')}</p>
        <p className="text-muted-foreground">
          {answer.message || 'The model could not answer this question.'}
        </p>
      </div>
    )
  switch (question.type) {
    case 'choice':
      return (
        <div className="flex flex-col gap-2.5">
          <div className="flex items-baseline gap-2">
            <span className="text-[17px] font-semibold tracking-[-0.015em]">
              {answer?.choice ?? 'No choice reported'}
            </span>
            {finiteNumber(answer?.confidence) ? (
              <span className="text-xs text-subtle-foreground">
                {probabilityLabel(answer.confidence)} confidence
              </span>
            ) : null}
          </div>
          <ProbabilityList
            values={probabilities(answer?.probabilities).sort((a, b) => b[1] - a[1])}
            selected={answer?.choice ? [answer.choice] : []}
          />
          {finiteNumber(answer?.abstain_probability) ? (
            <p className="text-xs text-subtle-foreground">
              Abstain probability: {probabilityLabel(answer.abstain_probability)} · uncalibrated
            </p>
          ) : null}
        </div>
      )
    case 'score': {
      const values = probabilities(answer?.probabilities).sort(
        (a, b) => Number(a[0]) - Number(b[0]),
      )
      const levels = question.levels ?? []
      const max = Math.max(1, levels.length - 1)
      return (
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline gap-2">
            <span className="text-[22px] font-semibold leading-none tracking-[-0.02em] tabular">
              {finiteNumber(answer?.score) ? answer.score.toFixed(2) : '—'}
            </span>
            <span className="text-xs text-subtle-foreground">
              expected level on a 0–{max} scale
            </span>
          </div>
          {finiteNumber(answer?.score) ? (
            <div className="relative pt-1">
              <div className="flex h-2 gap-0.5 overflow-hidden rounded-full">
                {values.map(([key, value]) => (
                  <span
                    key={key}
                    className="flex-1 bg-chart-1"
                    style={{ opacity: 0.14 + value * 0.75 }}
                  />
                ))}
              </div>
              <span
                className="absolute top-0 h-4 w-1 -translate-x-1/2 rounded-full bg-foreground shadow-sm ring-2 ring-card transition-[left] duration-700"
                style={{ left: `${Math.min(100, Math.max(0, (answer.score / max) * 100))}%` }}
              />
            </div>
          ) : null}
          <div className="grid gap-1.5" aria-label="Score distribution">
            {values.map(([key, value]) => (
              <Probability
                key={key}
                label={`${key} · ${answer?.legend?.[key] ?? levels[Number(key)] ?? ''}`}
                value={value}
              />
            ))}
          </div>
        </div>
      )
    }
    case 'noul': {
      const value = answer?.noul
      if (!finiteNumber(value))
        return <p className="text-xs text-subtle-foreground">No probability reported.</p>
      return (
        <div className="flex flex-col gap-2">
          <Probability label="true" value={value} highlighted={value >= 0.5} />
          <Probability label="false" value={1 - value} highlighted={value < 0.5} />
          <span className="text-xs text-subtle-foreground">
            P(true) = <span className="font-mono tabular text-foreground">{value.toFixed(2)}</span>{' '}
            · a probability, not a yes/no verdict
          </span>
        </div>
      )
    }
    case 'set': {
      const set = response.sets?.[name]
      if (!set) return <p className="text-xs text-subtle-foreground">No set result reported.</p>
      return (
        <div className="flex flex-col gap-2">
          <ProbabilityList
            values={probabilities(set.probabilities).sort((a, b) => b[1] - a[1])}
            selected={set.selected}
          />
          <span className="text-xs text-subtle-foreground">
            {set.selected.length ? `${set.selected.join(', ')} selected` : 'No labels selected'} ·
            each label is scored independently
            {finiteNumber(response.thresholds?.[name])
              ? ` · threshold ${probabilityLabel(response.thresholds[name])}`
              : ''}
          </span>
        </div>
      )
    }
    case 'span': {
      const spans = response.spans?.[name]
      if (!spans) return <p className="text-xs text-subtle-foreground">No span result reported.</p>
      const source = spanSource(run.request.state, question)
      return (
        <div className="flex flex-col gap-2.5">
          <div className="flex items-baseline gap-2">
            <span className="text-[15px] font-semibold">
              {spans.length} {spans.length === 1 ? 'span' : 'spans'} found
            </span>
            <span className="text-xs text-subtle-foreground">
              {response.span_heads?.[name]
                ? `${response.span_heads[name]} head`
                : 'text extraction'}
            </span>
          </div>
          {source !== null ? (
            <p
              aria-label="Highlighted spans"
              className="whitespace-pre-wrap rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5 text-[12.5px] leading-relaxed text-muted-foreground"
            >
              {spanSegments(source, spans).map((segment, index) =>
                segment.labels.length ? (
                  <mark
                    key={index}
                    title={segment.labels.join(', ')}
                    className="rounded-[4px] bg-chart-2/25 px-0.5 font-medium text-foreground"
                  >
                    {segment.text}
                    <small className="ml-1 font-mono text-[10px] text-subtle-foreground">
                      {segment.labels.join(' / ')}
                    </small>
                  </mark>
                ) : (
                  <span key={index}>{segment.text}</span>
                ),
              )}
            </p>
          ) : null}
          {spans.length ? (
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-left text-xs">
                <thead className="bg-surface-subtle/80 text-subtle-foreground">
                  <tr>
                    <th className="px-3 py-1.5 font-medium">Text</th>
                    <th className="px-3 py-1.5 font-medium">Label</th>
                    <th className="px-3 py-1.5 font-medium">Range</th>
                    <th className="px-3 py-1.5 text-right font-medium">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {spans.map((span, index) => (
                    <tr
                      key={`${span.start}-${span.end}-${span.label}-${index}`}
                      className="border-t border-border"
                    >
                      <td className="px-3 py-1.5">{span.text}</td>
                      <td className="px-3 py-1.5">
                        <Badge tone="outline" size="sm">
                          {span.label}
                        </Badge>
                      </td>
                      <td className="px-3 py-1.5 font-mono tabular">
                        {span.start}–{span.end}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono tabular">
                        {probabilityLabel(span.probability)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          <p className="text-xs text-subtle-foreground">
            Offsets count Unicode code points; the end is exclusive.
            {source === null ? ' Use a single text field to see in-context highlights.' : ''}
            {finiteNumber(response.thresholds?.[name])
              ? ` Applied threshold ${probabilityLabel(response.thresholds[name])}.`
              : ''}
          </p>
        </div>
      )
    }
  }
}

export function SystemOneRoutingOutcome({ run }: { run: SystemOneRun }) {
  const routing = run.response.routing
  if (!routing) return null
  return (
    <section
      aria-label="Auto routing outcome"
      className="flex flex-col gap-3 rounded-lg border border-border bg-surface-subtle/60 p-3.5"
    >
      <div className="flex flex-wrap items-baseline gap-2">
        <SectionLabel>Selected model</SectionLabel>
        <span className="font-mono text-[13px] font-semibold">{routing.selected_model}</span>
      </div>
      <KeyValueGrid
        columns={2}
        items={[
          { label: 'Stage', value: routing.stage, mono: true },
          { label: 'Algorithm calls', value: routing.model_calls },
          { label: 'Algorithm', value: routing.algorithm, mono: true },
          {
            label: 'Acceptance',
            value:
              routing.quality === 'calibrated' ? 'Calibrated risk gate' : 'Operator thresholds',
          },
        ]}
      />
      <p className="text-xs text-muted-foreground">
        {routing.recipe} · {routing.decision}. Acceptance describes the configured gate; it is not
        measured accuracy. Algorithm calls include transport retries. Signal inference is counted
        separately.
      </p>
    </section>
  )
}

export default function SystemOneResults({ run }: { run: SystemOneRun }) {
  return (
    <div className="flex flex-col divide-y divide-border">
      {Object.entries(run.request.questions).map(([name, question]) => {
        const Icon = QUESTION_ICONS[question.type] ?? CircleDot
        const answer = run.response.answers[name]
        return (
          <article
            key={name}
            aria-label={`Result for ${name}`}
            className="flex flex-col gap-3 py-4 first:pt-1 last:pb-1"
          >
            <div className="flex min-w-0 items-center gap-2">
              <Icon className="size-3.5 shrink-0 text-chart-3" />
              <h3 className="font-mono text-[12.5px] font-semibold">{name}</h3>
              <span className="min-w-0 truncate text-xs text-subtle-foreground">
                {question.instructions}
              </span>
              {finiteNumber(answer?.confidence) ? (
                <span className="ml-auto shrink-0 rounded-md bg-secondary px-1.5 py-0.5 font-mono text-[11px] tabular text-muted-foreground">
                  conf {answer.confidence.toFixed(2)}
                </span>
              ) : null}
            </div>
            <AnswerView name={name} question={question} answer={answer} run={run} />
          </article>
        )
      })}
    </div>
  )
}
