import type { ViewField } from '../components/ViewPanel'
import type { InsightsRecord } from './insightsPageTypes'
import { projectionMetric } from './insightsRoutingMetrics'
import InsightsSignalTable from './InsightsSignalTable'
import InsightsSystemOneAnswers from './InsightsSystemOneAnswers'
import { systemOneAnswers } from './insightsSystemOneAnswers'
import { DETAIL, DETAIL_BODY, INTRO, MUTED, PILL } from './insightsEvidenceStyles'

export function buildRoutingMetadataFields(record: InsightsRecord): ViewField[] {
  const outputs = record.projections ?? []
  const scores = Object.entries(record.projection_scores ?? {}).sort(([left], [right]) =>
    left.localeCompare(right),
  )
  const signals = [
    ...new Set([
      ...Object.keys(record.signal_values ?? {}),
      ...Object.keys(record.signal_confidences ?? {}),
      ...Object.keys(record.signal_error_matches ?? {}),
    ]),
  ].sort()
  if (!outputs.length && !scores.length && !signals.length) return []
  // with System One's answers drawn above, only their raw values fold away
  const answered = systemOneAnswers(record.signal_values, record.signal_confidences).length > 0
  const isAnswerValue = (name: string) => answered && name.startsWith('systemone:')
  // a signal measured at zero (a PII type that did not occur) folds away too, so the
  // rows that open show what fired
  const isZero = (name: string) => {
    const measured = [record.signal_values?.[name], record.signal_confidences?.[name]]
    return (
      measured.some((value) => value === 0) &&
      measured.every((value) => value === undefined || value === 0) &&
      !record.signal_error_matches?.[name]
    )
  }
  const shown = signals.filter((name) => !isAnswerValue(name) && !isZero(name))
  const zeros = signals.filter((name) => !isAnswerValue(name) && isZero(name))
  const folded = signals.filter(isAnswerValue)
  return [
    {
      label: 'Recorded routing values',
      fullWidth: true,
      value: (
        <div className="flex min-w-0 flex-col gap-4">
          <p className={INTRO}>
            Values captured for this request. Missing values remain unrecorded.
          </p>
          {outputs.length > 0 ? (
            <section
              className="flex min-w-0 flex-col gap-2 [&>h3]:flex [&>h3]:items-center [&>h3]:gap-2 [&>h3]:text-[13px] [&>h3]:font-semibold"
              aria-label="Projection outputs"
            >
              <h3>
                Projection outputs <span className={PILL}>{outputs.length}</span>
              </h3>
              <ul className="flex flex-wrap gap-1.5 [&>li]:inline-flex [&>li]:h-6 [&>li]:items-center [&>li]:rounded-full [&>li]:bg-chart-4/14 [&>li]:px-2.5 [&>li]:text-xs [&>li]:font-medium [&>li]:text-chart-4">
                {outputs.map((output, index) => (
                  <li key={`${output}-${index}`}>{output}</li>
                ))}
              </ul>
            </section>
          ) : null}
          {scores.length > 0 ? (
            <section
              className="flex min-w-0 flex-col gap-2 [&>h3]:flex [&>h3]:items-center [&>h3]:gap-2 [&>h3]:text-[13px] [&>h3]:font-semibold"
              aria-label="Projection scores"
            >
              <h3>
                Projection scores <span className={PILL}>{scores.length}</span>
              </h3>
              <dl className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2 [&>div]:flex [&>div]:items-baseline [&>div]:justify-between [&>div]:gap-3 [&>div]:border-b [&>div]:border-border [&>div]:py-1 [&_dt]:truncate [&_dt]:font-mono [&_dt]:text-xs [&_dt]:text-muted-foreground [&_dd]:font-mono [&_dd]:text-xs [&_dd]:tabular">
                {scores.map(([name, value]) => (
                  <div key={name}>
                    <dt>{name}</dt>
                    <dd>{projectionMetric(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ) : null}
          <InsightsSystemOneAnswers record={record} />
          {signals.length > 0 ? (
            <section
              className="flex min-w-0 flex-col gap-2 [&>h3]:flex [&>h3]:items-center [&>h3]:gap-2 [&>h3]:text-[13px] [&>h3]:font-semibold"
              aria-label="Signal evidence"
            >
              <h3>
                Signal evidence <span className={PILL}>{signals.length}</span>
              </h3>
              <p className={MUTED}>
                Values and confidence are distinct measurements; a missing confidence is not zero.
              </p>
              {shown.length > 0 ? <InsightsSignalTable record={record} names={shown} /> : null}
              {zeros.length > 0 ? (
                <details className={DETAIL}>
                  <summary>At zero ({zeros.length})</summary>
                  <div className={DETAIL_BODY}>
                    <InsightsSignalTable record={record} names={zeros} />
                  </div>
                </details>
              ) : null}
              {folded.length > 0 ? (
                <details className={DETAIL}>
                  <summary>Raw System One values ({folded.length})</summary>
                  <div className={DETAIL_BODY}>
                    <InsightsSignalTable record={record} names={folded} />
                  </div>
                </details>
              ) : null}
            </section>
          ) : null}
        </div>
      ),
    },
  ]
}
