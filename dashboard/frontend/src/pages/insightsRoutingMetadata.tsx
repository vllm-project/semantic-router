import type { ViewField } from '../components/ViewPanel'
import type { InsightsRecord } from './insightsPageTypes'
import { projectionMetric } from './insightsRoutingMetrics'
import InsightsSignalTable from './InsightsSignalTable'
import InsightsSystemOneAnswers from './InsightsSystemOneAnswers'
import { systemOneAnswers } from './insightsSystemOneAnswers'
import styles from './InsightsRoutingEvidence.module.css'

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
  const shown = signals.filter((name) => !isAnswerValue(name))
  const folded = signals.filter(isAnswerValue)
  return [
    {
      label: 'Recorded routing values',
      fullWidth: true,
      value: (
        <div className={styles.metadata}>
          <p className={styles.intro}>
            Values captured for this request. Missing values remain unrecorded.
          </p>
          {outputs.length > 0 ? (
            <section className={styles.metadataGroup} aria-label="Projection outputs">
              <h3>
                Projection outputs <span className={styles.count}>{outputs.length}</span>
              </h3>
              <ul className={styles.outputs}>
                {outputs.map((output, index) => (
                  <li key={`${output}-${index}`}>{output}</li>
                ))}
              </ul>
            </section>
          ) : null}
          {scores.length > 0 ? (
            <section className={styles.metadataGroup} aria-label="Projection scores">
              <h3>
                Projection scores <span className={styles.count}>{scores.length}</span>
              </h3>
              <dl className={styles.valueList}>
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
            <section className={styles.metadataGroup} aria-label="Signal evidence">
              <h3>
                Signal evidence <span className={styles.count}>{signals.length}</span>
              </h3>
              <p className={styles.muted}>
                Values and confidence are distinct measurements; a missing confidence is not zero.
              </p>
              {shown.length > 0 ? <InsightsSignalTable record={record} names={shown} /> : null}
              {folded.length > 0 ? (
                <details className={styles.allValues}>
                  <summary>Raw System One values ({folded.length})</summary>
                  <InsightsSignalTable record={record} names={folded} />
                </details>
              ) : null}
            </section>
          ) : null}
        </div>
      ),
    },
  ]
}
