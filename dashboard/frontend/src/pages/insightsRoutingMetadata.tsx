import type { ViewField } from '../components/ViewPanel'
import type { InsightsRecord } from './insightsPageTypes'
import { projectionMetric } from './insightsRoutingMetrics'
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
          {signals.length > 0 ? (
            <section className={styles.metadataGroup} aria-label="Signal evidence">
              <h3>
                Signal evidence <span className={styles.count}>{signals.length}</span>
              </h3>
              <p className={styles.muted}>
                Values and confidence are distinct measurements; a missing confidence is not zero.
              </p>
              <div className={styles.tableScroll}>
                <table aria-label="Recorded signal values and confidence">
                  <thead>
                    <tr>
                      <th>Signal</th>
                      <th>Value</th>
                      <th>Confidence</th>
                      {record.signal_error_matches &&
                      Object.keys(record.signal_error_matches).length ? (
                        <th>Error match</th>
                      ) : null}
                    </tr>
                  </thead>
                  <tbody>
                    {signals.map((name) => (
                      <tr key={name}>
                        <th scope="row">{name}</th>
                        <td>{projectionMetric(record.signal_values?.[name])}</td>
                        <td>{projectionMetric(record.signal_confidences?.[name])}</td>
                        {record.signal_error_matches &&
                        Object.keys(record.signal_error_matches).length ? (
                          <td>
                            {record.signal_error_matches[name] === undefined
                              ? 'Not recorded'
                              : record.signal_error_matches[name]
                                ? 'Yes'
                                : 'No'}
                          </td>
                        ) : null}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </div>
      ),
    },
  ]
}
