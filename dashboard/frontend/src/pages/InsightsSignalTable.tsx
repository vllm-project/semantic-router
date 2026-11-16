import type { InsightsRecord } from './insightsPageTypes'
import { projectionMetric } from './insightsRoutingMetrics'
import styles from './InsightsRoutingEvidence.module.css'

// The recorded value, confidence and error match of each named signal.
export default function InsightsSignalTable({
  record,
  names,
}: {
  record: InsightsRecord
  names: string[]
}) {
  const errors = record.signal_error_matches
  const hasErrors = Boolean(errors && Object.keys(errors).length)
  return (
    <div className={styles.tableScroll}>
      <table aria-label="Recorded signal values and confidence">
        <thead>
          <tr>
            <th>Signal</th>
            <th>Value</th>
            <th>Confidence</th>
            {hasErrors ? <th>Error match</th> : null}
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td>{projectionMetric(record.signal_values?.[name])}</td>
              <td>{projectionMetric(record.signal_confidences?.[name])}</td>
              {hasErrors ? (
                <td>
                  {errors?.[name] === undefined ? 'Not recorded' : errors[name] ? 'Yes' : 'No'}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
