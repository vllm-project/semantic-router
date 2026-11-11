import { recordedSignalMatches } from './insightsRecordSignals'
import type { InsightsRecord } from './insightsPageTypes'
import styles from './InsightsPage.module.css'

const SHOWN = 2

// The probability recorded for a match, such as `injection:jailbreak` under
// `systemone:injection:jailbreak`; undefined for a match with none (a keyword).
function probabilityOf(record: InsightsRecord, type: string, value: string) {
  const values = record.signal_values ?? {}
  const exact = values[`${type}:${value}`] ?? values[`systemone:${value}`]
  const found =
    exact ??
    Object.entries(values).find(
      ([key]) => key.endsWith(`:${value}`) && key.split(':').length === value.split(':').length + 1,
    )?.[1]
  return typeof found === 'number' && Number.isFinite(found)
    ? Math.min(1, Math.max(0, found))
    : undefined
}

// A row's signal matches, two at most: each name over a hairline bar filled to its
// recorded probability, with the rest counted and named in the title.
export default function InsightsSignalsCell({ record }: { record: InsightsRecord }) {
  // the matches with a recorded probability come first, since only they draw a bar
  const matches = recordedSignalMatches(record.signals)
    .map((match) => ({ ...match, probability: probabilityOf(record, match.type, match.value) }))
    .sort(
      (left, right) =>
        Number(right.probability !== undefined) - Number(left.probability !== undefined),
    )
  if (matches.length === 0) return <span>-</span>
  const all = matches.map((match) => match.text).join(', ')
  return (
    <span className={styles.signalCell} title={all}>
      {matches.slice(0, SHOWN).map((match) => {
        const [question, answer] = match.text.includes(': ')
          ? match.text.split(': ', 2)
          : ['', match.text]
        const { probability } = match
        return (
          <span key={`${match.type}:${match.value}`} className={styles.signalItem}>
            <span className={styles.signalText}>
              {question ? <span>{question} </span> : null}
              <strong>{answer}</strong>
            </span>
            {probability === undefined ? null : (
              <span
                className={styles.signalMeter}
                role="img"
                aria-label={`${Math.round(probability * 100)}%`}
              >
                <span style={{ width: `${probability * 100}%` }} />
              </span>
            )}
          </span>
        )
      })}
      {matches.length > SHOWN ? (
        <span className={styles.signalMore}>+{matches.length - SHOWN} more</span>
      ) : null}
    </span>
  )
}
