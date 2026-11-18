import { recordedSignalMatches } from './insightsRecordSignals'
import type { InsightsRecord } from './insightsPageTypes'

const SHOWN = 2

// The confidence recorded for a match, such as `injection:jailbreak` under
// `systemone:injection:jailbreak`; undefined for a match with none. Only a confidence
// is a probability: a signal's value can be a count (a keyword's), a feature (a
// conversation's) or a level (a System One score).
function probabilityOf(record: InsightsRecord, type: string, value: string) {
  const confidences = record.signal_confidences ?? {}
  const found = confidences[`${type}:${value}`] ?? confidences[`systemone:${value}`]
  return typeof found === 'number' && Number.isFinite(found) && found >= 0 && found <= 1
    ? found
    : undefined
}

// A row's signal matches, two at most: each name over a hairline bar filled to its
// recorded probability, with the rest counted and named in the title.
export default function InsightsSignalsCell({ record }: { record: InsightsRecord }) {
  // the matches with a recorded probability come first, since only they draw a bar, and
  // each group in name order, so every row lists the same signals in the same place
  const matches = recordedSignalMatches(record.signals)
    .map((match) => ({ ...match, probability: probabilityOf(record, match.type, match.value) }))
    .sort(
      (left, right) =>
        Number(right.probability !== undefined) - Number(left.probability !== undefined) ||
        left.text.localeCompare(right.text),
    )
  if (matches.length === 0) return <span>-</span>
  const all = matches.map((match) => match.text).join(', ')
  return (
    <span className="flex min-w-0 flex-col gap-1" title={all}>
      {matches.slice(0, SHOWN).map((match) => {
        const [question, answer] = match.text.includes(': ')
          ? match.text.split(': ', 2)
          : ['', match.text]
        const { probability } = match
        return (
          <span key={`${match.type}:${match.value}`} className="flex min-w-0 flex-col gap-0.5">
            <span className="truncate text-xs text-muted-foreground [&_strong]:font-medium [&_strong]:text-foreground">
              {question ? <span>{question} </span> : null}
              <strong>{answer}</strong>
            </span>
            {probability === undefined ? null : (
              <span
                className="block h-[3px] w-full overflow-hidden rounded-full bg-muted"
                role="img"
                aria-label={`${Math.round(probability * 100)}%`}
              >
                <span
                  className="block h-full rounded-full bg-chart-2"
                  style={{ width: `${probability * 100}%` }}
                />
              </span>
            )}
          </span>
        )
      })}
      {matches.length > SHOWN ? (
        <span className="text-[11px] text-subtle-foreground">+{matches.length - SHOWN} more</span>
      ) : null}
    </span>
  )
}
