import type { InsightsRecord } from './insightsPageTypes'
import { type SystemOneAnswer, systemOneAnswers } from './insightsSystemOneAnswers'
import styles from './InsightsRoutingEvidence.module.css'

const words = (name: string) => name.replace(/[_-]+/g, ' ')
const percent = (probability: number) => `${Math.round(probability * 100)}%`

function ChoiceRow({ answer }: { answer: Extract<SystemOneAnswer, { kind: 'choice' }> }) {
  const top = answer.options.reduce((best, one) =>
    one.probability > best.probability ? one : best,
  )
  const spread = answer.options.map((one) => `${one.key} ${percent(one.probability)}`).join(' · ')
  return (
    <li className={styles.answer} title={spread}>
      <span className={styles.answerQuestion}>{words(answer.question)}</span>
      <span className={styles.answerBar} aria-hidden="true">
        {answer.options.map((one) => (
          <span
            key={one.key}
            className={one === top ? styles.answerTop : undefined}
            style={{ flexGrow: Math.max(one.probability, 0.004) }}
          />
        ))}
      </span>
      <span className={styles.answerValue}>
        <strong>{top.key}</strong> {percent(top.probability)}
      </span>
    </li>
  )
}

function ValueRow({ answer }: { answer: Extract<SystemOneAnswer, { kind: 'value' }> }) {
  // a probability fills its meter; a score above 1 shows as its number alone
  const isProbability = answer.value >= 0 && answer.value <= 1
  return (
    <li className={styles.answer}>
      <span className={styles.answerQuestion}>{words(answer.question)}</span>
      <span className={styles.answerBar} aria-hidden="true">
        {isProbability ? (
          <>
            <span
              className={styles.answerTop}
              style={{ flexGrow: Math.max(answer.value, 0.004) }}
            />
            <span style={{ flexGrow: 1 - answer.value }} />
          </>
        ) : null}
      </span>
      <span className={styles.answerValue}>
        <strong>{answer.value.toFixed(2)}</strong>
      </span>
    </li>
  )
}

// The record's System One answers, one quiet row each: a thin bar of a choice's
// options with the answer lit, or a meter for a single probability.
export default function InsightsSystemOneAnswers({ record }: { record: InsightsRecord }) {
  const answers = systemOneAnswers(record.signal_values)
  if (answers.length === 0) return null
  return (
    <section className={styles.metadataGroup} aria-label="System One answers">
      <h3>
        System One answers <span className={styles.count}>{answers.length}</span>
      </h3>
      <ul className={styles.answers}>
        {answers.map((answer) =>
          answer.kind === 'choice' ? (
            <ChoiceRow key={answer.question} answer={answer} />
          ) : (
            <ValueRow key={answer.question} answer={answer} />
          ),
        )}
      </ul>
    </section>
  )
}
