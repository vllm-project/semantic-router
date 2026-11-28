import type { InsightsRecord } from './insightsPageTypes'
import { type SystemOneAnswer, systemOneAnswers } from './insightsSystemOneAnswers'
import { PILL } from './insightsEvidenceStyles'

const words = (name: string) => name.replace(/[_-]+/g, ' ')
const percent = (probability: number) => `${Math.round(probability * 100)}%`

// The bar under one answer: a choice's options side by side at their probabilities, or
// one probability's share; `lit` is the part that is the answer.
function Bar({ parts, lit, label }: { parts: number[]; lit: number; label: string }) {
  return (
    <span
      className="flex h-1.5 w-full gap-px overflow-hidden rounded-full bg-muted"
      role="img"
      aria-label={label}
      title={label}
    >
      {parts.map((part, index) => (
        <span
          key={index}
          className={index === lit ? 'bg-chart-2' : 'bg-chart-8/35'}
          style={{ flex: `0 0 ${part * 100}%` }}
        />
      ))}
    </span>
  )
}

function Row({ answer }: { answer: SystemOneAnswer }) {
  let bar = null
  let value
  if (answer.kind === 'choice' || answer.kind === 'labels') {
    const top = answer.options.reduce((best, one) =>
      one.probability > best.probability ? one : best,
    )
    const spread = answer.options.map((one) => `${one.key} ${percent(one.probability)}`).join(' · ')
    bar =
      answer.kind === 'choice' ? (
        <Bar
          parts={answer.options.map((one) => one.probability)}
          lit={answer.options.indexOf(top)}
          label={spread}
        />
      ) : (
        <Bar parts={[top.probability]} lit={0} label={spread} />
      )
    value = (
      <>
        <strong>{top.key}</strong> {percent(top.probability)}
      </>
    )
  } else if (answer.kind === 'probability') {
    bar = <Bar parts={[answer.value]} lit={0} label={percent(answer.value)} />
    value = <strong>{answer.value.toFixed(2)}</strong>
  } else {
    value = (
      <>
        level <strong>{answer.value.toFixed(2)}</strong>
      </>
    )
  }
  return (
    <li className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_auto] items-center gap-3 py-1 text-[13px]">
      <span className="truncate capitalize text-muted-foreground">{words(answer.question)}</span>
      {bar ?? <span />}
      <span className="font-mono text-xs tabular [&_strong]:font-semibold [&_strong]:text-foreground">
        {value}
      </span>
    </li>
  )
}

// The record's System One answers, one quiet row each: the answer lit on a thin bar,
// with every option's share in the bar's label.
export default function InsightsSystemOneAnswers({ record }: { record: InsightsRecord }) {
  const answers = systemOneAnswers(record.signal_values, record.signal_confidences)
  if (answers.length === 0) return null
  return (
    <section className="flex flex-col gap-2" aria-label="System One answers">
      <h3 className="flex items-center gap-2 text-[13px] font-semibold">
        System One answers <span className={PILL}>{answers.length}</span>
      </h3>
      <ul className="flex flex-col">
        {answers.map((answer) => (
          <Row key={answer.question} answer={answer} />
        ))}
      </ul>
    </section>
  )
}
