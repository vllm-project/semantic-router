import { useState, type CSSProperties } from 'react'
import {
  finiteNumber,
  probabilities,
  probabilityLabel,
  spanSegments,
  spanSource,
  type SystemOneAnswer,
  type SystemOneQuestion,
  type SystemOneResponse,
} from './systemOnePlayground'
import type { SystemOneRun } from './useSystemOnePlayground'
import styles from './SystemOnePlaygroundPage.module.css'

function ProbabilityBars({
  values,
  selected = [],
  legend,
}: {
  values: [string, number][]
  selected?: string[]
  legend?: Record<string, string>
}) {
  if (!values.length) return <p className={styles.muted}>No probabilities reported.</p>
  return (
    <div className={styles.probabilityList}>
      {values.map(([key, value]) => (
        <div
          key={key}
          className={`${styles.probabilityRow} ${selected.includes(key) ? styles.probabilitySelected : ''}`}
        >
          <div className={styles.probabilityLabel}>
            <span>
              {selected.includes(key) && <span aria-label="Selected">✓ </span>}
              {legend?.[key] || key}
            </span>
            <strong>{probabilityLabel(value)}</strong>
          </div>
          <div
            role="meter"
            aria-label={legend?.[key] || key}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={value * 100}
            className={styles.barTrack}
          >
            <span style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
          </div>
        </div>
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
      <div role="alert" className={styles.questionError}>
        <strong>{answer.error.replace(/_/g, ' ')}</strong>
        <p>{answer.message || 'The model could not answer this question.'}</p>
      </div>
    )
  switch (question.type) {
    case 'choice':
      return (
        <>
          <div className={styles.answerHeadline}>
            <strong>{answer?.choice ?? 'No choice reported'}</strong>
            {finiteNumber(answer?.confidence) && (
              <span>{probabilityLabel(answer.confidence)} confidence</span>
            )}
          </div>
          <ProbabilityBars
            values={probabilities(answer?.probabilities).sort((a, b) => b[1] - a[1])}
            selected={answer?.choice ? [answer.choice] : []}
          />
          {finiteNumber(answer?.abstain_probability) && (
            <p className={styles.caption}>
              Abstain probability: {probabilityLabel(answer.abstain_probability)} · uncalibrated
            </p>
          )}
        </>
      )
    case 'score': {
      const values = probabilities(answer?.probabilities).sort(
        (a, b) => Number(a[0]) - Number(b[0]),
      )
      const levels = question.levels ?? []
      return (
        <>
          <div className={styles.scoreHeadline}>
            <strong>{finiteNumber(answer?.score) ? answer.score.toFixed(2) : '—'}</strong>
            <span>
              expected level
              <br />
              on a 0–{Math.max(0, levels.length - 1)} scale
            </span>
          </div>
          <div className={styles.distribution} aria-label="Score distribution">
            {values.map(([key, value]) => (
              <div className={styles.distributionColumn} key={key}>
                <strong>{probabilityLabel(value)}</strong>
                <div className={styles.distributionTrack}>
                  <span style={{ height: `${Math.min(100, Math.max(0, value * 100))}%` }} />
                </div>
                <span>{key}</span>
              </div>
            ))}
          </div>
          <div className={styles.scoreLegend}>
            {levels.map((level, index) => (
              <span key={index}>
                <b>{index}</b>
                {answer?.legend?.[String(index)] ?? level}
              </span>
            ))}
          </div>
        </>
      )
    }
    case 'noul': {
      const value = answer?.noul
      if (!finiteNumber(value)) return <p className={styles.muted}>No probability reported.</p>
      return (
        <>
          <div className={styles.noulResult}>
            <div
              className={styles.probabilityRing}
              style={
                { '--probability': `${Math.max(0, Math.min(1, value)) * 360}deg` } as CSSProperties
              }
            >
              <div>
                <strong>{probabilityLabel(value)}</strong>
                <span>P(true)</span>
              </div>
            </div>
            <div>
              <strong>Probability of true</strong>
              <p>
                The model estimates how strongly the input supports your question. Noul returns a
                probability, not a yes/no verdict.
              </p>
            </div>
          </div>
          <ProbabilityBars
            values={[
              ['false', 1 - value],
              ['true', value],
            ]}
          />
        </>
      )
    }
    case 'set': {
      const set = response.sets?.[name]
      if (!set) return <p className={styles.muted}>No set result reported.</p>
      return (
        <>
          <div className={styles.selectedLabels}>
            {set.selected.length ? (
              set.selected.map((label) => <span key={label}>✓ {label}</span>)
            ) : (
              <span>No labels selected</span>
            )}
          </div>
          <ProbabilityBars
            values={probabilities(set.probabilities).sort((a, b) => b[1] - a[1])}
            selected={set.selected}
          />
          <p className={styles.caption}>
            Each label is scored independently
            {finiteNumber(response.thresholds?.[name])
              ? ` · applied threshold ${probabilityLabel(response.thresholds[name])}`
              : ''}
            .
          </p>
        </>
      )
    }
    case 'span': {
      const spans = response.spans?.[name]
      if (!spans) return <p className={styles.muted}>No span result reported.</p>
      const source = spanSource(run.request.state, question)
      return (
        <>
          <div className={styles.answerHeadline}>
            <strong>
              {spans.length} {spans.length === 1 ? 'span' : 'spans'} found
            </strong>
            <span>
              {response.span_heads?.[name]
                ? `${response.span_heads[name]} head`
                : 'Text extraction'}
            </span>
          </div>
          {source !== null && (
            <p className={styles.spanPreview} aria-label="Highlighted spans">
              {spanSegments(source, spans).map((segment, index) =>
                segment.labels.length ? (
                  <mark key={index} title={segment.labels.join(', ')}>
                    {segment.text}
                    <small>{segment.labels.join(' / ')}</small>
                  </mark>
                ) : (
                  <span key={index}>{segment.text}</span>
                ),
              )}
            </p>
          )}
          {spans.length > 0 && (
            <div className={styles.spanTableWrap}>
              <table className={styles.spanTable}>
                <thead>
                  <tr>
                    <th>Text</th>
                    <th>Label</th>
                    <th>Range</th>
                    <th>Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {spans.map((span, index) => (
                    <tr key={`${span.start}-${span.end}-${span.label}-${index}`}>
                      <td>{span.text}</td>
                      <td>
                        <span>{span.label}</span>
                      </td>
                      <td>
                        {span.start}–{span.end}
                      </td>
                      <td>{probabilityLabel(span.probability)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className={styles.caption}>
            Offsets count Unicode code points; the end is exclusive.
            {source === null ? ' Use a single text field to see in-context highlights.' : ''}
            {finiteNumber(response.thresholds?.[name])
              ? ` Applied threshold ${probabilityLabel(response.thresholds[name])}.`
              : ''}
          </p>
        </>
      )
    }
  }
}

export function JSONInspector({ title, value }: { title: string; value: unknown }) {
  const [copyState, setCopyState] = useState('Copy JSON')
  const serialized = JSON.stringify(value, null, 2)
  return (
    <details className={styles.inspector}>
      <summary>{title}</summary>
      <div className={styles.inspectToolbar}>
        <span>application/json</span>
        <button
          type="button"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(serialized)
              setCopyState('Copied')
            } catch {
              setCopyState('Copy unavailable')
            }
          }}
        >
          {copyState}
        </button>
      </div>
      <pre>
        <code>{serialized}</code>
      </pre>
    </details>
  )
}

function MetaChips({ response, elapsed }: { response: SystemOneResponse; elapsed: number }) {
  return (
    <dl className={styles.runMeta}>
      <div>
        <dt>Round trip</dt>
        <dd>{elapsed < 1000 ? `${Math.round(elapsed)} ms` : `${(elapsed / 1000).toFixed(2)} s`}</dd>
      </div>
      <div>
        <dt>Input tokens</dt>
        <dd>{response.usage.input_tokens.toLocaleString()}</dd>
      </div>
      {finiteNumber(response.meta?.compute_ms) && (
        <div>
          <dt>Compute</dt>
          <dd>{response.meta.compute_ms.toFixed(1)} ms</dd>
        </div>
      )}
      {response.meta?.profile && (
        <div>
          <dt>Profile</dt>
          <dd>{response.meta.profile}</dd>
        </div>
      )}
    </dl>
  )
}

export default function SystemOneResults({ run }: { run: SystemOneRun }) {
  return (
    <div className={styles.results}>
      <div className={styles.resultSummary}>
        <div>
          <span className={styles.eyebrow}>Completed {run.completedAt.toLocaleTimeString()}</span>
          <h3>{run.response.model}</h3>
        </div>
        <span className={styles.successPill}>Response received</span>
      </div>
      <MetaChips response={run.response} elapsed={run.elapsed} />
      {Object.entries(run.request.questions).map(([name, question]) => (
        <article className={styles.resultCard} key={name} aria-label={`Result for ${name}`}>
          <div className={styles.resultCardHeader}>
            <h3>{name}</h3>
            <span className={styles.typeBadge}>{question.type}</span>
          </div>
          <p className={styles.resultQuestion}>{question.instructions}</p>
          <AnswerView
            name={name}
            question={question}
            answer={run.response.answers[name]}
            run={run}
          />
        </article>
      ))}
      <JSONInspector title="Inspect response" value={run.response} />
      <JSONInspector
        title="Inspect submitted request"
        value={{ deployment: run.deployment, request: run.request }}
      />
    </div>
  )
}
