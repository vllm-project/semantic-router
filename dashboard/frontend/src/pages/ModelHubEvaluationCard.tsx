import React from 'react'

import type { CatalogBenchmark, CatalogEvaluation } from '../types/modelCatalog'
import {
  modelHubBenchmarkRawValueLabel,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'
import {
  modelHubEvaluationConditionLabel,
  readableModelHubValue as readable,
  type ModelHubRow,
} from './modelHubSupport'
import styles from './ModelHubDetail.module.css'

const scoreValue = (value: number, benchmark: CatalogBenchmark | undefined, metric: string) => {
  const definition = benchmark?.metrics.find((candidate) => candidate.id === metric)
  if (!definition) {
    return {
      label: Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2),
      raw: undefined,
    }
  }
  return {
    label: modelHubBenchmarkValueLabel(value, definition),
    raw: modelHubBenchmarkRawValueLabel(value, definition),
  }
}

export const EvaluationCard: React.FC<{
  model: ModelHubRow['model']
  evaluation: CatalogEvaluation
  benchmark: CatalogBenchmark | undefined
}> = ({ model, evaluation, benchmark }) => (
  <article className={styles.evaluationCard}>
    <div>
      <span>{benchmark?.display_name ?? evaluation.benchmark}</span>
      <small>
        {readable(evaluation.benchmark_profile)} ·{' '}
        {modelHubEvaluationConditionLabel(model, evaluation.reasoning_effort)}
      </small>
    </div>
    <dl>
      {Object.entries(evaluation.metrics).map(([metric, value]) => {
        const score = scoreValue(value, benchmark, metric)
        return (
          <div key={metric}>
            <dt>{readable(metric)}</dt>
            <dd title={score.raw ? `Raw: ${score.raw}` : undefined}>{score.label}</dd>
          </div>
        )
      })}
    </dl>
    <footer>
      <span>
        {readable(evaluation.evidence.provenance)} · {evaluation.evidence.verification}
        {evaluation.measured_at
          ? ` · measured ${evaluation.measured_at}`
          : evaluation.observed_at
            ? ` · observed ${evaluation.observed_at}`
            : ''}
      </span>
      {evaluation.evidence.source ? (
        <a href={evaluation.evidence.source} target="_blank" rel="noreferrer">
          Source ↗
        </a>
      ) : null}
    </footer>
  </article>
)
