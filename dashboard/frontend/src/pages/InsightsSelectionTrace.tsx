import { ChevronDown } from 'lucide-react'

import type {
  SelectionCandidateRef,
  SelectionCandidateValue,
  SelectionObjectiveStage,
  SelectionTrace,
} from '../types/selectionTrace'
import {
  CARD,
  CHEVRON,
  DETAIL,
  DETAIL_BODY,
  EVIDENCE,
  INTRO,
  METRICS,
  MUTED,
  PILL,
  TABLE,
  TABLE_WRAP,
} from './insightsEvidenceStyles'
import { ModelName } from './insightsModelLabelSupport'

const factors: Record<string, string> = {
  quality: 'Quality',
  latency: 'Latency',
  cost: 'Cost',
  load: 'Load',
}

const metrics: Record<string, string> = {
  ttft: 'First response observation (TTFT)',
  tpot: 'Response duration / output token (TPOT)',
  tpot_then_ttft: 'TPOT, otherwise TTFT',
}

const reasons: Record<SelectionObjectiveStage['reason'], string> = {
  tolerance_band: 'Kept candidates within the configured tolerance of the best available value.',
  no_available_values: 'No values were available. No candidates were removed.',
  incomplete_latency_coverage:
    'Latency measurements do not cover every candidate. No candidates were removed.',
}

function number(value: number) {
  return new Intl.NumberFormat('en-US', { maximumSignificantDigits: 6 }).format(value)
}

function candidateDetails(candidate: SelectionCandidateRef) {
  return [
    candidate.LoRAName && `Adapter: ${candidate.LoRAName}`,
    candidate.UseReasoning === true
      ? 'Reasoning on'
      : candidate.UseReasoning === false
        ? 'Reasoning off'
        : '',
    candidate.ReasoningMode && `Mode: ${candidate.ReasoningMode}`,
    candidate.ReasoningEffort && `Effort: ${candidate.ReasoningEffort}`,
  ]
    .filter(Boolean)
    .join(' · ')
}

function metricLabel(stage: SelectionObjectiveStage, row?: SelectionCandidateValue) {
  const metric = row ? row.metric : stage.metric
  if (metric) return metrics[metric] ?? metric
  switch (stage.factor) {
    case 'quality':
      return 'Quality score'
    case 'cost':
      return 'Request cost forecast (USD)'
    case 'load':
      return 'Active requests'
    default:
      return 'Not recorded'
  }
}

function candidateValue(row: SelectionCandidateValue, factor: string) {
  if (typeof row.value !== 'number' || !Number.isFinite(row.value)) return 'Unknown'
  const value = number(row.value)
  if (factor === 'latency') return `${value} ${row.metric === 'tpot' ? 's / token' : 's'}`
  if (factor === 'cost') return `$${value}`
  return value
}

function outcome(row: SelectionCandidateValue, stage: SelectionObjectiveStage) {
  if (stage.action === 'skipped') return 'Not filtered'
  if (row.elimination_reason === 'missing_measurement') return 'Removed · missing measurement'
  if (row.elimination_reason === 'outside_tolerance') return 'Removed · outside tolerance'
  return 'Retained at this stage'
}

export default function InsightsSelectionTrace({
  trace,
  selectedModel,
}: {
  trace: SelectionTrace
  selectedModel?: string
}) {
  return (
    <div className={EVIDENCE}>
      <p className={INTRO}>
        These priority stages follow eligibility checks. Learning and session protection may choose
        among the remaining candidates. Only stages reached are shown.
      </p>
      <dl className={METRICS}>
        <div>
          <dt>Actual selected model</dt>
          <dd>
            <ModelName name={selectedModel} fallback="No backend selected" />
          </dd>
        </div>
        <div>
          <dt>Candidates retained by the objective</dt>
          <dd>
            {trace.final_survivors.length ? (
              <ul className="flex flex-col gap-1 [&_small]:ml-1.5 [&_small]:text-[11px] [&_small]:font-normal [&_small]:text-subtle-foreground">
                {trace.final_survivors.map((candidate, index) => (
                  <li key={index}>
                    <ModelName name={candidate.Model} />
                    {candidateDetails(candidate) ? (
                      <small>{candidateDetails(candidate)}</small>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : (
              'None recorded'
            )}
          </dd>
        </div>
      </dl>
      {trace.stages.length === 0 ? (
        <p className={MUTED}>No objective stages were recorded.</p>
      ) : null}
      <ol className="flex flex-col gap-2.5">
        {trace.stages.map((stage, index) => {
          const title = factors[stage.factor] ?? stage.factor
          return (
            <li className={CARD} key={index}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-[13px] font-semibold">
                  {index + 1}. {title}
                </h3>
                <div className="flex items-center gap-2">
                  <span className={PILL}>{stage.action === 'skipped' ? 'Skipped' : 'Applied'}</span>
                  <span className={MUTED}>
                    {stage.available} / {stage.total}{' '}
                    {stage.factor === 'latency' ? 'measured' : 'values available'}
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">{reasons[stage.reason]}</p>
              <details className={DETAIL}>
                <summary>
                  <span>Inspect {title.toLowerCase()} candidates</span>
                  <ChevronDown className={CHEVRON} />
                </summary>
                <div className={DETAIL_BODY}>
                  <dl className={METRICS}>
                    <div>
                      <dt>Metric</dt>
                      <dd>{metricLabel(stage)}</dd>
                    </div>
                    <div>
                      <dt>Relative tolerance</dt>
                      <dd>{number(stage.tolerance * 100)}%</dd>
                    </div>
                    {stage.percentile !== undefined ? (
                      <div>
                        <dt>Percentile</dt>
                        <dd>P{stage.percentile}</dd>
                      </div>
                    ) : null}
                  </dl>
                  {stage.factor === 'cost' ? (
                    <p className={MUTED}>
                      Configured-rate forecast, not a provider invoice or output-token limit.
                    </p>
                  ) : null}
                  <div className={TABLE_WRAP}>
                    <table className={TABLE} aria-label={`${title} candidate evidence`}>
                      <thead>
                        <tr>
                          <th scope="col">Candidate</th>
                          <th scope="col">Value</th>
                          <th scope="col">Metric</th>
                          <th scope="col">Stage outcome</th>
                        </tr>
                      </thead>
                      <tbody>
                        {stage.candidates.map((row, candidateIndex) => (
                          <tr key={candidateIndex}>
                            <th scope="row">
                              <ModelName name={row.candidate.Model} compact />
                              {candidateDetails(row.candidate) ? (
                                <small>{candidateDetails(row.candidate)}</small>
                              ) : null}
                            </th>
                            <td>{candidateValue(row, stage.factor)}</td>
                            <td>{metricLabel(stage, row)}</td>
                            <td>{outcome(row, stage)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </details>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
