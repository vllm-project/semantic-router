import { ChevronDown } from 'lucide-react'

import type {
  SelectionCandidateEstimate,
  SelectionCandidateEstimates,
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

const estimateStatuses: Record<SelectionCandidateEstimate['status'], string> = {
  available: 'Estimated',
  unavailable: 'Not estimated',
  deadline_exceeded: 'Deadline exceeded',
}

function estimateCost(estimate: SelectionCandidateEstimate) {
  if (typeof estimate.estimated_cost !== 'number') return 'Not priced'
  const value = number(estimate.estimated_cost)
  if (estimate.currency === 'USD') return `$${value}`
  return estimate.currency ? `${value} ${estimate.currency}` : value
}

function estimateStatus(estimate: SelectionCandidateEstimate) {
  const status = estimateStatuses[estimate.status] ?? estimate.status
  return estimate.reason ? `${status} · ${estimate.reason}` : status
}

/** Candidate estimates in observe mode: recorded beside the selection. */
function CandidateEstimates({ estimates }: { estimates: SelectionCandidateEstimates }) {
  const deployments = Array.from(
    new Set(estimates.candidates.map((estimate) => estimate.deployment).filter(Boolean)),
  )
  const uncalibrated = estimates.candidates.some(
    (estimate) => estimate.status === 'available' && !estimate.calibrated,
  )
  return (
    <section className={CARD} aria-label="Candidate estimates">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[13px] font-semibold">Candidate estimates</h3>
        <span className={PILL}>Observe only</span>
      </div>
      <p className="text-xs text-muted-foreground">
        The decision model&apos;s estimate for each eligible candidate. They are recorded only and
        did not change the selection.
        {uncalibrated ? ' Success probabilities are not calibrated.' : ''}
      </p>
      {deployments.length ? <p className={MUTED}>Estimated by {deployments.join(', ')}</p> : null}
      <div className={TABLE_WRAP}>
        <table className={TABLE} aria-label="Candidate estimates by candidate">
          <thead>
            <tr>
              <th scope="col">Candidate</th>
              <th scope="col">Success probability</th>
              <th scope="col">Expected output tokens</th>
              <th scope="col">Estimated cost</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {estimates.candidates.map((estimate, index) => (
              <tr key={index}>
                <th scope="row">
                  <ModelName name={estimate.candidate.Model} compact />
                  {candidateDetails(estimate.candidate) ? (
                    <small>{candidateDetails(estimate.candidate)}</small>
                  ) : null}
                </th>
                <td>
                  {typeof estimate.success_probability === 'number'
                    ? number(estimate.success_probability)
                    : 'Unknown'}
                </td>
                <td>
                  {typeof estimate.expected_output_tokens === 'number'
                    ? number(estimate.expected_output_tokens)
                    : 'Unknown'}
                </td>
                <td>{estimate.status === 'available' ? estimateCost(estimate) : 'Unknown'}</td>
                <td>{estimateStatus(estimate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default function InsightsSelectionTrace({
  trace,
  selectedModel,
}: {
  trace: SelectionTrace
  selectedModel?: string
}) {
  const stages = trace.stages ?? []
  const survivors = trace.final_survivors ?? []
  // A weighted objective records no stages or survivors, only estimates.
  const objective = trace.stages !== undefined || trace.final_survivors !== undefined
  return (
    <div className={EVIDENCE}>
      {objective ? (
        <p className={INTRO}>
          These priority stages follow eligibility checks. Learning and session protection may
          choose among the remaining candidates. Only stages reached are shown.
        </p>
      ) : null}
      <dl className={METRICS}>
        <div>
          <dt>Actual selected model</dt>
          <dd>
            <ModelName name={selectedModel} fallback="No backend selected" />
          </dd>
        </div>
        {objective ? (
          <div>
            <dt>Candidates retained by the objective</dt>
            <dd>
              {survivors.length ? (
                <ul className="flex flex-col gap-1 [&_small]:ml-1.5 [&_small]:text-[11px] [&_small]:font-normal [&_small]:text-subtle-foreground">
                  {survivors.map((candidate, index) => (
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
        ) : null}
      </dl>
      {objective && stages.length === 0 ? (
        <p className={MUTED}>No objective stages were recorded.</p>
      ) : null}
      {trace.estimates ? <CandidateEstimates estimates={trace.estimates} /> : null}
      <ol className="flex flex-col gap-2.5">
        {stages.map((stage, index) => {
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
