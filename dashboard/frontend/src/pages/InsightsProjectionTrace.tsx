import type { ReactNode } from 'react'
import { ArrowRight, ChevronDown } from 'lucide-react'

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
  RAW,
  TABLE,
  TABLE_WRAP,
} from './insightsEvidenceStyles'
import type { ProjectionTrace } from './insightsPageTypes'
import { projectionMetric } from './insightsRoutingMetrics'

const STAGE_NUMBER =
  'grid size-6 shrink-0 place-items-center rounded-full bg-chart-4/14 font-mono text-[11px] font-semibold text-chart-4'
const RESULT_META =
  'flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground [&_strong]:font-mono [&_strong]:font-medium [&_strong]:text-foreground'

function EvidenceDetail({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className={DETAIL}>
      <summary>
        <span>{title}</span>
        <ChevronDown className={CHEVRON} />
      </summary>
      <div className={DETAIL_BODY}>{children}</div>
    </details>
  )
}

export default function InsightsProjectionTrace({
  trace,
  recordID,
}: {
  trace: ProjectionTrace
  recordID: string
}) {
  const partitions = trace.partitions ?? []
  const scores = trace.scores ?? []
  const mappings = trace.mappings ?? []
  const scoreID = (index: number) => `projection-${encodeURIComponent(recordID)}-score-${index}`
  return (
    <div className={EVIDENCE}>
      <p className={INTRO}>
        Follow the recorded inputs to their routing outputs. Expand a result to inspect its
        calculation.
      </p>
      {!partitions.length && !scores.length && !mappings.length ? (
        <p className={MUTED}>No projection stages were recorded.</p>
      ) : null}
      <div className="flex flex-col gap-4">
        {partitions.length > 0 ? (
          <section className="flex flex-col gap-2.5" aria-label="Signal groups">
            <div className="flex items-center gap-2.5 [&_h3]:text-[13px] [&_h3]:font-semibold [&_p]:text-xs [&_p]:text-subtle-foreground">
              <span className={STAGE_NUMBER}>1</span>
              <div>
                <h3>Signal groups</h3>
                <p>Which signal won within each group</p>
              </div>
              <span className={`${PILL} ml-auto`}>{partitions.length}</span>
            </div>
            <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-2">
              {partitions.map((partition, index) => (
                <article className={CARD} key={`${partition.group_name}-${index}`}>
                  <div className="flex items-start justify-between gap-3 [&_h4]:font-mono [&_h4]:text-[13px] [&_h4]:font-medium">
                    <div>
                      <h4>{partition.group_name}</h4>
                      <span className={MUTED}>{partition.signal_type}</span>
                    </div>
                    <div className="flex flex-wrap items-center justify-end gap-1.5 text-right text-[13px]">
                      <strong>{partition.winner || 'No winner recorded'}</strong>
                      {partition.default_used ? (
                        <span className={PILL}>Default fallback</span>
                      ) : null}
                    </div>
                  </div>
                  <p className={RESULT_META}>
                    Winner score <strong>{projectionMetric(partition.winner_score)}</strong>
                    <span>
                      Margin <strong>{projectionMetric(partition.margin)}</strong>
                    </span>
                  </p>
                  <EvidenceDetail title={`Inspect ${partition.group_name} candidates`}>
                    <dl className={METRICS}>
                      <div>
                        <dt>Score semantics</dt>
                        <dd>{partition.semantics || 'Not recorded'}</dd>
                      </div>
                      <div>
                        <dt>Raw winner score</dt>
                        <dd>{projectionMetric(partition.raw_winner_score)}</dd>
                      </div>
                      <div>
                        <dt>Temperature</dt>
                        <dd>{projectionMetric(partition.temperature)}</dd>
                      </div>
                    </dl>
                    {partition.contenders?.length ? (
                      <div className={TABLE_WRAP}>
                        <table className={TABLE} aria-label={`${partition.group_name} candidates`}>
                          <thead>
                            <tr>
                              <th>Candidate</th>
                              <th>Raw score</th>
                              <th>Normalized score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {partition.contenders.map((contender, candidateIndex) => (
                              <tr key={`${contender.name}-${candidateIndex}`}>
                                <th scope="row">
                                  {contender.name}
                                  {contender.name === partition.winner ? (
                                    <span className={PILL}>Winner</span>
                                  ) : null}
                                </th>
                                <td>{projectionMetric(contender.raw_score)}</td>
                                <td>{projectionMetric(contender.normalized_score)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className={MUTED}>Candidate scores were not recorded.</p>
                    )}
                  </EvidenceDetail>
                </article>
              ))}
            </div>
          </section>
        ) : null}
        {scores.length > 0 ? (
          <section className="flex flex-col gap-2.5" aria-label="Weighted scores">
            <div className="flex items-center gap-2.5 [&_h3]:text-[13px] [&_h3]:font-semibold [&_p]:text-xs [&_p]:text-subtle-foreground">
              <span className={STAGE_NUMBER}>2</span>
              <div>
                <h3>Weighted scores</h3>
                <p>How signal inputs contribute to each score</p>
              </div>
              <span className={`${PILL} ml-auto`}>{scores.length}</span>
            </div>
            <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-2">
              {scores.map((score, index) => (
                <article className={CARD} id={scoreID(index)} key={`${score.name}-${index}`}>
                  <div className="flex items-start justify-between gap-3 [&_h4]:font-mono [&_h4]:text-[13px] [&_h4]:font-medium">
                    <h4>{score.name}</h4>
                    <div className="flex flex-wrap items-center justify-end gap-1.5 text-right text-[13px]">
                      <span className={MUTED}>Total</span>
                      <strong>{projectionMetric(score.total)}</strong>
                    </div>
                  </div>
                  <EvidenceDetail title={`Inspect ${score.name} calculation`}>
                    {score.inputs?.length ? (
                      <div className={TABLE_WRAP}>
                        <table className={TABLE} aria-label={`${score.name} calculation`}>
                          <thead>
                            <tr>
                              <th>Input</th>
                              <th>Value</th>
                              <th>Weight</th>
                              <th>Contribution</th>
                            </tr>
                          </thead>
                          <tbody>
                            {score.inputs.map((input, inputIndex) => (
                              <tr key={`${input.type}-${input.name}-${inputIndex}`}>
                                <th scope="row">
                                  {input.name || input.type}
                                  <small>{input.type}</small>
                                </th>
                                <td>{projectionMetric(input.value)}</td>
                                <td>{projectionMetric(input.weight)}</td>
                                <td>{projectionMetric(input.contribution)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className={MUTED}>Input contributions were not recorded.</p>
                    )}
                  </EvidenceDetail>
                </article>
              ))}
            </div>
          </section>
        ) : null}
        {mappings.length > 0 ? (
          <section className="flex flex-col gap-2.5" aria-label="Routing outputs">
            <div className="flex items-center gap-2.5 [&_h3]:text-[13px] [&_h3]:font-semibold [&_p]:text-xs [&_p]:text-subtle-foreground">
              <span className={STAGE_NUMBER}>3</span>
              <div>
                <h3>Routing outputs</h3>
                <p>Which output matched each score mapping</p>
              </div>
              <span className={`${PILL} ml-auto`}>{mappings.length}</span>
            </div>
            <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-2">
              {mappings.map((mapping, index) => {
                const sourceIndex = scores.findIndex((score) => score.name === mapping.source_score)
                return (
                  <article className={CARD} key={`${mapping.mapping_name}-${index}`}>
                    <div className="flex items-start justify-between gap-3 [&_h4]:font-mono [&_h4]:text-[13px] [&_h4]:font-medium">
                      <h4>{mapping.mapping_name}</h4>
                      <strong className="flex flex-wrap items-center justify-end gap-1.5 text-right text-[13px]">
                        {mapping.selected_output || 'No output selected'}
                      </strong>
                    </div>
                    <div className={RESULT_META.replace('gap-x-4', 'items-center gap-x-2')}>
                      <span>
                        {mapping.source_score}{' '}
                        <strong>{projectionMetric(mapping.score_value)}</strong>
                      </span>
                      <ArrowRight className="size-3.5 text-subtle-foreground" />
                      <span>{mapping.selected_output || 'No match'}</span>
                    </div>
                    <EvidenceDetail title={`Inspect ${mapping.mapping_name} thresholds`}>
                      <dl className={METRICS}>
                        <div>
                          <dt>Confidence</dt>
                          <dd>{projectionMetric(mapping.confidence)}</dd>
                        </div>
                        <div>
                          <dt>Boundary distance</dt>
                          <dd>{projectionMetric(mapping.boundary_distance)}</dd>
                        </div>
                        <div>
                          <dt>Source score</dt>
                          <dd>
                            {sourceIndex >= 0 ? (
                              <a href={`#${scoreID(sourceIndex)}`}>{mapping.source_score}</a>
                            ) : (
                              mapping.source_score || 'Not recorded'
                            )}
                          </dd>
                        </div>
                      </dl>
                      {mapping.outputs?.length ? (
                        <div className={TABLE_WRAP}>
                          <table
                            className={TABLE}
                            aria-label={`${mapping.mapping_name} thresholds`}
                          >
                            <thead>
                              <tr>
                                <th>Output</th>
                                <th>Matched</th>
                                <th>Boundary distance</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mapping.outputs.map((output, outputIndex) => (
                                <tr key={`${output.name}-${outputIndex}`}>
                                  <th scope="row">{output.name}</th>
                                  <td>{output.matched ? 'Yes' : 'No'}</td>
                                  <td>{projectionMetric(output.boundary_distance)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className={MUTED}>Threshold evaluations were not recorded.</p>
                      )}
                    </EvidenceDetail>
                  </article>
                )
              })}
            </div>
          </section>
        ) : null}
      </div>
      <EvidenceDetail
        title={`Raw projection evidence · schema ${trace.schema_version || 'unknown'}`}
      >
        <pre className={RAW}>{JSON.stringify(trace, null, 2)}</pre>
      </EvidenceDetail>
    </div>
  )
}
