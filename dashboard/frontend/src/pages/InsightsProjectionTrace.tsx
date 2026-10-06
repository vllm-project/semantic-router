import type { ReactNode } from 'react'

import ProductIcon from '../components/ProductIcon'
import type { ProjectionTrace } from './insightsPageTypes'
import { projectionMetric } from './insightsRoutingMetrics'
import styles from './InsightsRoutingEvidence.module.css'

function EvidenceDetail({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className={styles.detail}>
      <summary>
        <span>{title}</span>
        <ProductIcon name="chevron-down" width={14} height={14} />
      </summary>
      <div className={styles.detailBody}>{children}</div>
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
    <div className={styles.evidence}>
      <p className={styles.intro}>
        Follow the recorded inputs to their routing outputs. Expand a result to inspect its
        calculation.
      </p>
      {!partitions.length && !scores.length && !mappings.length ? (
        <p className={styles.muted}>No projection stages were recorded.</p>
      ) : null}
      <div className={styles.stages}>
        {partitions.length > 0 ? (
          <section className={styles.stage} aria-label="Signal groups">
            <div className={styles.stageHeader}>
              <span className={styles.stageNumber}>1</span>
              <div>
                <h3>Signal groups</h3>
                <p>Which signal won within each group</p>
              </div>
              <span className={styles.count}>{partitions.length}</span>
            </div>
            <div className={styles.results}>
              {partitions.map((partition, index) => (
                <article className={styles.result} key={`${partition.group_name}-${index}`}>
                  <div className={styles.resultHeading}>
                    <div>
                      <h4>{partition.group_name}</h4>
                      <span className={styles.muted}>{partition.signal_type}</span>
                    </div>
                    <div className={styles.outcome}>
                      <strong>{partition.winner || 'No winner recorded'}</strong>
                      {partition.default_used ? (
                        <span className={styles.badge}>Default fallback</span>
                      ) : null}
                    </div>
                  </div>
                  <p className={styles.resultMeta}>
                    Winner score <strong>{projectionMetric(partition.winner_score)}</strong>
                    <span>
                      Margin <strong>{projectionMetric(partition.margin)}</strong>
                    </span>
                  </p>
                  <EvidenceDetail title={`Inspect ${partition.group_name} candidates`}>
                    <dl className={styles.metrics}>
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
                      <div className={styles.tableScroll}>
                        <table aria-label={`${partition.group_name} candidates`}>
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
                                    <span className={styles.badge}>Winner</span>
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
                      <p className={styles.muted}>Candidate scores were not recorded.</p>
                    )}
                  </EvidenceDetail>
                </article>
              ))}
            </div>
          </section>
        ) : null}
        {scores.length > 0 ? (
          <section className={styles.stage} aria-label="Weighted scores">
            <div className={styles.stageHeader}>
              <span className={styles.stageNumber}>2</span>
              <div>
                <h3>Weighted scores</h3>
                <p>How signal inputs contribute to each score</p>
              </div>
              <span className={styles.count}>{scores.length}</span>
            </div>
            <div className={styles.results}>
              {scores.map((score, index) => (
                <article
                  className={styles.result}
                  id={scoreID(index)}
                  key={`${score.name}-${index}`}
                >
                  <div className={styles.resultHeading}>
                    <h4>{score.name}</h4>
                    <div className={styles.outcome}>
                      <span className={styles.muted}>Total</span>
                      <strong>{projectionMetric(score.total)}</strong>
                    </div>
                  </div>
                  <EvidenceDetail title={`Inspect ${score.name} calculation`}>
                    {score.inputs?.length ? (
                      <div className={styles.tableScroll}>
                        <table aria-label={`${score.name} calculation`}>
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
                                  <small>
                                    {[input.type, input.kb, input.metric]
                                      .filter(Boolean)
                                      .join(' · ')}
                                  </small>
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
                      <p className={styles.muted}>Input contributions were not recorded.</p>
                    )}
                  </EvidenceDetail>
                </article>
              ))}
            </div>
          </section>
        ) : null}
        {mappings.length > 0 ? (
          <section className={styles.stage} aria-label="Routing outputs">
            <div className={styles.stageHeader}>
              <span className={styles.stageNumber}>3</span>
              <div>
                <h3>Routing outputs</h3>
                <p>Which output matched each score mapping</p>
              </div>
              <span className={styles.count}>{mappings.length}</span>
            </div>
            <div className={styles.results}>
              {mappings.map((mapping, index) => {
                const sourceIndex = scores.findIndex((score) => score.name === mapping.source_score)
                return (
                  <article className={styles.result} key={`${mapping.mapping_name}-${index}`}>
                    <div className={styles.resultHeading}>
                      <h4>{mapping.mapping_name}</h4>
                      <strong className={styles.outcome}>
                        {mapping.selected_output || 'No output selected'}
                      </strong>
                    </div>
                    <div className={styles.relationship}>
                      <span>
                        {mapping.source_score}{' '}
                        <strong>{projectionMetric(mapping.score_value)}</strong>
                      </span>
                      <ProductIcon name="arrow-right" width={14} height={14} />
                      <span>{mapping.selected_output || 'No match'}</span>
                    </div>
                    <EvidenceDetail title={`Inspect ${mapping.mapping_name} thresholds`}>
                      <dl className={styles.metrics}>
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
                        <div className={styles.tableScroll}>
                          <table aria-label={`${mapping.mapping_name} thresholds`}>
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
                        <p className={styles.muted}>Threshold evaluations were not recorded.</p>
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
        <pre className={styles.raw}>{JSON.stringify(trace, null, 2)}</pre>
      </EvidenceDetail>
    </div>
  )
}
