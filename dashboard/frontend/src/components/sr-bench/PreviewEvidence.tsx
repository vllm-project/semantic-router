import styles from './SrBench.module.css'

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function text(value: unknown): string {
  return typeof value === 'string' && value ? value : 'Not recorded'
}

export default function PreviewEvidence({ routing: value }: { routing: unknown }) {
  const routing = record(value)
  const decision = record(routing.decision_result)
  const provenance = record(routing.selection_provenance)
  return (
    <section aria-label="Routing preview evidence">
      <h4>Routing preview</h4>
      <dl className={styles.facts}>
        <div>
          <dt>Matched decision</dt>
          <dd>{text(decision.decision_name)}</dd>
        </div>
        <div>
          <dt>Selected model</dt>
          <dd>{text(routing.selected_model)}</dd>
        </div>
        <div>
          <dt>Selection status</dt>
          <dd>{text(routing.selection_status)}</dd>
        </div>
        <div>
          <dt>Selection method</dt>
          <dd>{text(routing.selection_method)}</dd>
        </div>
        <div>
          <dt>Config snapshot</dt>
          <dd>
            <code>{text(provenance.config_hash ?? routing.config_hash)}</code>
          </dd>
        </div>
      </dl>
      {typeof routing.selection_reason === 'string' && (
        <p className={styles.notice}>{routing.selection_reason}</p>
      )}
      {typeof provenance.mode === 'string' && (
        <>
          <h4>Selection provenance</h4>
          <dl className={styles.facts}>
            <div>
              <dt>Selection mode</dt>
              <dd>{text(provenance.mode)}</dd>
            </div>
            <div>
              <dt>Depends on mutable state</dt>
              <dd>
                {provenance.state_dependent === true
                  ? 'Yes · not eligible for replay'
                  : provenance.state_dependent === false
                    ? 'No'
                    : 'Not recorded'}
              </dd>
            </div>
            {typeof provenance.state_hash === 'string' && (
              <div>
                <dt>State snapshot</dt>
                <dd>
                  <code>{text(provenance.state_hash)}</code>
                </dd>
              </div>
            )}
            {typeof provenance.captured_at === 'string' && (
              <div>
                <dt>State captured at</dt>
                <dd>{text(provenance.captured_at)}</dd>
              </div>
            )}
            <div>
              <dt>Simulated sampling</dt>
              <dd>
                {provenance.sampled === true
                  ? 'Yes · later live choice may differ'
                  : provenance.sampled === false
                    ? 'No'
                    : 'Not recorded'}
              </dd>
            </div>
            {typeof provenance.sampling_seed === 'number' && (
              <div>
                <dt>Sampling seed</dt>
                <dd>{provenance.sampling_seed}</dd>
              </div>
            )}
          </dl>
          {typeof provenance.caveat === 'string' && (
            <p className={styles.notice}>{provenance.caveat}</p>
          )}
        </>
      )}
    </section>
  )
}
