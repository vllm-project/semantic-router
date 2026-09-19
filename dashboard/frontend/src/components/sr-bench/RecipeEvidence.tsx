import type { Report, Target } from './types'
import styles from './SrBench.module.css'

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

export default function RecipeEvidence({
  report,
  targets,
}: {
  report: Report | null
  targets: Target[]
}) {
  const momTargets = targets.filter((target) => target.kind === 'mom')
  if (!momTargets.length) return null
  const snapshots = record(record(report?.provenance.runner)?.recipe_snapshots)
  function download(id: string, value: Record<string, unknown>) {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }),
    )
    const link = document.createElement('a')
    link.href = url
    link.download = `${report?.run_id ?? 'run'}-${id}-recipe.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return (
    <section id="run-recipe">
      <h3>Frozen recipes</h3>
      <p className={styles.muted}>
        The service captures recipes only while the generated and active runtime hashes match the
        frozen target and the source configuration stays unchanged. Deployment details are redacted;
        runtime call acknowledgements are recorded separately.
      </p>
      {momTargets.map((target) => {
        const snapshot = record(snapshots?.[target.id])
        const verified =
          !!target.config_hash &&
          snapshot?.source === 'router_config_api_bracketed_hashes' &&
          snapshot.config_hash === target.config_hash &&
          snapshot.generated_runtime_hash === target.config_hash &&
          snapshot.active_runtime_hash === target.config_hash &&
          snapshot.redacted === true
        return (
          <div className={styles.caseDetail} key={target.id}>
            <h4>
              {target.id} · {verified ? 'Verified config snapshot' : 'Snapshot unavailable'}
            </h4>
            <dl className={styles.identity}>
              <dt>Configuration hash</dt>
              <dd>
                <code>{target.config_hash ?? 'Not recorded'}</code>
              </dd>
              {verified && (
                <>
                  <dt>Captured</dt>
                  <dd>{String(snapshot?.captured_at ?? 'Not recorded')}</dd>
                  <dt>Source configuration hash</dt>
                  <dd>
                    <code>{String(snapshot?.source_config_hash ?? 'Not recorded')}</code>
                  </dd>
                  <dt>Recipe hash</dt>
                  <dd>
                    <code>{String(snapshot?.recipe_sha256 ?? 'Not recorded')}</code>
                  </dd>
                </>
              )}
            </dl>
            {verified && snapshot ? (
              <>
                <details>
                  <summary>Inspect recipe and routing requirements</summary>
                  <pre>{JSON.stringify(snapshot, null, 2)}</pre>
                </details>
                <div className={styles.actions}>
                  <button onClick={() => download(target.id, snapshot)}>
                    Download {target.id} recipe
                  </button>
                </div>
              </>
            ) : (
              <p className={styles.muted}>
                This run does not contain a matching server-captured recipe. The recorded
                configuration hash remains available; no recipe is inferred from later
                configuration.
              </p>
            )}
          </div>
        )
      })}
    </section>
  )
}
