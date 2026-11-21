import { Download } from 'lucide-react'
import { Badge, Button } from '../../ui'
import type { Report, Target } from './types'
import { targetLabel } from './targetPresentation'
import { BenchBlock, Disclosure } from './benchUi'
import { benchText, codeBlockClass, identityClass } from './benchStyles'

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
    <section id="run-recipe" className="min-w-0">
      <h3 className={benchText.h3}>Frozen recipes</h3>
      <p className={`${benchText.muted} max-w-3xl`}>
        The service captures recipes only while the generated and active runtime hashes match the
        frozen target and the source configuration stays unchanged. Deployment details are redacted;
        runtime call acknowledgements are recorded separately.
      </p>
      <div className="mt-3 grid gap-3">
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
            <BenchBlock key={target.id}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className={benchText.h4}>
                  {targetLabel(target)} ·{' '}
                  {verified ? 'Verified config snapshot' : 'Snapshot unavailable'}
                </h4>
                <Badge tone={verified ? 'success' : 'neutral'} size="sm" dot>
                  {verified ? 'Verified' : 'Not captured'}
                </Badge>
              </div>
              <dl className={identityClass}>
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
                  <Disclosure summary="Inspect recipe and routing requirements">
                    <pre className={codeBlockClass}>{JSON.stringify(snapshot, null, 2)}</pre>
                  </Disclosure>
                  <Button variant="outline" size="sm" onClick={() => download(target.id, snapshot)}>
                    <Download aria-hidden="true" />
                    Download {targetLabel(target)} recipe
                  </Button>
                </>
              ) : (
                <p className={benchText.muted}>
                  This run does not contain a matching server-captured recipe. The recorded
                  configuration hash remains available; no recipe is inferred from later
                  configuration.
                </p>
              )}
            </BenchBlock>
          )
        })}
      </div>
    </section>
  )
}
