import { Layers } from 'lucide-react'
import { Badge, Progress, ProviderLogo, ProviderStack, StatusDot, Tooltip, cn } from '../../ui'
import { catalogBenchmarkTitle } from './datasetPresentation'
import { runProgressShare, runStatusLabel, runStatusTone, targetProvider } from './runPresentation'
import { targetLabel } from './targetPresentation'
import type { Catalog, Run, Target } from './types'

/** One status value for tables: a dot and label, with a bar and percentage while running. */
export function RunStatusCell({
  run,
  className,
}: {
  run: Pick<Run, 'status' | 'progress'>
  className?: string
}) {
  if (run.status === 'running') {
    const share = Math.floor(runProgressShare(run) * 100)
    return (
      <span className={cn('flex w-32 flex-col gap-1.5', className)}>
        <span className="flex items-center justify-between gap-2 text-xs">
          <span className="inline-flex items-center gap-1.5 text-foreground/85">
            <StatusDot tone="info" pulse />
            Running
          </span>
          <span className="font-mono tabular text-muted-foreground">{share}%</span>
        </span>
        <Progress value={share} className="h-1" aria-label="Run progress" />
      </span>
    )
  }
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-[13px]', className)}>
      <StatusDot tone={runStatusTone(run.status)} />
      {runStatusLabel(run.status)}
    </span>
  )
}

export function TargetChip({ target, className }: { target: Target; className?: string }) {
  const recipe = target.kind === 'mom'
  return (
    <span
      className={cn(
        'inline-flex h-6 max-w-[180px] items-center gap-1.5 rounded-md border border-border bg-surface-subtle px-1.5 text-xs',
        className,
      )}
    >
      {recipe ? (
        <Layers className="size-3 shrink-0 text-chart-1" aria-hidden />
      ) : (
        <ProviderLogo provider={targetProvider(target)} size={12} />
      )}
      <span className={cn('truncate', recipe && 'font-mono text-[11.5px]')}>
        {targetLabel(target)}
      </span>
    </span>
  )
}

/** Compact target summary for tables: recipes by name, models as logos. */
export function TargetSummary({ targets }: { targets: Target[] }) {
  const recipes = targets.filter((target) => target.kind === 'mom')
  const models = targets.filter((target) => target.kind !== 'mom')
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      {recipes.slice(0, 1).map((target) => (
        <TargetChip key={target.id} target={target} />
      ))}
      {recipes.length > 1 ? (
        <Tooltip content={recipes.map(targetLabel).join(', ')}>
          <span className="text-xs text-subtle-foreground">+{recipes.length - 1}</span>
        </Tooltip>
      ) : null}
      {models.length ? (
        <Tooltip content={models.map(targetLabel).join(', ')}>
          <span className="inline-flex min-w-0 items-center gap-1.5">
            <ProviderStack providers={models.map(targetProvider)} size={14} max={3} />
            {recipes.length === 0 ? (
              <span className="truncate text-xs text-muted-foreground">
                {models.length === 1 ? targetLabel(models[0]) : `${models.length} models`}
              </span>
            ) : null}
          </span>
        </Tooltip>
      ) : null}
    </span>
  )
}

export function BenchmarkChips({
  ids,
  catalog,
  max = 2,
}: {
  ids: string[]
  catalog?: Catalog | null
  max?: number
}) {
  if (!ids.length) return <span className="text-subtle-foreground">—</span>
  const title = (id: string) => catalogBenchmarkTitle(id, catalog?.benchmarks)
  const extra = ids.slice(max)
  return (
    <span className="flex min-w-0 items-center gap-1">
      {ids.slice(0, max).map((id) => (
        <Badge key={id} tone="outline" className="max-w-[140px] truncate text-foreground/80">
          {title(id)}
        </Badge>
      ))}
      {extra.length ? (
        <Tooltip content={extra.map(title).join(', ')}>
          <span className="text-xs text-subtle-foreground">+{extra.length}</span>
        </Tooltip>
      ) : null}
    </span>
  )
}
