import type { Run } from './types'

/** Navigation eligibility only; candidate-plan revalidates the frozen protocol. */
export function canReuseBaseline(run: Run | null | undefined): run is Run {
  return !!(
    run &&
    ['completed', 'failed', 'cancelled', 'interrupted'].includes(run.status) &&
    run.manifest.mode === 'live' &&
    run.manifest.targets.some((target) => target.kind === 'single') &&
    !run.manifest.recovery &&
    run.manifest.execution_cells == null
  )
}
