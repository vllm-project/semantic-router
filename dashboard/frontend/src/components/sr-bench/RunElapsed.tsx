import { active } from './model'
import { duration, elapsedSeconds } from './activityPresentation'
import { useActivityClock } from './useActivityClock'
import { SummaryStat } from './benchUi'
import type { Run } from './types'

export default function RunElapsed({
  run,
  savedSeconds,
}: {
  run: Run
  savedSeconds?: number | null
}) {
  const running = active(run.status)
  const now = useActivityClock(running)
  return (
    <SummaryStat
      aria-label="Elapsed wall time"
      label="Duration"
      value={
        running
          ? duration(elapsedSeconds(run.created_at, now))
          : savedSeconds == null || !Number.isFinite(savedSeconds)
            ? '—'
            : duration(savedSeconds)
      }
      hint={running ? 'Since this run was created · updates while active' : 'Saved run duration'}
    />
  )
}
