import { active, seconds } from './model'
import { duration, elapsedSeconds } from './activityPresentation'
import { useActivityClock } from './useActivityClock'
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
    <div aria-label="Elapsed wall time">
      <span>Elapsed wall time</span>
      <strong>
        {running ? duration(elapsedSeconds(run.created_at, now)) : seconds(savedSeconds)}
      </strong>
      <small>
        {running ? 'Since this run was created · updates while active' : 'Saved run duration'}
      </small>
    </div>
  )
}
