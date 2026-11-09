import type { Effort, EffortFrom, Model, Steps } from '../../types'

export const EFFORTS: readonly Effort[] = ['low', 'medium', 'high', 'xhigh', 'max']

// The `steps` lines: below `low` a few steps, below `high` several, from it on many; under `minConfidence` the answer is not acted on.
export type StepsLines = { low: number; high: number; minConfidence: number }

// The effort for fewer steps, several and many. The middle rung is also the default when the steps answer cannot be used.
const LADDER: Record<Model, readonly [Effort, Effort, Effort]> = {
  haiku: ['medium', 'medium', 'medium'],
  sonnet: ['low', 'medium', 'high'],
  opus: ['high', 'high', 'xhigh'],
}

export type Picked = { effort: Effort; from: Exclude<EffortFrom, 'floor'> }

// One level up, at most max.
export const raised = (effort: Effort): Effort => EFFORTS[Math.min(EFFORTS.length - 1, EFFORTS.indexOf(effort) + 1)]

export function pickEffort(model: Model, steps: Steps | null, { low, high, minConfidence }: StepsLines): Picked {
  const [few, several, many] = LADDER[model]
  if (model === 'haiku') return { effort: several, from: 'haiku' }
  if (steps === null || steps.confidence < minConfidence) return { effort: several, from: 'unsure' }
  return { effort: steps.score < low ? few : steps.score < high ? several : many, from: 'steps' }
}

// What a request runs at: the picked effort, or for a route with a floor the effort the subagent would have had when that is higher.
export function effortFor(picked: Picked, floor: boolean, current: string): { effort: string; from: EffortFrom } {
  const isAbove = EFFORTS.indexOf(current as Effort) > EFFORTS.indexOf(picked.effort)
  return floor && isAbove ? { effort: current, from: 'floor' } : picked
}
