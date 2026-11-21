import type { RecipeProbeValidationResult } from '../types/recipe'

export interface RecipeProbeKey {
  id: string
  decision: string
  variant: string
}

export const RECIPE_PROBE_PAGE_SIZE = 50

export function mapRecipeProbeFacetOptions(values: Record<string, number>, current: string) {
  const options = Object.entries(values).sort(([left], [right]) => left.localeCompare(right))
  if (current && !Object.prototype.hasOwnProperty.call(values, current)) {
    options.unshift([current, 0])
  }
  return options
}

/** A validation's outcome: the route held, it did not, or the Router could not judge it. */
export function validationOutcome(result: RecipeProbeValidationResult) {
  if (result.error) return 'unavailable' as const
  return result.passed ? ('pass' as const) : ('mismatch' as const)
}
