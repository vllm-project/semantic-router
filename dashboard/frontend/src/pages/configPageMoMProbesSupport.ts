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
