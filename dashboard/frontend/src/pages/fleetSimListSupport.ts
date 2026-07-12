export function matchesFleetSimSearch(search: string, values: readonly unknown[]): boolean {
  const query = search.trim().toLocaleLowerCase()
  if (!query) return true

  return values.some((value) =>
    typeof value === 'string' || typeof value === 'number'
      ? String(value).toLocaleLowerCase().includes(query)
      : false,
  )
}

export function parseArrivalRateCheckpoints(values: readonly string[]): number[] {
  if (values.length === 0) {
    throw new Error('Add at least one arrival-rate checkpoint for the what-if sweep.')
  }

  const checkpoints = values.map((value, index) => {
    const checkpoint = Number(value.trim())
    if (!Number.isFinite(checkpoint) || checkpoint <= 0) {
      throw new Error(`Arrival-rate checkpoint ${index + 1} must be a positive number.`)
    }
    return checkpoint
  })

  if (new Set(checkpoints).size !== checkpoints.length) {
    throw new Error('Arrival-rate checkpoints must be unique.')
  }
  return checkpoints
}
