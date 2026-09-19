export function projectionMetric(value: number | undefined | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : 'Not recorded'
}
