export function describeServingMode(mode: unknown): { label: string; description: string } {
  switch (mode) {
    case 'router':
      return {
        label: 'Router mode',
        description: 'Serves System One and routes Chat through configured recipes.',
      }
    case 'engine':
      return {
        label: 'Engine mode',
        description: 'Serves System One with routing disabled.',
      }
    default:
      return {
        label: 'Mode unavailable',
        description: 'The connected service could not be identified.',
      }
  }
}
