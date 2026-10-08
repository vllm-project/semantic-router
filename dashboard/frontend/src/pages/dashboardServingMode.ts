export function describeServingMode(mode: unknown): { label: string; description: string } {
  switch (mode) {
    case 'router':
      return {
        label: 'Router mode',
        description: 'Routes requests through configured recipes.',
      }
    case 'engine':
      return {
        label: 'Engine mode',
        description: 'Serves model inference APIs directly.',
      }
    default:
      return {
        label: 'Mode unavailable',
        description: 'The connected service could not be identified.',
      }
  }
}
