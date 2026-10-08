import type { ModelRuntimeInventory } from './decisionModelManagement'

// Engine exposes native model cards. Keep observed readiness and deployment
// identity; configuration defaults must never manufacture ready resources.
export function engineModelInventory(value: unknown): ModelRuntimeInventory {
  if (!value || typeof value !== 'object' || !Array.isArray((value as { data?: unknown }).data))
    throw new Error('Engine model inventory is unavailable.')
  const data = (value as { data: Array<Record<string, unknown>> }).data
  const text = (item: Record<string, unknown>, key: string) =>
    typeof item[key] === 'string' ? (item[key] as string) : undefined
  return {
    deployments: data.flatMap((card) => {
      if (!card || typeof card.id !== 'string' || typeof card.ready !== 'boolean') return []
      return [
        {
          name: card.id,
          managed: true,
          process: 'instance-engine',
          served_name: card.id,
          ready: card.ready,
          state: text(card, 'status') || (card.ready ? 'ready' : 'unknown'),
          repo: text(card, 'repo'),
          family: text(card, 'family'),
          device: text(card, 'device'),
          engine: text(card, 'engine'),
          reason: text(card, 'reason'),
          revision: text(card, 'revision'),
          surfaces: Array.isArray(card.surfaces)
            ? card.surfaces.filter((entry): entry is string => typeof entry === 'string')
            : [],
        },
      ]
    }),
  }
}
