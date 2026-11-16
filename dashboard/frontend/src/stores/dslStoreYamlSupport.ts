import type { DSLState } from './dslStoreTypes'
import { deployBaseYaml } from './dslStoreSupport'

export async function renderCanonicalYaml(
  yaml: string,
  dsl: string,
  base: Pick<DSLState, 'baseConfigYaml' | 'baseConfigFromRouter'>,
  signal?: AbortSignal,
): Promise<string> {
  if (!base.baseConfigYaml.trim()) return yaml
  const baseYaml = deployBaseYaml(base)

  const response = await fetch('/api/router/config/deploy/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ yaml, dsl, baseYaml, mode: 'replace' }),
    signal,
  })
  if (!response.ok) {
    throw new Error(`Failed to render full YAML: HTTP ${response.status}`)
  }
  const preview = (await response.json()) as { preview?: string }
  return preview.preview?.trim() ? preview.preview : yaml
}
