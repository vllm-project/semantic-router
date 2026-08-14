export async function renderCanonicalYaml(
  yaml: string,
  dsl: string,
  baseYaml: string,
): Promise<string> {
  if (!baseYaml.trim()) return yaml

  const response = await fetch('/api/router/config/deploy/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ yaml, dsl, baseYaml, mode: 'replace' }),
  })
  if (!response.ok) {
    throw new Error(`Failed to render full YAML: HTTP ${response.status}`)
  }
  const preview = await response.json() as { preview?: string }
  return preview.preview?.trim() ? preview.preview : yaml
}
