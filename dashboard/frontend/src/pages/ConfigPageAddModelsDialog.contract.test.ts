import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('provider-first model onboarding', () => {
  it('starts with the four product providers in the requested order', () => {
    const catalog = readSource('./modelProviderCatalog.ts')
    const vllm = catalog.indexOf("'vllm',")
    const sglang = catalog.indexOf("'sglang',")
    const atom = catalog.indexOf("'amd-atom',")
    const compatible = catalog.indexOf("'openai-compatible',")

    expect(vllm).toBeGreaterThan(-1)
    expect(vllm).toBeLessThan(sglang)
    expect(sglang).toBeLessThan(atom)
    expect(atom).toBeLessThan(compatible)
    expect(catalog.indexOf("'openrouter',")).toBeGreaterThan(compatible)
  })

  it('preserves the complete model, endpoint, and pricing editor in Advanced', () => {
    const dialog = readSource('./ConfigPageAddModelsDialog.tsx')

    for (const label of [
      'Parameter size',
      'Context window',
      'Modality',
      'Quality score',
      'Capabilities',
      'Tags',
      'LoRA adapters',
      'Wire protocol',
      'API version',
      'Chat path',
      'Extra headers',
      'Endpoint weight',
      'Cached input',
      'Cache write',
      'Output',
    ]) {
      expect(dialog).toContain(label)
    }
    expect(dialog).toContain('<ModelProviderLogo')
  })

  it('keeps both the decision list and each model picker independently scrollable', () => {
    const styles = readSource('./ConfigPageMixtureDialog.module.css')

    expect(styles).toMatch(/\.assignments\s*{[^}]*max-height:[^}]*overflow-y: auto;/s)
    expect(styles).toMatch(/\.modelPicker\s*{[^}]*max-height:[^}]*overflow-y: auto;/s)
  })
})
