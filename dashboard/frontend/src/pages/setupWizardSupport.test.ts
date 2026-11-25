import { describe, expect, it } from 'vitest'
import {
  buildSetupConfig,
  createSetupRequestGuard,
  createModelDraft,
  filterSetupModels,
  getModelDraftFieldErrors,
  getStepOneErrors,
  decisionModelFits,
  hostMemoryGiB,
  isCpuOnlyPlatform,
  maskedSetupYaml,
  paginateSetupModels,
  recommendedDecisionModels,
  removeSetupModel,
  restoreSetupModel,
  summarizeSetupConfig,
  upcomingCpuDecisionModels,
  switchSetupModelProvider,
  type ModelDraft,
} from './setupWizardSupport'

function modelDraft(overrides: Partial<ModelDraft> = {}): ModelDraft {
  return {
    id: 'model-a',
    name: 'qwen/qwen3.5-rocm',
    providerKind: 'vllm',
    baseUrl: 'vllm:8000',
    accessKey: '',
    endpointName: 'primary',
    ...overrides,
  }
}

describe('setup wizard model drafts', () => {
  it('keeps the first model prefilled and opens added models as editable drafts', () => {
    const first = createModelDraft(1)
    const second = createModelDraft(2, [first])

    expect(first.name).toBe('qwen/qwen3.5-rocm')
    expect(first.baseUrl).toBe('vllm:8000')
    expect(second.name).toBe('')
    expect(second.baseUrl).toBe('vllm:8001')
  })

  it('skips local vLLM endpoints that are already used', () => {
    const next = createModelDraft(3, [
      modelDraft({ id: 'model-a', baseUrl: 'vllm:8000' }),
      modelDraft({
        id: 'model-b',
        name: 'qwen/qwen3.5-rocm-alt',
        baseUrl: 'http://vllm:8001',
      }),
    ])

    expect(next.baseUrl).toBe('vllm:8002')
  })

  it('selects an Ollama endpoint when switching an untouched local model', () => {
    const model = switchSetupModelProvider(modelDraft(), 'ollama')
    expect(model.providerKind).toBe('ollama')
    expect(model.baseUrl).toBe('host.docker.internal:11434')

    const config = buildSetupConfig([model], model.id) as {
      providers: { models: Array<{ backend_refs: Array<Record<string, unknown>> }> }
    }
    expect(config.providers.models[0].backend_refs[0]).toMatchObject({
      provider: 'ollama',
      endpoint: 'host.docker.internal:11434',
      protocol: 'http',
    })
    expect(switchSetupModelProvider(modelDraft({ baseUrl: 'vllm:8001' }), 'ollama').baseUrl).toBe(
      'host.docker.internal:11434',
    )
  })

  it('keeps an explicitly entered endpoint when changing providers', () => {
    const model = switchSetupModelProvider(
      modelDraft({ baseUrl: 'http://my-model-host:11434' }),
      'ollama',
    )
    expect(model.baseUrl).toBe('http://my-model-host:11434')
  })

  it('reports duplicate model names and local endpoints at field level', () => {
    const models = [
      modelDraft({ id: 'model-a', name: 'alpha', baseUrl: 'vllm:8000' }),
      modelDraft({
        id: 'model-b',
        name: 'Alpha',
        baseUrl: 'http://vllm:8000',
      }),
    ]

    const errors = getModelDraftFieldErrors(models)

    expect(errors['model-a'].name).toBe('Model name "alpha" is duplicated.')
    expect(errors['model-b'].name).toBe('Model name "Alpha" is duplicated.')
    expect(errors['model-a'].baseUrl).toBe(
      'Endpoint "vllm:8000" is already used by another local model.',
    )
    expect(errors['model-b'].baseUrl).toBe(
      'Endpoint "vllm:8000" is already used by another local model.',
    )
  })

  it('includes duplicate local endpoints in step one summary errors', () => {
    const models = [
      modelDraft({ id: 'model-a', name: 'alpha', baseUrl: 'vllm:8000' }),
      modelDraft({
        id: 'model-b',
        name: 'beta',
        baseUrl: 'http://vllm:8000',
      }),
    ]

    expect(getStepOneErrors(models, 'model-a')).toEqual([
      'Endpoint "vllm:8000" is already used by another local model.',
    ])
  })

  it('filters and bounds large model inventories', () => {
    const models = Array.from({ length: 9 }, (_, index) =>
      modelDraft({
        id: `model-${index}`,
        name: index === 7 ? 'anthropic/claude' : `qwen/model-${index}`,
        baseUrl: `vllm:${8000 + index}`,
      }),
    )

    expect(filterSetupModels(models, 'CLAUDE').map((model) => model.id)).toEqual(['model-7'])
    expect(paginateSetupModels(models, 3, 4)).toMatchObject({
      page: 3,
      pageCount: 3,
      total: 9,
      startIndex: 8,
    })
    expect(paginateSetupModels(models, 99, 4).items.map((model) => model.id)).toEqual(['model-8'])
  })

  it('removes and restores a default model without losing its position', () => {
    const models = [
      modelDraft({ id: 'model-a', name: 'alpha' }),
      modelDraft({ id: 'model-b', name: 'beta', baseUrl: 'vllm:8001' }),
      modelDraft({ id: 'model-c', name: 'gamma', baseUrl: 'vllm:8002' }),
    ]

    const removal = removeSetupModel(models, 'model-b', 'model-b')
    expect(removal.models.map((model) => model.id)).toEqual(['model-a', 'model-c'])
    expect(removal.defaultModelId).toBe('model-a')

    const restored = restoreSetupModel(removal.models, removal.removed!, removal.defaultModelId)
    expect(restored.models.map((model) => model.id)).toEqual(['model-a', 'model-b', 'model-c'])
    expect(restored.defaultModelId).toBe('model-b')
  })

  it('drops stale request generations', () => {
    const guard = createSetupRequestGuard()
    const first = guard.begin()
    const second = guard.begin()

    expect(guard.isCurrent(first)).toBe(false)
    expect(guard.isCurrent(second)).toBe(true)
    guard.invalidate()
    expect(guard.isCurrent(second)).toBe(false)
  })

  it('summarizes provider models and the default recipe', () => {
    expect(
      summarizeSetupConfig({
        providers: { models: [{ name: 'alpha' }, { name: 'beta' }] },
        recipes: [
          {
            name: 'default',
            decisions: [{ name: 'default' }],
            signals: { domains: [{ name: 'code' }], keywords: [{ name: 'fast' }] },
          },
        ],
      }),
    ).toEqual({ models: 2, decisions: 1, signals: 2, canActivate: true })
  })

  it('summarizes every recipe', () => {
    expect(
      summarizeSetupConfig({
        providers: { models: [{ name: 'shared' }] },
        recipes: [
          {
            name: 'balanced',
            decisions: [{ name: 'balanced-route' }],
            signals: { keywords: [{ name: 'balanced-keyword' }] },
          },
          {
            name: 'private',
            decisions: [{ name: 'private-route' }],
            signals: { pii: [{ name: 'private-pii' }] },
          },
        ],
      }),
    ).toEqual({ models: 1, decisions: 2, signals: 2, canActivate: true })
  })
})

describe('setup wizard provider projection', () => {
  it('derives provider wire behavior from the shared provider catalog', () => {
    const models = [
      modelDraft({ id: 'local', name: 'qwen/qwen3.5-rocm' }),
      modelDraft({
        id: 'claude',
        name: 'anthropic/claude-opus-4.1',
        providerKind: 'anthropic',
        baseUrl: 'https://api.anthropic.com',
      }),
    ]

    const config = buildSetupConfig(models, 'local') as {
      providers: { models: Array<Record<string, unknown>> }
    }
    const [local, claude] = config.providers.models

    expect(local).not.toHaveProperty('api_format')
    expect(local.backend_refs).toEqual([
      expect.objectContaining({ provider: 'vllm', endpoint: 'vllm:8000' }),
    ])
    expect(claude).toMatchObject({ api_format: 'messages' })
    expect(claude.backend_refs).toEqual([
      expect.objectContaining({
        provider: 'anthropic',
        endpoint: 'https://api.anthropic.com',
      }),
    ])
  })
})

describe('setup decision model recommendation', () => {
  const option = (name: string, label = name) => ({ name, label, artifact: `vllm-sr/${name}` })
  const catalog = [option('vela3-fast', 'Vela 3.0 fast'), option('Vela-2.0-0.3B', 'Vela 2.0 0.3B')]

  it('treats a dashboard without a GPU platform as CPU-only', () => {
    expect(isCpuOnlyPlatform('')).toBe(true)
    expect(isCpuOnlyPlatform('cpu')).toBe(true)
    expect(isCpuOnlyPlatform('rocm')).toBe(false)
    expect(isCpuOnlyPlatform('CUDA')).toBe(false)
  })

  it('recommends the small Vela 3.1 models on CPU when the catalog has them', () => {
    const withVela31 = [
      ...catalog,
      option('vela3.1-pico-380m', 'Vela 3.1 pico 380M'),
      option('vela3.1-fast', 'Vela 3.1 fast'),
      option('vela3.1-pico-150m', 'Vela 3.1 pico 150M'),
    ]
    expect(
      recommendedDecisionModels(withVela31, true, 'vela3-fast').map((entry) => entry.name),
    ).toEqual(['vela3.1-fast', 'vela3.1-pico-150m', 'vela3.1-pico-380m'])
    expect(
      recommendedDecisionModels(withVela31, false, 'vela3-fast').map((entry) => entry.name),
    ).toEqual(['vela3-fast'])
  })

  it('falls back to the default when the catalog has no Vela 3.1 model', () => {
    expect(
      recommendedDecisionModels(catalog, true, 'vela3-fast').map((entry) => entry.name),
    ).toEqual(['vela3-fast'])
  })

  it('recommends only the CPU models that fit the host memory, largest first', () => {
    const sized = [
      { ...option('vela3-fast', 'Vela 3.0 fast'), minMemoryGiB: 12 },
      { ...option('vela3.1-fast', 'Vela 3.1 fast'), minMemoryGiB: 15 },
      { ...option('vela3.1-pico-150m', 'Vela 3.1 pico 150M'), minMemoryGiB: 2 },
      { ...option('vela3.1-pico-380m', 'Vela 3.1 pico 380M'), minMemoryGiB: 3 },
    ]
    const names = (memoryGiB: number | null) =>
      recommendedDecisionModels(sized, true, 'vela3-fast', memoryGiB).map((entry) => entry.name)
    // A 7.7 GiB Docker host cannot run Vela 3.1 fast (15 GiB).
    expect(names(hostMemoryGiB(8_268_021_760))).toEqual(['vela3.1-pico-380m', 'vela3.1-pico-150m'])
    expect(names(16)).toEqual(['vela3.1-fast', 'vela3.1-pico-380m', 'vela3.1-pico-150m'])
    // Unknown memory keeps the catalog order; nothing fitting keeps every CPU model.
    expect(names(null)).toEqual(['vela3.1-fast', 'vela3.1-pico-150m', 'vela3.1-pico-380m'])
    expect(names(1)).toEqual(['vela3.1-fast', 'vela3.1-pico-150m', 'vela3.1-pico-380m'])
    // GPU deployments are not compared against host memory.
    expect(
      recommendedDecisionModels(sized, false, 'vela3-fast', 4).map((entry) => entry.name),
    ).toEqual(['vela3-fast'])
  })

  it('reads unknown host memory as fitting every model', () => {
    expect(hostMemoryGiB(0)).toBeNull()
    expect(hostMemoryGiB(undefined)).toBeNull()
    expect(hostMemoryGiB(2 * 1024 ** 3)).toBe(2)
    expect(decisionModelFits({ minMemoryGiB: 15 }, null)).toBe(true)
    expect(decisionModelFits({ minMemoryGiB: null }, 4)).toBe(true)
    expect(decisionModelFits({ minMemoryGiB: 15 }, 7.7)).toBe(false)
    expect(decisionModelFits({ minMemoryGiB: 3 }, 7.7)).toBe(true)
  })

  it('lists the Vela 3.1 models the catalog announces before the runtime serves them', () => {
    const announced = [
      { ...option('vela3-fast', 'Vela 3.0 fast'), servable: true },
      { ...option('vela3.1-fast', 'Vela 3.1 fast'), servable: false },
      { ...option('vela3.1-pico-150m', 'Vela 3.1 pico 150M'), servable: true },
    ]
    expect(upcomingCpuDecisionModels(announced).map((entry) => entry.name)).toEqual([
      'vela3.1-fast',
    ])
  })

  it('previews the draft as YAML with access keys masked', () => {
    const yaml = maskedSetupYaml({
      providers: { models: [{ name: 'm', backend_refs: [{ api_key: 'sk-secret' }] }] },
    })
    expect(yaml).toContain('api_key: ••••••••')
    expect(yaml).not.toContain('sk-secret')
  })
})
