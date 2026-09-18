import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import {
  getLoadedModelCount,
  getPreviewRouterModels,
  getRouterModelConsumers,
  getRouterModelResources,
  getTotalKnownModelCount,
  type RouterModelInfo,
} from '../utils/routerRuntime'
import RouterModelInventory from './RouterModelInventory'
import {
  getRouterModelArtifactPath,
  getRouterModelDevice,
  getRouterModelDisplayName,
  getRouterModelInputLimits,
  getRouterModelPreviewName,
} from './routerModelPresentation'
import { filterAndSortRouterModels } from './routerModelInventorySupport'

// Public repository identities and synthetic local derivation paths; no fleet data.
const models: RouterModelInfo[] = [
  ['category_classifier', 'intent_classification', 'Domain'],
  ['feedback_detector', 'feedback_detection', 'Feedback'],
  ['fact_check_classifier', 'fact_check_classification', 'FactCheck'],
  ['mmbert_embedding_model', 'embedding', 'Embedding'],
  ['prompt_guard', 'jailbreak_detection', 'Guard'],
  ['pii_classifier', 'pii_detection', 'PII'],
  ['safety_unsafe', 'safety_detection', 'Safety'],
  ['classifier_content_risk', 'label_scores', 'Hazard'],
].map(([name, type, task]) => ({
  name,
  type,
  recipe: 'example',
  loaded: true,
  model_path: `models/local-ckfa/${task}-0123456789ab`,
  registry: {
    repo_id: `llm-semantic-router/Vela-1.0-Encoder-307M-${task}`,
    revision: 'test-registry-revision',
    max_context_length: 32768,
    model_card_url: `https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-${task}`,
  },
  metadata: {
    binding: name,
    resource_id: `resource-${task.toLowerCase()}`,
    contract: task === 'Embedding' ? 'embedding.v1' : 'label_distribution.v1',
    deployment: `${task.toLowerCase()}-gpu`,
    device: 'rocm:0',
    max_sequence_length: '32768',
    input_max_tokens: '32768',
    forward_max_tokens: '32768',
    overflow: 'truncate',
    precision: 'native',
    provider: 'ort',
  },
}))

function render(
  modelList: RouterModelInfo[],
  mode: 'preview' | 'detail' | 'full' = 'preview',
): string {
  return renderToStaticMarkup(
    <RouterModelInventory modelsInfo={{ models: modelList }} mode={mode} />,
  )
}

function headings(markup: string): string[] {
  return [...markup.matchAll(/<h3[^>]*>(.*?)<\/h3>/g)].map((match) => match[1])
}

describe('router model presentation', () => {
  it('shows all eight canonical HF model names, including PII, Safety and Embedding', () => {
    const markup = render(models)
    expect(headings(markup).sort()).toEqual(
      models.map((model) => model.registry!.repo_id!.split('/')[1]).sort(),
    )
    expect(headings(markup)).toHaveLength(8)
    expect(markup.match(/>llm-semantic-router<\/p>/g)).toHaveLength(8)
    expect(markup).not.toContain('Vela Embedding')
    expect(markup).not.toContain('0123456789ab')
    expect(markup).not.toContain('models/local-ckfa')
    expect(getPreviewRouterModels({ models })).toHaveLength(8)
  })

  it('prefers the reported repository over even a recognizable local derivation name', () => {
    const model = {
      ...models[0],
      model_path: 'models/Vela-1.0-Encoder-307M-Feedback-CK-32K-local-0123456789ab',
    }
    expect(getRouterModelDisplayName(model)).toBe(models[0].registry?.repo_id)
    expect(headings(render([model]))).toEqual(['Vela-1.0-Encoder-307M-Domain'])
    const detail = render([model], 'detail')
    expect(detail).toContain(model.model_path)
    expect(detail).toContain('test-registry-revision')
    expect(detail).toContain('Open model card')
  })

  it('explicitly identifies unknown local models instead of inventing HF identity from hash paths', () => {
    const model: RouterModelInfo = { ...models[0], registry: undefined }
    expect(getRouterModelPreviewName(model)).toEqual({
      title: model.name,
      subtitle: 'Runtime identity · repository not reported',
    })
    expect(headings(render([model]))).toEqual([model.name])
    expect(render([model])).not.toContain('Vela-')
    const detail = render([model], 'detail')
    expect(detail).toContain(model.model_path)
    expect(detail).not.toContain('Open model card')
    const resolved = { ...model, resolved_model_path: '/models/derived/current' }
    expect(getRouterModelArtifactPath(resolved)).toBe('/models/derived/current')
  })

  it('separates a 262K document budget from the actual 32K physical window', () => {
    const model: RouterModelInfo = {
      ...models[5],
      metadata: {
        ...models[5].metadata,
        max_sequence_length: '262144',
        input_max_tokens: '262144',
        document_max_tokens: '262144',
        forward_max_tokens: '32768',
        window_size: '32768',
        window_overlap: '256',
        overflow: 'window',
      },
    }
    expect(render([model])).toContain('Window: 32,768 tokens')
    expect(render([model])).toContain('Document budget: 262,144 tokens')
    const detail = render([model], 'detail')
    expect(detail).toMatch(/Document budget<\/dt><dd[^>]*>262,144 tokens/)
    expect(detail).toMatch(/Physical token window<\/dt><dd[^>]*>32,768 tokens/)
    expect(detail).toMatch(/Window overlap<\/dt><dd[^>]*>256 tokens/)
    expect(detail).toMatch(/Published model context<\/dt><dd[^>]*>32,768 tokens/)
    expect(detail).not.toContain('Max Sequence Length')
    expect(detail).not.toContain('Context Window')
  })

  it('keeps input limits, smaller configured windows and forward capacity distinct', () => {
    const limited: RouterModelInfo = {
      ...models[3],
      metadata: { ...models[3].metadata, input_max_tokens: '8192' },
    }
    const detail = render([limited], 'detail')
    expect(detail).toMatch(/Input budget<\/dt><dd[^>]*>8,192 tokens/)
    expect(detail).toMatch(/Single-forward capacity<\/dt><dd[^>]*>32,768 tokens/)
    const windowed = {
      ...limited,
      metadata: {
        ...limited.metadata,
        overflow: 'window',
        document_max_tokens: '262144',
        window_size: '2048',
        window_overlap: '0',
      },
    }
    expect(render([windowed], 'detail')).toMatch(/Physical token window<\/dt><dd[^>]*>2,048 tokens/)
    expect(getRouterModelInputLimits(windowed).overlap).toBe(0)
    expect(render([windowed], 'detail')).toMatch(/Window overlap<\/dt><dd[^>]*>0 tokens/)
  })

  it('does not reinterpret a legacy document budget or published context as a physical window', () => {
    const legacy: RouterModelInfo = {
      ...models[0],
      metadata: { max_sequence_length: '262144', overflow: 'window' },
    }
    expect(getRouterModelInputLimits(legacy).window).toBeUndefined()
    expect(render([legacy], 'detail')).toMatch(/Physical token window<\/dt><dd[^>]*>Not reported/)
    expect(render([legacy])).toContain('Document budget: 262,144 tokens')
    const registryOnly = { ...models[0], metadata: {} }
    expect(render([registryOnly])).toContain('Published model context: 32,768 tokens')
    expect(getRouterModelInputLimits(registryOnly).forward).toBeUndefined()
    for (const value of ['0', '-1', 'Infinity', 'NaN', '32K', '12.5']) {
      const limits = getRouterModelInputLimits({
        ...models[0],
        metadata: { input_max_tokens: value, window_size: value },
      })
      expect(limits.input).toBeUndefined()
      expect(limits.window).toBeUndefined()
    }
  })

  it('groups details and keeps full provenance in a collapsed technical section', () => {
    const detail = render([models[0]], 'detail')
    for (const section of ['Execution', 'Input limits', 'Model metadata', 'Consumer']) {
      expect(detail).toMatch(new RegExp(`<h4[^>]*>${section}</h4>`))
    }
    expect(detail).toMatch(/<details[^>]*><summary[^>]*>Technical details<\/summary>/)
    expect(detail).not.toMatch(/<details[^>]*\bopen\b/)
    expect(detail).toContain(models[0].model_path)
    expect(detail).toContain('resource-domain')
    expect(detail).toContain('example / category_classifier')
    const marketing = {
      ...models[0],
      registry: { ...models[0].registry, description: '<div>Promotional model card</div>' },
    }
    expect(render([marketing], 'detail')).not.toContain('Promotional model card')
  })

  it('shows only the model actual device and searches canonical names and provenance', () => {
    const cpu = { ...models[3], metadata: { ...models[3].metadata, device: 'cpu' } }
    expect(render([cpu], 'detail')).toContain('CPU')
    expect(render([cpu], 'detail')).not.toContain('amd-logo.png')
    expect(render([models[3]], 'detail')).toContain('ROCm 0')
    expect(render([models[3]])).not.toContain('AMD GPU')
    expect(getRouterModelDevice({ ...cpu, metadata: { device: 'migraphx:2' } })).toEqual({
      label: 'MIGraphX 2',
      isAmd: true,
    })
    expect(getRouterModelDevice({ ...cpu, metadata: {} })).toEqual({
      label: 'Device not reported',
      isAmd: false,
    })
    expect(filterAndSortRouterModels(models, models[0].registry!.repo_id!, 'all', 'name')).toEqual([
      models[0],
    ])
    expect(filterAndSortRouterModels(models, 'Domain-0123456789ab', 'all', 'name')).toEqual([
      models[0],
    ])
  })
})

describe('shared runtime resource inventory', () => {
  const embedding = models[3]
  const cache: RouterModelInfo = {
    ...embedding,
    recipe: '@global',
    metadata: {
      ...embedding.metadata,
      resource_id: 'shared-embedding-resource',
      binding: 'response_cache.embedding',
      default_layer: '6',
      default_dimension: '256',
    },
  }
  const routing: RouterModelInfo = {
    ...embedding,
    recipe: 'balance',
    metadata: {
      ...embedding.metadata,
      resource_id: 'shared-embedding-resource',
      binding: 'embedding',
      default_layer: '22',
      default_dimension: '768',
    },
  }

  it('counts and displays the same complete resource set without dropping consumers', () => {
    const inventory = [...models.filter((model) => model !== embedding), cache, routing]
    const info = { models: inventory, summary: { loaded_models: 9, total_models: 9 } }
    expect(getLoadedModelCount(info)).toBe(8)
    expect(getTotalKnownModelCount(info)).toBe(8)
    expect(headings(render(inventory))).toHaveLength(8)
    expect(getPreviewRouterModels(info)).toHaveLength(8)
    expect(getPreviewRouterModels(info, 6)).toHaveLength(6)
    expect(render(inventory)).toContain('Shared by 2 consumers')
    const detail = render([cache, routing], 'detail')
    for (const value of [
      'Consumers (2)',
      '@global / response_cache.embedding',
      'balance / embedding',
      'Layer 6',
      '256 dimensions',
      'Layer 22',
      '768 dimensions',
      'shared-embedding-resource',
    ]) {
      expect(detail).toContain(value)
    }
    expect(getRouterModelConsumers(inventory, cache)).toEqual([cache, routing])
  })

  it('does not invent cards or ready resources from a stale binding summary', () => {
    const info = { models: [cache, routing], summary: { loaded_models: 10, total_models: 12 } }
    expect(getLoadedModelCount(info)).toBe(1)
    expect(getTotalKnownModelCount(info)).toBe(1)
    expect(headings(render(info.models))).toHaveLength(1)
    expect(getTotalKnownModelCount({ models: null, summary: { total_models: 8 } })).toBe(8)
    expect(getLoadedModelCount({ models: [{ ...cache, state: 'initializing' }] })).toBe(0)
  })

  it('exposes different consumer input budgets without assigning one to the shared engine', () => {
    const limited = {
      ...routing,
      metadata: { ...routing.metadata, input_max_tokens: '8192' },
    }
    const detail = render([cache, limited], 'detail')
    expect(detail).toMatch(/Input budget<\/dt><dd[^>]*>Varies by consumer/)
    expect(detail).toContain('Input budget 8,192 tokens')
    expect(detail).toContain('Input budget 32,768 tokens')
    expect(render([cache, limited])).toContain('Input budget: Varies by consumer')
  })

  it('includes failed and loading models beside ready cards and never hides a failed shared consumer', () => {
    const failed = { ...routing, loaded: false, state: 'not_loaded' }
    const loading = { ...models[5], loaded: false, state: 'initializing' }
    const info = {
      models: [cache, failed, models[0], loading],
      summary: { loaded_models: 2, total_models: 4 },
    }
    const [resource] = getRouterModelResources(info.models)
    expect(resource.model.loaded).toBe(false)
    expect(resource.model.state).toBe('not_loaded')
    expect(getLoadedModelCount(info)).toBe(1)
    expect(getTotalKnownModelCount(info)).toBe(3)
    expect(getPreviewRouterModels(info)).toHaveLength(3)
    expect(headings(render(info.models))).toHaveLength(3)
    expect(render(info.models)).toContain('Not Loaded')
    expect(render(info.models)).toContain('Initializing')
    expect(render([cache, failed], 'detail')).toContain('Ready')
  })

  it('never merges separate resources just because their names, paths or HF repository match', () => {
    const cpu = {
      ...routing,
      metadata: { ...routing.metadata, resource_id: 'separate-cpu-resource', device: 'cpu' },
    }
    expect(getRouterModelResources([cache, cpu])).toHaveLength(2)
    const unknown = { ...routing, metadata: { ...routing.metadata, resource_id: '' } }
    expect(getRouterModelResources([unknown, { ...unknown, recipe: 'other' }])).toHaveLength(2)
  })
})
