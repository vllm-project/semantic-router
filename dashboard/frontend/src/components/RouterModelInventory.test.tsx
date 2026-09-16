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
  getRouterModelContext,
  getRouterModelDevice,
  getRouterModelDisplayName,
  getRouterModelPreviewName,
  isLocalDerivedRouterModel,
} from './routerModelPresentation'
import { filterAndSortRouterModels } from './routerModelInventorySupport'

// Public names and synthetic artifact hashes with the native inventory shape.
const models: RouterModelInfo[] = [
  ['category_classifier', 'intent_classification', 'Domain'],
  ['feedback_detector', 'feedback_detection', 'Feedback'],
  ['fact_check_classifier', 'fact_check_classification', 'FactCheck'],
  ['mmbert_embedding_model', 'embedding', 'Embedding'],
].map(([name, type, task]) => ({
  name,
  type,
  recipe: 'example',
  loaded: true,
  model_path: `models/Vela-1.0-Encoder-307M-${task}${task === 'Embedding' ? '' : '-CK-32K-local-0123456789ab'}`,
  metadata: {
    binding: name,
    contract: task === 'Embedding' ? 'embedding.v1' : 'label_distribution.v1',
    deployment: `${task.toLowerCase()}-gpu`,
    device: 'rocm:0',
    max_sequence_length: '32768',
    overflow: 'truncate',
    precision: 'native',
    provider: 'ort',
  },
}))

function render(modelList: RouterModelInfo[], mode: 'preview' | 'detail' = 'preview'): string {
  return renderToStaticMarkup(
    <RouterModelInventory modelsInfo={{ models: modelList }} mode={mode} />,
  )
}

function headings(markup: string): string[] {
  return [...markup.matchAll(/<h3[^>]*>(.*?)<\/h3>/g)].map((match) => match[1])
}

describe('router model presentation', () => {
  it('shows clean Vela names and exact token windows for all four native model cards', () => {
    const markup = render(models)
    expect(headings(markup).sort()).toEqual([
      'Vela Domain',
      'Vela Embedding',
      'Vela Fact Check',
      'Vela Feedback',
    ])
    expect(markup.match(/v1.0 · 307M encoder/g)).toHaveLength(4)
    expect(markup.match(/Context window: 32,768 tokens/g)).toHaveLength(4)
    expect(markup).not.toContain('CK-32K-local')
    expect(markup).not.toContain('0123456789ab')
    expect(markup).not.toContain('models/')
  })

  it('retains local provenance and runtime settings in details without inventing a registry', () => {
    const model = models[0]
    const markup = render([model], 'detail')
    expect(headings(markup)).toEqual(['Vela-1.0-Encoder-307M-Domain'])
    for (const value of [
      model.model_path,
      'Local derived artifact',
      'example',
      'domain-gpu',
      'rocm:0',
      '32768',
      'truncate',
      'Context Window',
      '32,768 tokens',
    ]) {
      expect(markup).toContain(value)
    }
    expect(markup).not.toContain('Open model card')
    expect(markup).not.toContain('huggingface.co')
  })

  it('shows the actual input budget separately from available registry capacity and source', () => {
    const model: RouterModelInfo = {
      ...models[3],
      metadata: { ...models[3].metadata, max_sequence_length: '8192' },
      registry: {
        repo_id: 'llm-semantic-router/Vela-1.0-Encoder-307M-Embedding',
        revision: 'test-registry-revision',
        max_context_length: 32768,
        model_card_url:
          'https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Embedding',
      },
    }
    expect(render([model])).toContain('Context window: 8,192 tokens')
    const detail = render([model], 'detail')
    expect(detail).toMatch(/Context Window<\/dt><dd[^>]*>8,192 tokens/)
    expect(detail).toMatch(/Registry Context<\/dt><dd[^>]*>32,768 tokens/)
    expect(detail).toContain('test-registry-revision')
    expect(detail).toContain('Open model card')
  })

  it('keeps device labels in details and off the compact homepage cards', () => {
    const cpu = {
      ...models[3],
      recipe: 'cpu-example',
      metadata: { ...models[3].metadata, device: 'cpu' },
    }
    const markup = render([models[3], cpu])
    expect(markup).not.toContain('AMD GPU')
    expect(markup).not.toContain('ROCm 0')
    expect(markup).not.toContain('CPU')
    const details = render([models[3], cpu], 'detail')
    expect(details.match(/alt="AMD GPU"/g)).toHaveLength(1)
    expect(details).toContain('ROCm 0')
    expect(details).toContain('CPU')
    expect(render([cpu], 'detail')).not.toContain('amd-logo.png')
    expect(getRouterModelDevice({ ...cpu, metadata: { device: 'migraphx:2' } })).toEqual({
      label: 'MIGraphX 2',
      isAmd: true,
    })
    expect(getRouterModelDevice({ ...cpu, metadata: { device: 'cuda:0' } })).toEqual({
      label: 'CUDA 0',
      isAmd: false,
    })
    expect(getRouterModelDevice({ ...cpu, metadata: {} })).toEqual({
      label: 'Device not reported',
      isAmd: false,
    })
  })

  it('does not guess a Vela identity from arbitrary local paths or unrecognized suffixes', () => {
    const other = { ...models[0], model_path: 'models/custom-export-local-0123456789ab' }
    expect(getRouterModelDisplayName(other)).toBe('custom-export-local-0123456789ab')
    expect(isLocalDerivedRouterModel(other)).toBe(false)
    const unrecognized = { ...other, model_path: 'models/Vela-1.0-Encoder-307M-Domain-custom' }
    expect(getRouterModelDisplayName(unrecognized)).toBe('Vela-1.0-Encoder-307M-Domain-custom')
    expect(isLocalDerivedRouterModel(unrecognized)).toBe(false)
    expect(getRouterModelPreviewName(unrecognized)).toEqual({
      title: 'Vela-1.0-Encoder-307M-Domain-custom',
    })
    expect(headings(render([other]))).toEqual(['custom-export-local-0123456789ab'])
    const resolved = {
      ...models[0],
      resolved_model_path: '/models/Vela-1.0-Encoder-307M-Domain-CK-32768-local-abcdef012345/',
      registry: { local_path: 'models/old-cache' },
    }
    expect(getRouterModelDisplayName(resolved)).toBe('Vela-1.0-Encoder-307M-Domain')
    expect(getRouterModelArtifactPath(resolved)).toBe(resolved.resolved_model_path)
  })

  it('distinguishes registry-only context and unknown runtime metadata', () => {
    const model = { ...models[0], metadata: {}, registry: { max_context_length: 32768 } }
    expect(render([model])).toContain('Registry context: 32,768 tokens')
    expect(render([model], 'detail')).toMatch(/Context Window<\/dt><dd[^>]*>Not reported/)
    for (const value of ['0', '-1', 'Infinity', 'NaN', '32K', '12.5']) {
      expect(
        getRouterModelContext({ ...models[0], metadata: { max_sequence_length: value } }).source,
      ).toBe('unknown')
    }
    expect(render([{ ...models[0], metadata: {} }])).toContain('Context window: Not reported')
  })

  it('searches and sorts the same clean model names shown in cards', () => {
    const sorted = filterAndSortRouterModels(models, '', 'all', 'name')
    expect(sorted.map(getRouterModelDisplayName)).toEqual(
      [...models.map(getRouterModelDisplayName)].sort(),
    )
    expect(
      filterAndSortRouterModels(models, 'Vela-1.0-Encoder-307M-Domain', 'all', 'name'),
    ).toEqual([models[0]])
    expect(filterAndSortRouterModels(models, 'Vela Domain', 'all', 'name')).toEqual([models[0]])
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

  it('shows one card per reported shared resource and retains both consumer views in details', () => {
    const inventory = [...models.slice(0, 3), cache, routing]
    expect(headings(render(inventory))).toHaveLength(4)
    expect(render(inventory)).toContain('Shared by 2 consumers')
    const detail = render([cache, routing], 'detail')
    expect(headings(detail)).toEqual(['Vela-1.0-Encoder-307M-Embedding'])
    for (const value of [
      'Consumers',
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
    expect(cache.metadata?.default_layer).toBe('6')
    expect(routing.metadata?.default_layer).toBe('22')
  })

  it('adjusts binding summary counts and preview limits by actual resource identity', () => {
    const info = {
      models: [cache, routing, ...models.slice(0, 3)],
      summary: { loaded_models: 5, total_models: 5 },
    }
    expect(getLoadedModelCount(info)).toBe(4)
    expect(getTotalKnownModelCount(info)).toBe(4)
    expect(getPreviewRouterModels(info, 4)).toHaveLength(4)
    expect(new Set(getPreviewRouterModels(info, 4).map((model) => model.type)).size).toBe(4)
    expect(getTotalKnownModelCount({ ...info, summary: { total_models: 7 } })).toBe(6)
  })

  it('does not deduplicate by matching model names or paths, or merge different serving resources', () => {
    const cpu = {
      ...routing,
      metadata: { ...routing.metadata, resource_id: 'separate-cpu-resource', device: 'cpu' },
    }
    expect(getRouterModelResources([cache, cpu])).toHaveLength(2)
    const unknown = { ...routing, metadata: { ...routing.metadata, resource_id: '' } }
    expect(getRouterModelResources([unknown, { ...unknown, recipe: 'other' }])).toHaveLength(2)
    expect(getRouterModelConsumers([cache], unknown)).toEqual([unknown])
    expect(render([cache, cpu], 'detail').match(/alt="AMD GPU"/g)).toHaveLength(1)
  })

  it('does not conceal a failed consumer behind a ready consumer or inflate readiness', () => {
    const failed = { ...routing, loaded: false, state: 'not_loaded' }
    const info = { models: [cache, failed], summary: { loaded_models: 1, total_models: 2 } }
    const [resource] = getRouterModelResources(info.models)
    expect(resource.model.loaded).toBe(false)
    expect(resource.model.state).toBe('not_loaded')
    expect(resource.consumers).toEqual([cache, failed])
    expect(getLoadedModelCount(info)).toBe(0)
    expect(getTotalKnownModelCount(info)).toBe(1)
    const detail = render(info.models, 'detail')
    expect(detail).toContain('Not Loaded')
    expect(detail).toContain('Ready')
  })
})
