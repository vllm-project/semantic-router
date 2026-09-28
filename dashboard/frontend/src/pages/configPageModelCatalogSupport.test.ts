import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  catalogSnapshotsForEntrypoint,
  effectiveModelAPIFormat,
  modelAPIFormats,
  modelsForCatalogVersion,
  preferredCatalogModelForEntrypoint,
} from './configPageModelCatalogSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog

describe('config page model catalog support', () => {
  it('projects the one coherent resource graph for the active catalog header', () => {
    expect(modelsForCatalogVersion(catalog, catalog.catalogs[0])).toEqual(catalog.models)
    expect(
      modelsForCatalogVersion(catalog, {
        ...catalog.catalogs[0],
        catalog_version: 'stale',
      }),
    ).toEqual([])
  })

  it('resolves entrypoints without duplicating model resources per release header', () => {
    const entrypoint = { model_names: ['vllm-sr/mom-v1-blend'], recipe: 'balance' }

    expect(catalogSnapshotsForEntrypoint(catalog, entrypoint).map((item) => item.id)).toEqual([
      'vllm-sr/mom-v1-blend',
    ])
    expect(preferredCatalogModelForEntrypoint(catalog, entrypoint)?.id).toBe('vllm-sr/mom-v1-blend')
  })

  it('does not mislabel a custom entrypoint as a verified built-in model', () => {
    expect(
      preferredCatalogModelForEntrypoint(catalog, {
        model_names: ['team/custom-mom'],
        recipe: 'custom',
      }),
    ).toBeNull()
  })
})

describe('effective model API format', () => {
  const chat = 'openai/chat-completions@1'
  const responses = 'openai/responses@1'
  const messages = 'anthropic/messages@1'
  const modelID = catalog.models.find((model) => model.kind === 'physical')!.id
  const provider = catalog.providers.find((entry) => entry.id === 'vllm')!
  const fixture = (protocols: string[], defaultProtocol = chat) => ({
    ...catalog,
    providers: [
      {
        ...provider,
        id: 'test',
        protocols: [chat, responses, messages],
        supported_operations: [chat, responses, messages].map((protocol) => `${protocol}#create`),
        default_protocol: defaultProtocol,
        models: [{ ...provider.models![0], catalog: modelID, id: 'native', protocols }],
      },
    ],
  })
  const model = { name: 'alias', catalog: modelID, backend_refs: [{ provider: 'test' }] }

  it('exposes only formats accepted by canonical provider input', () => {
    expect(modelAPIFormats).toEqual(['openai', 'responses', 'anthropic'])
  })

  it('uses the single model protocol ahead of the provider default', () => {
    expect(effectiveModelAPIFormat(model, fixture([responses]))).toEqual({ format: 'responses' })
    expect(effectiveModelAPIFormat(model, fixture([messages]))).toEqual({ format: 'anthropic' })
  })

  it('uses a provider default only when a multiple-protocol binding includes it', () => {
    expect(effectiveModelAPIFormat(model, fixture([chat, responses]))).toEqual({ format: 'openai' })
    expect(effectiveModelAPIFormat(model, fixture([messages, responses])).error).toMatch(
      /ambiguous/,
    )
  })

  it('honors explicit formats and native ID/deployment-name bindings', () => {
    expect(
      effectiveModelAPIFormat({ ...model, api_format: 'responses' }, fixture([chat, responses])),
    ).toEqual({ format: 'responses' })
    const bound = fixture([messages])
    bound.providers[0].models[0].restrictions = { provider_model_id_kind: 'deployment_name' }
    expect(effectiveModelAPIFormat({ ...model, provider_model_id: 'production' }, bound)).toEqual({
      format: 'anthropic',
    })
    expect(
      effectiveModelAPIFormat(
        { ...model, external_model_ids: { test: 'native' } },
        fixture([responses]),
      ),
    ).toEqual({ format: 'responses' })
    expect(
      effectiveModelAPIFormat({ ...model, provider_model_id: 'other' }, fixture([responses])),
    ).toEqual({ format: 'openai' })
  })

  it('does not guess formats for unresolved models or unsupported operations', () => {
    expect(effectiveModelAPIFormat(model, fixture([])).error).toMatch(/provider model ID/)
    expect(effectiveModelAPIFormat({ ...model, backend_refs: [] }, catalog).error).toMatch(
      /backend/,
    )
    expect(effectiveModelAPIFormat({ ...model, api_format: 'images' }, catalog).error).toMatch(
      /Unsupported/,
    )
    const noCreate = fixture([chat])
    noCreate.providers[0].supported_operations = []
    expect(effectiveModelAPIFormat(model, noCreate).error).toMatch(/cannot create/)
  })

  it('allows a virtual catalog card to inherit a provider default without a physical model ID', () => {
    const virtual = catalog.models.find((entry) => entry.kind === 'virtual')!
    expect(virtual).toBeDefined()
    expect(effectiveModelAPIFormat({ ...model, catalog: virtual.id }, fixture([]))).toEqual({
      format: 'openai',
    })
  })

  it('matches the compiler selection order and rejects conflicting backends', () => {
    const combined = fixture([responses])
    combined.providers.push({
      ...combined.providers[0],
      id: 'second',
      models: [{ ...combined.providers[0].models[0], protocols: [chat, responses] }],
    })
    const multi = { ...model, backend_refs: [{ provider: 'test' }, { provider: 'second' }] }
    expect(effectiveModelAPIFormat(multi, combined)).toEqual({ format: 'responses' })
    combined.providers[1].models[0].protocols = [chat]
    expect(effectiveModelAPIFormat(multi, combined).error).toMatch(/same API format/)
  })
})
