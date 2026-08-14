import { describe, expect, it } from 'vitest'

import type { BuiltInModelCatalog, BuiltInModelMetadata } from '../types/modelCatalog'
import {
  catalogSnapshotsForEntrypoint,
  modelsForCatalogVersion,
  preferredCatalogModelForEntrypoint,
} from './configPageModelCatalogSupport'

const model = (
  catalogVersion: string,
  channel: 'latest' | 'release',
  id = 'vllm-sr/chorus-v1',
): BuiltInModelMetadata =>
  ({
    id,
    entrypoint: id,
    display_name: id,
    description: 'test',
    kind: 'virtual',
    family: 'chorus',
    generation: 1,
    policy_version: '1.0.0',
    recipe: 'balance',
    protocols: ['openai_chat'],
    traits: ['balanced'],
    roles: [],
    catalog_version: catalogVersion,
    channel,
    compatible: true,
    compatibility_reason: 'compatible',
    enabled_by_default: true,
    default: true,
    verification: {
      status: 'verified',
      authority: 'vllm-sr-maintainers',
      asset_sha256: `sha256:${'a'.repeat(64)}`,
    },
  }) as BuiltInModelMetadata

const catalog: BuiltInModelCatalog = {
  catalogs: [
    {
      catalog_version: 'latest',
      channel: 'latest',
      default_model: 'vllm-sr/chorus-v1',
      enabled_models: ['vllm-sr/chorus-v1'],
    },
    {
      catalog_version: 'v0.4',
      channel: 'release',
      default_model: 'vllm-sr/chorus-v1',
      enabled_models: ['vllm-sr/chorus-v1'],
    },
  ],
  models: [
    model('v0.4', 'release'),
    model('latest', 'latest'),
    model('latest', 'latest', 'vllm-sr/other'),
  ],
}

describe('config page model catalog support', () => {
  it('groups models by both version and immutable/moving channel', () => {
    expect(modelsForCatalogVersion(catalog, catalog.catalogs[0]).map((item) => item.id)).toEqual([
      'vllm-sr/chorus-v1',
      'vllm-sr/other',
    ])
    expect(modelsForCatalogVersion(catalog, catalog.catalogs[1])).toHaveLength(1)
  })

  it('prefers latest metadata while retaining release snapshots', () => {
    const entrypoint = { model_names: ['vllm-sr/chorus-v1'], recipe: 'balance' }

    expect(catalogSnapshotsForEntrypoint(catalog, entrypoint).map((item) => item.channel)).toEqual([
      'latest',
      'release',
    ])
    expect(preferredCatalogModelForEntrypoint(catalog, entrypoint)?.catalog_version).toBe('latest')
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
