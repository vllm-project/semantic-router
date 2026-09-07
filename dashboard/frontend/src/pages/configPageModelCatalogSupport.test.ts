import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  catalogSnapshotsForEntrypoint,
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
    expect(preferredCatalogModelForEntrypoint(catalog, entrypoint)?.id).toBe(
      'vllm-sr/mom-v1-blend',
    )
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
