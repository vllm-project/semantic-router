import { describe, expect, it } from 'vitest'
import catalog from './modelCatalogDocument'
import metadata from './modelCatalogMetadata'
describe('lightweight canonical catalog metadata', () => {
  it('keeps the generated model and provider contract without embedding evaluation records', () => {
    const { evaluations, index_results: indexResults, ...expected } = catalog
    const { evaluations: omittedEvaluations, index_results: omittedResults, ...actual } = metadata
    expect(actual).toEqual(expected)
    expect(evaluations.length).toBeGreaterThan(0)
    expect(indexResults.length).toBeGreaterThan(0)
    expect(omittedEvaluations).toEqual([])
    expect(omittedResults).toEqual([])
    expect(JSON.stringify(metadata).length).toBeLessThan(JSON.stringify(catalog).length / 5)
  })
})
