import { describe, expect, expectTypeOf, it } from 'vitest'
import selector from '../../../../src/semantic-router/pkg/trainingcontract/testdata/selector.json'
import neural from '../../../../src/semantic-router/pkg/trainingcontract/testdata/neural.json'
import type { Fixture, Profile, RunSpec, WorkerResult } from '../generated/trainingContract'

// Python validates these exact JSON documents against the generated schema.
// The Console consumes the generated types; it does not maintain another model.
const fixtures: Fixture[] = [selector as Fixture, neural as Fixture]

function outputLabels(profile: Profile): string[] {
  switch (profile.target_contract) {
    case 'selector.model-choice/v1':
      return profile.selector.candidate_models
    case 'signal.label-scores/v1':
      return Object.keys(profile.classifier.label_mapping)
    case 'signal.spans/v1':
      return profile.spans.labels
  }
}

describe.each(fixtures)('training contract: $asset.name', (fixture) => {
  it('renders typed resources and follows stable artifact references', () => {
    expect(outputLabels(fixture.snapshot.profile)).toHaveLength(2)
    expect(fixture.graph.run.spec).toEqual(fixture.submit.spec)
    expect(fixture.evaluate_request.inputs?.[0].id).toBe(fixture.variant.id)
    expect(fixture.evaluate_result.evaluations?.[0].variant_id).toBe(fixture.variant.id)
    expect(fixture.evaluate_result.artifacts).toBeUndefined()
    expect(fixture.proposal.qualification_id).toBe(fixture.qualification.id)
  })
})

it('keeps trainer parameters open and worker outcomes distinct from run scheduling', () => {
  expectTypeOf<RunSpec['parameters']>().toEqualTypeOf<Record<string, unknown> | undefined>()
  expectTypeOf<WorkerResult['status']>().toEqualTypeOf<
    'running' | 'succeeded' | 'failed' | 'cancelled'
  >()
})
