import { readFileSync } from 'node:fs'
import { describe, expect, expectTypeOf, it } from 'vitest'
import type {
  Artifact, ArtifactVariant, BindingProposalSpec, Evaluation, Fixture,
  Profile, Qualification, RunGraph, RunSpec, WorkerResult,
} from '../generated/trainingContract'

// Python validates these exact JSON documents against the generated schema.
// The Console consumes the generated types; it does not maintain another model.
// Read fixtures at test runtime so production builds do not need test data.
const [selector, neural] = ['selector', 'neural'].map((name): Fixture => JSON.parse(
  readFileSync(new URL(
    `../../../../src/semantic-router/pkg/trainingcontract/testdata/${name}.json`,
    import.meta.url,
  ), 'utf8'),
))
const fixtures: Fixture[] = [selector, neural]

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
  it('discovers downloads and binding evidence starting with only a run ID', () => {
    // Fixtures populate the server; consumers receive only a run ID and API reads.
    const responses = new Map<string, unknown>([
      [`/runs/${fixture.graph.run.id}`, fixture.graph],
      [`/artifacts/${fixture.artifact.id}`, fixture.artifact],
      [`/artifacts/${fixture.artifact.id}/variants`, [fixture.variant]],
      [`/evaluations/${fixture.evaluation.id}`, fixture.evaluation],
      [`/qualifications/${fixture.qualification.id}`, fixture.qualification],
      ...Object.entries(fixture.variant.files).map(([name, file]) =>
        [`/files/${file.handle}`, new TextEncoder().encode(name)] as [string, unknown]),
    ])
    const read = <T,>(path: string): T => {
      expect(responses.has(path)).toBe(true)
      return responses.get(path) as T
    }

    const graph = read<RunGraph>(`/runs/${fixture.graph.run.id}`)
    expect(graph.run.status).toBe('succeeded')
    const variants = graph.outputs.artifact_ids.flatMap((id) => {
      const artifact = read<Artifact>(`/artifacts/${id}`)
      expect(artifact.provenance.run_id).toBe(graph.run.id)
      return read<ArtifactVariant[]>(`/artifacts/${artifact.id}/variants`)
    })
    const files = variants.flatMap((variant) => Object.entries(variant.files).map(
      ([name, file]) => ({ name, bytes: read<Uint8Array>(`/files/${file.handle}`) }),
    ))
    const evaluations = graph.outputs.evaluation_ids.map((id) =>
      read<Evaluation>(`/evaluations/${id}`))
    const proposals: BindingProposalSpec[] = graph.outputs.qualification_ids.map((id) => {
      const qualification = read<Qualification>(`/qualifications/${id}`)
      expect(qualification.compatible).toBe(true)
      expect(variants.some((variant) => variant.id === qualification.variant_id)).toBe(true)
      return {
        name: 'candidate',
        variant_id: qualification.variant_id,
        qualification_id: qualification.id,
      }
    })

    expect(files.map(({ name }) => name)).toEqual(Object.keys(fixture.variant.files))
    for (const file of files) {
      expect(new TextDecoder().decode(file.bytes)).toBe(file.name)
    }
    expect(evaluations).toEqual([fixture.evaluation])
    expect(proposals).toEqual([{
      name: 'candidate',
      variant_id: fixture.proposal.variant_id,
      qualification_id: fixture.proposal.qualification_id,
    }])
  })

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

it('preserves a model layout with multiple files and nested directories', () => {
  expect(Object.keys(neural.variant.files)).toEqual([
    'config.json',
    'tokenizer/tokenizer.json',
    'model-00001-of-00002.safetensors',
    'model-00002-of-00002.safetensors',
  ])
})
