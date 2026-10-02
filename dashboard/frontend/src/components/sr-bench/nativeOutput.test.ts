import { describe, expect, it } from 'vitest'
import { DEFAULT_LIMITS, makeManifest, validateManifest } from './model'
import { nativeOutputIssue } from './nativeOutput'
import type { Target } from './types'

const target: Target = {
  id: 'single',
  kind: 'single',
  model: 'physical-model',
  base_url: 'http://localhost:8000/v1',
  native_limits: { 'physical-model': { context_window: 131072, max_output_tokens: 65536 } },
}

describe('native output registry capability', () => {
  it('requires the actual single-model identity and valid registered limits', () => {
    expect(nativeOutputIssue(target)).toBeNull()
    expect(nativeOutputIssue({ ...target, native_limits: undefined })).toContain('not registered')
    expect(nativeOutputIssue({ ...target, expected_response_model: 'another-model' })).toContain(
      'identify',
    )
    expect(
      nativeOutputIssue({
        ...target,
        native_limits: { 'physical-model': { context_window: 1, max_output_tokens: 2 } },
      }),
    ).toContain('invalid')
    expect(nativeOutputIssue({ ...target, request_params: { max_tokens: 100 } })).toContain(
      'fixed output cap',
    )
  })

  it('requires captured single-dispatch MoM without inventing its model set', () => {
    const mom: Target = {
      ...target,
      kind: 'mom',
      model: 'recipe',
      capture_recipe: true,
      max_inference_calls: 1,
    }
    expect(nativeOutputIssue(mom)).toBeNull()
    expect(nativeOutputIssue({ ...mom, capture_recipe: false })).toContain('recipe capture')
    expect(nativeOutputIssue({ ...mom, max_inference_calls: 2 })).toContain('one dispatch')
  })

  it('accepts omitted native caps and the frozen server ceiling without reintroducing a request cap', () => {
    const manifest = makeManifest(
      'Native',
      'live',
      'quick',
      { id: 'data', path: 'data.jsonl', sha256: 'a'.repeat(64), case_count: 1 },
      [target],
      DEFAULT_LIMITS,
    )
    manifest.output_policy = 'native'
    delete manifest.sampling.max_tokens
    delete manifest.limits.max_output_tokens
    expect(validateManifest(manifest)).toBeNull()
    manifest.limits.max_output_tokens = 65536
    expect(validateManifest(manifest)).toBeNull()
    expect(manifest.sampling).not.toHaveProperty('max_tokens')
    expect(
      validateManifest({ ...manifest, sampling: { ...manifest.sampling, max_tokens: 4096 } }),
    ).toContain('does not use')
  })
})
