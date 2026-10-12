import { describe, expect, it } from 'vitest'
import { systemOneDeploymentOption } from './systemOneDeploymentPresentation'

const deployment = {
  id: 'primary',
  model: 'primary',
  ready: true,
  question_types: ['choice'],
  surfaces: ['decisions'],
}

describe('native runtime target presentation', () => {
  it('uses runtime model identity without changing the request deployment', () => {
    expect(systemOneDeploymentOption({ ...deployment, repo: 'vllm-sr/Vela-2.0-4B' })).toEqual({
      value: 'primary',
      label: 'vllm-sr/Vela-2.0-4B',
      description: 'Deployment: primary · Ready · 1 question types',
    })
  })
  it('keeps an unavailable runtime identifiable without inventing an artifact', () => {
    expect(systemOneDeploymentOption({ ...deployment, repo: ' ', ready: false })).toEqual({
      value: 'primary',
      label: 'primary',
      description: 'Unavailable · 1 question types',
    })
  })
})
