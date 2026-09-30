import { describe, expect, it } from 'vitest'

import type { ReasoningFamily } from './configPageSupport'
import {
  defaultReasoningEnabled,
  modelSelectionReasoningState,
  reasoningFamilyCanDisable,
  reasoningFamilyForModel,
  reasoningFamilyIsAlwaysOn,
} from './configPageReasoningControlSupport'

describe('reasoning control support', () => {
  it('locks always-on effort families on', () => {
    const family: ReasoningFamily = {
      type: 'reasoning_effort',
      parameter: 'reasoning_effort',
      levels: ['low', 'high'],
      default: 'high',
      modes: ['enabled'],
      default_mode: 'enabled',
    }

    expect(reasoningFamilyCanDisable(family)).toBe(false)
    expect(reasoningFamilyIsAlwaysOn(family)).toBe(true)
    expect(modelSelectionReasoningState(family)).toEqual({
      use_reasoning: true,
      reasoning_mode: '',
      reasoning_effort: '',
    })
  })

  it('honors a model family whose native default is disabled', () => {
    const family: ReasoningFamily = {
      type: 'reasoning_effort',
      parameter: 'reasoning_effort',
      levels: ['low', 'high'],
      default: 'high',
      disabled: 'none',
      modes: ['enabled', 'disabled'],
      default_mode: 'disabled',
    }

    expect(reasoningFamilyCanDisable(family)).toBe(true)
    expect(defaultReasoningEnabled(family)).toBe(false)
  })

  it('keeps an operator family without an explicit mode default off', () => {
    const family: ReasoningFamily = {
      type: 'reasoning_effort',
      parameter: 'reasoning_effort',
      levels: ['low', 'high'],
      default: 'high',
      disabled: 'none',
    }

    expect(defaultReasoningEnabled(family)).toBe(false)
  })

  it('uses provider-specific modes and efforts instead of wider family controls', () => {
    const family: ReasoningFamily = {
      type: 'reasoning_effort',
      parameter: 'reasoning_effort',
      levels: ['low', 'medium', 'high'],
      default: 'medium',
      modes: ['enabled', 'disabled', 'adaptive'],
      default_mode: 'adaptive',
    }

    expect(
      reasoningFamilyForModel(
        {
          name: 'official-model',
          reasoning_family: 'provider-family',
          reasoning_modes: ['disabled', 'adaptive'],
          reasoning_efforts: ['low', 'high'],
          endpoints: [],
        },
        { 'provider-family': family },
      ),
    ).toEqual({
      ...family,
      modes: ['disabled', 'adaptive'],
      levels: ['low', 'high'],
    })
  })
})
