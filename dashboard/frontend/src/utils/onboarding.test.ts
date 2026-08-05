import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ONBOARDING_STATUS_KEY,
  ONBOARDING_STEP_KEY,
  clearOnboardingStep,
  getOnboardingStatus,
  getOnboardingStep,
  markOnboardingPending,
  setOnboardingStatus,
  setOnboardingStep,
} from './onboarding'

function installLocalStorage() {
  const values = new Map<string, string>()
  vi.stubGlobal('window', {
    localStorage: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      removeItem: (key: string) => values.delete(key),
    },
  })
  return values
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('onboarding persistence', () => {
  it('stores and clamps the resumable step', () => {
    const values = installLocalStorage()

    setOnboardingStep(12.9)
    expect(values.get(ONBOARDING_STEP_KEY)).toBe('12')
    expect(getOnboardingStep(5)).toBe(4)

    clearOnboardingStep()
    expect(getOnboardingStep()).toBe(0)
  })

  it('starts a fresh pending guide at the first step', () => {
    const values = installLocalStorage()
    setOnboardingStep(3)

    markOnboardingPending()

    expect(getOnboardingStatus()).toBe('pending')
    expect(getOnboardingStep()).toBe(0)
    expect(values.get(ONBOARDING_STATUS_KEY)).toBe('pending')
  })

  it('clears status and step when reset to idle', () => {
    const values = installLocalStorage()
    setOnboardingStatus('dismissed')
    setOnboardingStep(2)

    setOnboardingStatus('idle')

    expect(values.has(ONBOARDING_STATUS_KEY)).toBe(false)
    expect(values.has(ONBOARDING_STEP_KEY)).toBe(false)
  })
})
