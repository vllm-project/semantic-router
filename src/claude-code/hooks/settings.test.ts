import { expect, test } from 'claude-code/testing'

import { DEFAULTS, parseSettings } from './lib/settings'

test('no options parse to the defaults', () => {
  expect(parseSettings({})).toEqual(DEFAULTS)
  expect(DEFAULTS.thresholds).toEqual({
    router: { easyBelow: 0.5, hardFrom: 1.5, minConfidence: 0.3, downgradeConfidence: 0.6, stepsLow: 0.5, stepsHigh: 1.2, stepsMinConfidence: 0.3 },
    escalate: { feedback: 0.8, reask: 0.75 },
    redact: { pii: 0.7 },
    shield: { inject: 0.9 },
    guard: { review: 0.9 },
    check: { unsupported: 0.5 },
  })
})

test('numbers are held to their ranges, and the hard line never under the easy one', () => {
  const settings = parseSettings({ guardReview: 3, redactPii: -1, routerEasyBelow: 1.2, routerHardFrom: 0.4, calibrateEvery: -5 })

  expect(settings.thresholds.guard.review).toBe(1)
  expect(settings.thresholds.redact.pii).toBe(0)
  expect(settings.thresholds.router).toMatchObject({ easyBelow: 1.2, hardFrom: 1.2 })
  expect(settings.log.calibrateEvery).toBe(0)
})

test('the steps lines are held to 0 to 2, their confidence to 0 to 1, and the high line never under the low one', () => {
  expect(parseSettings({ routerStepsLow: 3, routerStepsHigh: 5, routerStepsMinConfidence: 2 }).thresholds.router).toMatchObject({
    stepsLow: 2,
    stepsHigh: 2,
    stepsMinConfidence: 1,
  })
  expect(parseSettings({ routerStepsLow: -1, routerStepsHigh: -2, routerStepsMinConfidence: -1 }).thresholds.router).toMatchObject({
    stepsLow: 0,
    stepsHigh: 0,
    stepsMinConfidence: 0,
  })
  expect(parseSettings({ routerStepsLow: 1, routerStepsHigh: 0.4 }).thresholds.router).toMatchObject({ stepsLow: 1, stepsHigh: 1 })
  expect(parseSettings({ routerStepsLow: 1.6 }).thresholds.router).toMatchObject({ stepsLow: 1.6, stepsHigh: 1.6 })
  expect(parseSettings({ routerStepsHigh: 0.3 }).thresholds.router).toMatchObject({ stepsLow: 0.5, stepsHigh: 0.5 })
  expect(parseSettings({ routerStepsLow: 0.2, routerStepsHigh: 0.9 }).thresholds.router).toMatchObject({ stepsLow: 0.2, stepsHigh: 0.9 })
})

test('a steps line of the wrong kind takes its default', () => {
  const { router } = parseSettings({ routerStepsLow: 'few', routerStepsHigh: null, routerStepsMinConfidence: Number.NaN }).thresholds

  expect(router).toMatchObject({ stepsLow: 0.5, stepsHigh: 1.2, stepsMinConfidence: 0.3 })
})

test('a value of the wrong kind takes the default', () => {
  const settings = parseSettings({ runtime: 'mainframe', runtimeUrl: '  ', guard: 'yes', shieldInject: 'high', runtimeToken: ' tok ' })

  expect(settings.runtime).toEqual({ kind: 'local-cpu', url: 'http://127.0.0.1:8790/v1/decisions', token: 'tok' })
  expect(settings.apps.guard).toBe(true)
  expect(settings.thresholds.shield.inject).toBe(0.9)
})
