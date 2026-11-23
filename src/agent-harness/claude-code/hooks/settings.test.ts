import { expect, test } from 'claude-code/testing'

import { appToggles, engine } from './kit'
import { parseSettings } from './lib/settings'

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
  expect(parseSettings({ routerStepsLow: 0.2, routerStepsHigh: 0.9 }).thresholds.router).toMatchObject({ stepsLow: 0.2, stepsHigh: 0.9 })
})

test('equal easy and hard lines are allowed, as the setup script allows them', () => {
  expect(parseSettings({ routerEasyBelow: 1, routerHardFrom: 1 }).thresholds.router).toMatchObject({ easyBelow: 1, hardFrom: 1 })
})

test('the runtime URL, token and model are trimmed, and no model is the empty string', () => {
  expect(parseSettings({ runtime: 'remote', runtimeUrl: ' https://sr.example/v1/decisions ', runtimeToken: ' tok ', runtimeModel: ' vllm-sr/vela3-flash ' }).runtime).toEqual({
    kind: 'remote',
    url: 'https://sr.example/v1/decisions',
    token: 'tok',
    model: 'vllm-sr/vela3-flash',
  })
  expect(parseSettings({ runtime: 'remote', runtimeUrl: 'https://sr.example/v1/decisions', runtimeToken: '' }).runtime.model).toBe('')
})

test('the manifest defaults reach the hooks: check is off and the others are on', { options: {} }, async ($, on) => {
  engine(on, {})

  expect(await appToggles($)).toEqual(['● guard', '○ check', '● shield', '● redact', '● router', '● escalate'])
})
