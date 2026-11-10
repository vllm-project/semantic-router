import { expect, test } from 'claude-code/testing'

import { effortFor, pickEffort } from './lib/effort'

const LINES = { low: 0.5, high: 1.2, minConfidence: 0.3 }
const sure = (score: number) => ({ score, confidence: 0.8 })

const TABLE = [
  ['haiku', 'medium', 'medium', 'medium'],
  ['sonnet', 'low', 'medium', 'high'],
  ['opus', 'high', 'high', 'xhigh'],
] as const

for (const [model, few, several, many] of TABLE) {
  test(`${model} picks ${few}, ${several} or ${many} by the steps lines`, () => {
    const from = model === 'haiku' ? 'haiku' : 'steps'

    expect(pickEffort(model, sure(0.17), LINES)).toEqual({ effort: few, from })
    expect(pickEffort(model, sure(0.9), LINES)).toEqual({ effort: several, from })
    expect(pickEffort(model, sure(1.61), LINES)).toEqual({ effort: many, from })
  })
}

test('a score on a line takes the rung above it', () => {
  expect(pickEffort('sonnet', sure(0.5), LINES).effort).toBe('medium')
  expect(pickEffort('sonnet', sure(0.4999), LINES).effort).toBe('low')
  expect(pickEffort('sonnet', sure(1.2), LINES).effort).toBe('high')
  expect(pickEffort('opus', sure(1.2), LINES).effort).toBe('xhigh')
})

test('Haiku is at medium whatever steps says or whether it answered', () => {
  expect(pickEffort('haiku', null, LINES)).toEqual({ effort: 'medium', from: 'haiku' })
  expect(pickEffort('haiku', { score: 1.9, confidence: 0 }, LINES)).toEqual({ effort: 'medium', from: 'haiku' })
})

test('no steps answer, or one under the confidence line, is the default of the band', () => {
  expect(pickEffort('sonnet', null, LINES)).toEqual({ effort: 'medium', from: 'unsure' })
  expect(pickEffort('opus', null, LINES)).toEqual({ effort: 'high', from: 'unsure' })
  expect(pickEffort('sonnet', { score: 1.9, confidence: 0.29 }, LINES)).toEqual({ effort: 'medium', from: 'unsure' })
  expect(pickEffort('opus', { score: 1.9, confidence: 0.29 }, LINES)).toEqual({ effort: 'high', from: 'unsure' })
})

test('a confidence on the line is sure enough', () => {
  expect(pickEffort('sonnet', { score: 1.9, confidence: 0.3 }, LINES)).toEqual({ effort: 'high', from: 'steps' })
})

test('the lines and the confidence line move the picks', () => {
  const lines = { low: 1, high: 1.5, minConfidence: 0.9 }

  expect(pickEffort('sonnet', { score: 0.9, confidence: 0.95 }, lines).effort).toBe('low')
  expect(pickEffort('sonnet', { score: 1.2, confidence: 0.95 }, lines).effort).toBe('medium')
  expect(pickEffort('sonnet', { score: 1.6, confidence: 0.8 }, lines)).toEqual({ effort: 'medium', from: 'unsure' })
})

test('a route with a floor keeps a higher effort the subagent would have had, one without takes the pick', () => {
  const high = pickEffort('opus', null, LINES)
  const low = pickEffort('sonnet', sure(0.1), LINES)

  expect(effortFor(high, true, 'max')).toEqual({ effort: 'max', from: 'floor' })
  expect(effortFor(high, true, 'xhigh')).toEqual({ effort: 'xhigh', from: 'floor' })
  expect(effortFor(high, true, 'high')).toEqual({ effort: 'high', from: 'unsure' })
  expect(effortFor(high, true, 'medium')).toEqual({ effort: 'high', from: 'unsure' })
  expect(effortFor(low, false, 'max')).toEqual({ effort: 'low', from: 'steps' })
})
