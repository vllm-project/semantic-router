import { describe, expect, it } from 'vitest'
import {
  buildSystemOneRequest,
  EXAMPLE_STATES,
  newQuestion,
  spanSegments,
  spanSource,
  systemOneError,
} from './systemOnePlayground'

describe('System One request authoring', () => {
  it('builds native ordered questions without changing state or adding synthetic answers', () => {
    const drafts = ['choice', 'score', 'noul', 'span', 'set'].map((type) =>
      newQuestion(type as 'choice' | 'score' | 'noul' | 'span' | 'set'),
    )
    const result = buildSystemOneRequest('{"request":"hello","answer":"world"}', 'json', drafts)
    expect(result.error).toBeNull()
    expect(result.request?.state).toEqual({ request: 'hello', answer: 'world' })
    expect(Object.keys(result.request?.questions ?? {})).toEqual([
      'task',
      'difficulty',
      'needs_reasoning',
      'entities',
      'needs',
    ])
    expect(result.request?.questions.needs_reasoning).toEqual({
      type: 'noul',
      instructions: 'Does answering this request require multi-step reasoning?',
    })
    expect(result.request?.questions.difficulty.levels).toHaveLength(3)
    expect(result.request?.questions.needs.labels).toHaveLength(3)
    expect(result.request?.questions.entities.labels).toContainEqual({
      key: 'language',
      description: 'The programming language',
    })
    expect(EXAMPLE_STATES.find((example) => example.id === 'entities')?.state).toContain('Python')
    expect(result.request?.options).toEqual({ return_meta: true })
  })

  it('rejects invalid JSON, duplicate keys and Set-generated answer collisions before inference', () => {
    const choice = newQuestion('choice')
    expect(buildSystemOneRequest('{', 'json', [choice]).error).toMatch(/valid JSON/)
    expect(buildSystemOneRequest('true', 'json', [choice]).error).toMatch(/object or array/)
    expect(
      buildSystemOneRequest('hello', 'text', [choice, { ...choice, id: 'another' }]).error,
    ).toMatch(/unique/)
    const set = newQuestion('set')
    choice.name = `${set.name}.coding`
    expect(buildSystemOneRequest('hello', 'text', [set, choice]).error).toMatch(/overlaps/)
  })

  it('rejects ambiguous option keys, blank score anchors and nonfinite thresholds', () => {
    const choice = newQuestion('choice')
    choice.question.choices = [{ key: 'code' }, { key: ' code ' }]
    expect(buildSystemOneRequest('hello', 'text', [choice]).error).toMatch(/unique/)
    const score = newQuestion('score')
    score.question.levels = ['low', ' ']
    expect(buildSystemOneRequest('hello', 'text', [score]).error).toMatch(/nonempty score/)
    const span = newQuestion('span')
    span.question.threshold = NaN
    expect(buildSystemOneRequest('hello', 'text', [span]).error).toMatch(/threshold/)
  })

  it('preserves special question keys as data without prototype mutation', () => {
    const question = newQuestion('choice')
    question.name = '__proto__'
    const result = buildSystemOneRequest('hello', 'text', [question])
    expect(Object.prototype.hasOwnProperty.call(result.request?.questions, '__proto__')).toBe(true)
    expect(Object.getPrototypeOf(result.request?.questions)).toBe(Object.prototype)
  })

  it('preserves task templates requiring complete input when questions are edited', () => {
    const question = newQuestion('noul')
    question.name = 'pii_presence'
    question.question.require_full_input = true
    question.question.instructions = '  Does the text contain personal information?  '
    const result = buildSystemOneRequest('Alex', 'text', [question])
    expect(result.error).toBeNull()
    expect(result.request?.questions.pii_presence).toMatchObject({
      require_full_input: true,
      instructions: 'Does the text contain personal information?',
    })
  })
})

describe('native span offsets', () => {
  it('uses Unicode code points rather than UTF-16 indices and represents overlapping labels', () => {
    const result = spanSegments('Hi 👋 Maya Chen!', [
      { label: 'person', start: 5, end: 14, text: 'Maya Chen', probability: 0.9 },
      { label: 'first_name', start: 5, end: 9, text: 'Maya', probability: 0.8 },
    ])
    expect(result).toEqual([
      { text: 'Hi 👋 ', labels: [] },
      { text: 'Maya', labels: ['person', 'first_name'] },
      { text: ' Chen', labels: ['person'] },
      { text: '!', labels: [] },
    ])
    expect(result.map((part) => part.text).join('')).toBe('Hi 👋 Maya Chen!')
  })

  it('never highlights an invalid offset or a span whose text does not match its source', () => {
    expect(
      spanSegments('Maya', [
        { label: 'bad', start: -1, end: 4, text: 'Maya', probability: 1 },
        { label: 'bad', start: 0, end: 4, text: 'Other', probability: 1 },
        { label: 'bad', start: 0, end: 9, text: 'Maya', probability: 1 },
      ]),
    ).toEqual([{ text: 'Maya', labels: [] }])
  })

  it('honors explicit fields and declines to guess concatenated typed-part offsets', () => {
    const question = newQuestion('span').question
    expect(spanSource({ request: 'question', answer: 'reply' }, question)).toBe('reply')
    expect(
      spanSource({ request: 'question', answer: 'reply' }, { ...question, over: 'request' }),
    ).toBe('question')
    expect(spanSource({ answer: 'first', response: 'second' }, question)).toBeNull()
    expect(spanSource({ context: { data: true } }, { ...question, over: 'context' })).toBeNull()
  })
})

it('reads canonical nested runtime errors without exposing raw non-JSON response bodies', () => {
  expect(
    systemOneError({ error: { code: 'unavailable', message: 'Model is not ready.' } }, 'Fallback'),
  ).toBe('Model is not ready.')
  expect(systemOneError('<html>upstream details</html>', 'HTTP 503')).toBe('HTTP 503')
})
