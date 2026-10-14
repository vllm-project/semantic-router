import type { DecisionModelSignal } from '../types/config'

// Question fields follow the canonical DecisionModelSignal authoring contract;
// response fields follow src/model-runtime/vllm_srun/api/openapi.yaml.
export type QuestionType = DecisionModelSignal['question']['type']
export type SystemOneQuestion = DecisionModelSignal['question'] & {
  over?: string
  require_full_input?: boolean
}
export interface QuestionDraft {
  id: string
  name: string
  question: SystemOneQuestion
  variants?: Partial<Record<QuestionType, SystemOneQuestion>>
}
export interface SystemOneRequest {
  state: string | Record<string, unknown> | unknown[]
  questions: Record<string, SystemOneQuestion>
  options: { return_meta: true }
}
export interface SystemOneAnswer {
  type: string | null
  choice?: string
  noul?: number
  score?: number
  probabilities?: Record<string, number>
  confidence?: number
  abstain_probability?: number
  legend?: Record<string, string>
  error?: string
  message?: string
  input_coverage?: 'complete'
}
export interface SystemOneSpan {
  label: string
  start: number
  end: number
  text: string
  probability: number
}
export interface SystemOneRouting {
  recipe: string
  decision: string
  algorithm: string
  stage: string
  selected_model: string
  quality: string
  model_calls: number
}
export interface SystemOneResponse {
  routing?: SystemOneRouting
  model: string
  answers: Record<string, SystemOneAnswer>
  spans?: Record<string, SystemOneSpan[]>
  sets?: Record<
    string,
    { selected: string[]; probabilities: Record<string, number>; input_coverage?: 'complete' }
  >
  thresholds?: Record<string, number>
  span_heads?: Record<string, string>
  usage: { input_tokens: number; output_tokens: number }
  meta?: {
    profile?: string
    engine?: string
    device?: string
    compute_ms?: number
    queue_ms?: number
  }
}

export const QUESTION_TYPES: {
  type: QuestionType
  label: string
  symbol: string
  detail: string
}[] = [
  { type: 'choice', label: 'Choice', symbol: '◉', detail: 'Pick one option from a distribution.' },
  {
    type: 'score',
    label: 'Score',
    symbol: '▥',
    detail: 'Measure an expected level on an ordered scale.',
  },
  {
    type: 'noul',
    label: 'Noul',
    symbol: '◐',
    detail: 'Estimate the probability that a statement is true.',
  },
  { type: 'span', label: 'Span', symbol: '⌁', detail: 'Find and label passages in the input.' },
  {
    type: 'set',
    label: 'Set',
    symbol: '⊞',
    detail: 'Select any number of independently scored labels.',
  },
]

export function newQuestion(type: QuestionType, index = 1): QuestionDraft {
  const names: Record<QuestionType, string> = {
    choice: 'task',
    score: 'difficulty',
    noul: 'needs_reasoning',
    span: 'entities',
    set: 'needs',
  }
  const common = { id: crypto.randomUUID(), name: `${names[type]}${index > 1 ? `_${index}` : ''}` }
  switch (type) {
    case 'choice':
      return {
        ...common,
        question: {
          type,
          instructions: 'What is the main task in this request?',
          choices: [
            { key: 'code', description: 'Programming or debugging' },
            { key: 'writing', description: 'Writing or editing prose' },
            { key: 'other', description: 'Any other task' },
          ],
        },
      }
    case 'score':
      return {
        ...common,
        question: {
          type,
          instructions: 'How difficult is this request to answer correctly?',
          levels: [
            'Simple, one-step answer',
            'Moderate, some reasoning',
            'Complex, multi-step reasoning',
          ],
        },
      }
    case 'noul':
      return {
        ...common,
        question: {
          type,
          instructions: 'Does answering this request require multi-step reasoning?',
        },
      }
    case 'span':
      return {
        ...common,
        question: {
          type,
          instructions: 'Find names of people, organizations, and programming languages.',
          labels: [
            { key: 'person', description: 'The name of a person' },
            { key: 'organization', description: 'The name of a company or organization' },
            { key: 'language', description: 'The programming language' },
          ],
        },
      }
    case 'set':
      return {
        ...common,
        question: {
          type,
          instructions: 'Which capabilities are needed to answer this request?',
          labels: [
            { key: 'coding', description: 'Writing or understanding code' },
            { key: 'reasoning', description: 'Reasoning through multiple steps' },
            { key: 'creative', description: 'Creative writing' },
          ],
        },
      }
  }
}

export interface SystemOneExample {
  id: string
  label: string
  description: string
  state: string
  questions: Record<string, SystemOneQuestion>
}

export function exampleDrafts(example: SystemOneExample): QuestionDraft[] {
  return Object.entries(example.questions).map(([name, question]) => ({
    id: crypto.randomUUID(),
    name,
    question: structuredClone(question),
  }))
}

export const EXAMPLE_STATES: SystemOneExample[] = [
  {
    questions: {
      task: newQuestion('choice').question,
      difficulty: newQuestion('score').question,
      needs_reasoning: newQuestion('noul').question,
    },
    id: 'coding',
    label: 'Code & reasoning',
    description: 'Choice, Score, and Noul · assess a coding request',
    state:
      'Write a Python function that merges two sorted lists without allocating another list. Explain why it works and analyze its time and space complexity.',
  },
  {
    questions: {
      intent: {
        type: 'choice',
        instructions: 'What does the customer need help with?',
        choices: [
          { key: 'billing', description: 'A charge, payment, or refund' },
          { key: 'delivery', description: 'An order shipment or delivery' },
          { key: 'account', description: 'Account access or settings' },
        ],
      },
      needs: {
        type: 'set',
        instructions: 'Which actions does the customer request?',
        labels: [
          { key: 'refund', description: 'Request a refund or billing correction' },
          { key: 'explanation', description: 'Explain a policy or process' },
          { key: 'draft_message', description: 'Draft a message to the support team' },
        ],
      },
    },
    id: 'support',
    label: 'Customer support',
    description: 'Choice and Set · classify a support request',
    state:
      'I was charged twice for order #4821 yesterday. Please explain how to request a refund and help me write a polite message to the support team.',
  },
  {
    questions: { entities: newQuestion('span').question },
    id: 'entities',
    label: 'Entity extraction',
    description: 'Span · find people, organizations, and languages',
    state:
      'Maya Chen joined Northstar Labs in June. She will collaborate with Daniel Rivera at Open Robotics on a new research project written in Python.',
  },
]

export function buildSystemOneRequest(
  source: string,
  format: 'text' | 'json',
  drafts: QuestionDraft[],
): { request: SystemOneRequest | null; error: string | null } {
  const fail = (error: string) => ({ request: null, error })
  if (!source.trim()) return fail('Add the input that you want the model to analyze.')
  let state: SystemOneRequest['state'] = source
  if (format === 'json') {
    try {
      const parsed: unknown = JSON.parse(source)
      if (parsed === null || typeof parsed !== 'object')
        return fail('Structured state must be a JSON object or array.')
      state = parsed as SystemOneRequest['state']
    } catch {
      return fail('The structured state is not valid JSON.')
    }
  }
  if (!drafts.length) return fail('Add at least one question.')
  const entries: [string, SystemOneQuestion][] = []
  const names = new Set<string>()
  for (const draft of drafts) {
    const name = draft.name.trim()
    if (!name) return fail('Give every question a name.')
    if (names.has(name)) return fail(`Question names must be unique: “${name}” is repeated.`)
    names.add(name)
    const question = draft.question
    if (!question.instructions.trim()) return fail(`Add instructions for “${name}”.`)
    if (
      question.threshold !== undefined &&
      (!finiteNumber(question.threshold) || question.threshold < 0 || question.threshold > 1)
    )
      return fail(`Use a threshold from 0 to 1 in “${name}”.`)
    const options =
      question.type === 'choice'
        ? question.choices
        : question.type === 'span' || question.type === 'set'
          ? question.labels
          : undefined
    if (options) {
      const minimum = question.type === 'choice' ? 2 : 1
      if (options.length < minimum || options.length > 255)
        return fail(
          `“${name}” needs ${minimum}–255 ${question.type === 'choice' ? 'options' : 'labels'}.`,
        )
      const keys = options.map((option) => option.key.trim())
      if (keys.some((key) => !key) || new Set(keys).size !== keys.length)
        return fail(`Use unique, nonempty option names in “${name}”.`)
    }
    if (
      question.type === 'score' &&
      (!question.levels ||
        question.levels.length < 2 ||
        question.levels.length > 10 ||
        question.levels.some((level) => !level.trim()))
    )
      return fail(`“${name}” needs 2–10 nonempty score levels, from lowest to highest.`)
    const clean = { ...question, instructions: question.instructions.trim() }
    if (options) {
      const cleaned = options.map((option) => ({
        key: option.key.trim(),
        ...(option.description?.trim() ? { description: option.description.trim() } : {}),
      }))
      if (question.type === 'choice') clean.choices = cleaned
      else clean.labels = cleaned
    }
    if (clean.over?.trim()) clean.over = clean.over.trim()
    else delete clean.over
    entries.push([name, clean])
  }
  for (const [name, question] of entries) {
    if (
      question.type === 'set' &&
      question.labels?.some((label) => names.has(`${name}.${label.key}`))
    )
      return fail(
        `“${name}” generates per-label answers. Rename the question that overlaps with one of its labels.`,
      )
  }
  return {
    request: { state, questions: Object.fromEntries(entries), options: { return_meta: true } },
    error: null,
  }
}

export function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function probabilities(value?: Record<string, number>): [string, number][] {
  return Object.entries(value ?? {}).filter((entry): entry is [string, number] =>
    finiteNumber(entry[1]),
  )
}

export function probabilityLabel(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

// Do not guess the renderer's concatenation of arbitrary typed parts. Highlight
// only an unambiguous source; all native offsets and texts remain in the table.
export function spanSource(
  state: SystemOneRequest['state'],
  question: SystemOneQuestion,
): string | null {
  if (typeof state === 'string') return state
  if (Array.isArray(state)) return null
  if (question.over)
    return typeof state[question.over] === 'string' ? (state[question.over] as string) : null
  const answerKeys = Object.keys(state).filter((key) =>
    ['answer', 'response'].includes(key.toLowerCase()),
  )
  const requestKeys = Object.keys(state).filter((key) =>
    ['request', 'user', 'prompt'].includes(key.toLowerCase()),
  )
  const keys = answerKeys.length ? answerKeys : requestKeys
  return keys.length === 1 && typeof state[keys[0]] === 'string' ? (state[keys[0]] as string) : null
}

export function spanSegments(
  source: string,
  spans: SystemOneSpan[],
): { text: string; labels: string[] }[] {
  const points = Array.from(source)
  const valid = spans.filter(
    (span) =>
      Number.isInteger(span.start) &&
      Number.isInteger(span.end) &&
      span.start >= 0 &&
      span.end > span.start &&
      span.end <= points.length &&
      points.slice(span.start, span.end).join('') === span.text,
  )
  const boundaries = [
    ...new Set([0, points.length, ...valid.flatMap((span) => [span.start, span.end])]),
  ].sort((a, b) => a - b)
  return boundaries.slice(0, -1).map((start, index) => {
    const end = boundaries[index + 1]
    return {
      text: points.slice(start, end).join(''),
      labels: [
        ...new Set(
          valid.filter((span) => span.start <= start && span.end >= end).map((span) => span.label),
        ),
      ],
    }
  })
}

export function systemOneError(value: unknown, fallback: string): string {
  if (typeof value !== 'object' || value === null) return fallback
  const object = value as Record<string, unknown>
  if (typeof object.message === 'string') return object.message
  if (typeof object.error === 'string') return object.error
  if (typeof object.error === 'object' && object.error !== null)
    return systemOneError(object.error, fallback)
  return fallback
}
