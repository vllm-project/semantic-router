import { mock, type test } from 'claude-code/testing'
import type { On } from 'claude-code'

import type { Segment } from './lib/meters'
import type { Span } from './lib/redact'

export const PLUGIN = 'vsr'
export const PANE_ID = 'vsr-decisions'
export const PANE_TITLE = 'vSR · decisions'
export const HOME = '/home/u'
export const LOG_FILE = `${HOME}/.claude/vsr/decisions.jsonl`
export const SESSION = 'session-1'

export const PANE = {
  plugin: PLUGIN,
  component: 'Pane',
  requestId: PANE_ID,
  props: {
    title: PANE_TITLE,
    isFocused: false,
    bodyColumns: 60,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 20 },
    view: {},
  },
  // Tall enough for the logo, the tally, two sections and the setup under them.
  viewport: { columns: 60, rows: 30 },
} as const

export const CORE = { plugin: 'engine', tier: 'core' } as const
export const BOTTOM = { decision: 'allow', reason: 'the mode allowed it' } as const
export const ENGINE_ROW = 'the engine row'

export type { Span }

// What the stub classifier answers, by the question a request asks.
export type Reply = {
  complexity?: { score: number; confidence: number }
  // The `steps` answer riding in the same reply; left out of the reply when unset or 'fails'.
  steps?: { score: number; confidence: number } | 'fails'
  // The exact `answers` of a complexity request, for an answer the shorthands above cannot shape.
  answers?: Record<string, unknown>
  review?: number
  // A review score, or 'fails', for the request whose Command is exactly this part.
  parts?: Record<string, number | 'fails'>
  // `user_feedback` probabilities, by choice.
  feedback?: Record<string, number>
  reask?: number
  pii?: Span[] | 'fails'
  jailbreak?: number | 'fails'
  hallucination?: number | 'fails'
  // Collects the state of every guard request.
  seen?: string[]
  // Collects every request body, parsed.
  bodies?: { state: unknown; questions: Record<string, unknown> }[]
  // Collects the id of every pane the plugin opens.
  opened?: string[]
  // Collects the context of every prompt that entered.
  contexts?: (readonly string[] | undefined)[]
  fails?: boolean
  // Every decision request is answered HTTP 500: an error, not an unreachable runtime.
  errors?: boolean
  hangs?: boolean
  // The output requests (pii or jailbreak) from this one on never answer, counted from 0.
  hangFrom?: number
  // How many output requests were made.
  outputAsks?: number
  // The runtime's `/health` does not answer.
  offline?: boolean
  bottom?: { decision: 'allow' | 'ask' | 'deny'; reason?: string }
  // Collects the headers of every request to the runtime.
  headers?: (Record<string, string> | undefined)[]
  // Collects the URL of every request.
  urls?: string[]
  // Collects every toast's text, and every status line entry.
  toasts?: string[]
  statuses?: (string | undefined)[]
  // What `$.fs.write` last wrote, by path.
  files?: Record<string, string>
  // What `$.store` holds at the start, and the environment.
  store?: Record<string, unknown>
  env?: Record<string, string>
}

const REFUSED = { deny: 'connection refused' } as const

const answered = (body: object) => ({ value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } })

const choiceOf = (probabilities: Record<string, number>) => {
  const [choice, confidence] = Object.entries(probabilities).reduce((top, one) => (one[1] > top[1] ? one : top))
  return { type: 'choice', choice, probabilities, confidence }
}

const NO_FEEDBACK = { SAT: 0.02, NEED_CLARIFICATION: 0.02, WRONG_ANSWER: 0.02, WANT_DIFFERENT: 0.02, NO_FEEDBACK: 0.92 }

// A span where its text first stands in the state the runtime was asked about, in code points as the runtime counts; none when it is not there.
function locate(span: Span, state: string): Span[] {
  const at = state.indexOf(span.text)
  if (at < 0) return []
  const start = Array.from(state.slice(0, at)).length
  return [{ ...span, start, end: start + Array.from(span.text).length }]
}

function answer(reply: Reply, state: unknown, questions: Record<string, unknown>) {
  if (questions.review) {
    const forPart = reply.parts?.[String(state).replace(/^Tool: Bash\nCommand: /, '')]
    if (forPart === 'fails') return REFUSED
    return answered({
      answers: {
        kind: { type: 'choice', choice: 'destructive', probabilities: { safe: 0.01, destructive: 0.84, exfiltration: 0.14, remote_code: 0.01 }, confidence: 0.78 },
        review: { type: 'noul', noul: forPart ?? reply.review },
      },
    })
  }
  if (questions.user_feedback) return answered({ answers: { user_feedback: choiceOf(reply.feedback ?? NO_FEEDBACK) } })
  if (questions.reask) return answered({ answers: { reask: { type: 'noul', noul: reply.reask ?? 0.1 } } })
  if (questions.pii) {
    if (reply.pii === 'fails') return REFUSED
    const spans = (reply.pii ?? []).flatMap(span => locate(span, String(state)))
    return answered({ answers: { pii: { type: 'noul', noul: spans.length > 0 ? 0.99 : 0.01 } }, spans: { pii: spans }, thresholds: { pii: 0.7 } })
  }
  if (questions.jailbreak) {
    if (reply.jailbreak === 'fails') return REFUSED
    const score = reply.jailbreak ?? 0.02
    return answered({ answers: { jailbreak: choiceOf({ benign: 1 - score, jailbreak: score }) } })
  }
  if (questions.hallucination) {
    if (reply.hallucination === 'fails') return REFUSED
    return answered({ answers: { hallucination: { type: 'noul', noul: reply.hallucination ?? 0.05 } } })
  }
  if (reply.answers) return answered({ answers: reply.answers })
  const steps = reply.steps === undefined || reply.steps === 'fails' ? {} : { steps: { type: 'score', ...reply.steps } }
  return answered({ answers: { complexity: { type: 'score', ...reply.complexity }, ...steps } })
}

// The engine beneath the plugin: the classifier at `$.http.fetch`, and a bottom for every event the plugin raises or continues.
export function engine(on: On, reply: Reply) {
  const clock = mock.clock(on)
  mock.store(on, reply.store)
  mock.env(on, reply.env ?? { HOME })
  on('session.id', () => ({ value: SESSION }))
  on('fs.write', (_, e) => {
    if (reply.files) reply.files[e.path] = e.text
    return { value: undefined }
  })
  // A file exists once the plugin wrote it, as on a disk that starts empty.
  on('fs.exists', (_, e) => ({ value: reply.files !== undefined && e.path in reply.files }))
  on('ui.toast', (_, e) => {
    reply.toasts?.push(e.text)
    return { value: undefined }
  })
  on('command.register', () => ({ value: { command: 'vsr' } }))
  on('ui.status', (_, e) => {
    reply.statuses?.push(e.text)
    return { value: undefined }
  })
  on('ui.open', (_, e) => {
    reply.opened?.push(e.id)
    return { value: { isPlaced: true } }
  })
  on('ui.render', ($, e) => {
    const { Text } = $.ui.resolve(e)
    return <Text>{ENGINE_ROW}</Text>
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.cwd', () => ({ value: '/repo' }))
  on('prompt.submit', (_, e) => {
    reply.contexts?.push(e.context)
    return { text: e.text, context: e.context }
  })
  on('turn.start', (_, e) => ({ turnId: e.turnId }))
  on('turn.complete', (_, e) => ({ text: e.answer }))
  on('tool.check', () => reply.bottom ?? BOTTOM)
  on('http.fetch', (_, e) => {
    reply.urls?.push(e.url)
    reply.headers?.push(e.init?.headers)
    if (e.url.endsWith('/health')) return reply.offline || reply.fails ? REFUSED : answered({ status: 'ok' })
    if (reply.hangs) return new Promise(() => {})
    if (reply.fails) return REFUSED
    if (reply.errors) return { value: { status: 500, ok: false, headers: {}, text: '' } }
    const body = JSON.parse(e.init?.body ?? '{}')
    if (body.questions?.pii || body.questions?.jailbreak) {
      const asked = reply.outputAsks ?? 0
      reply.outputAsks = asked + 1
      if (reply.hangFrom !== undefined && asked >= reply.hangFrom) return new Promise(() => {})
    }
    reply.bodies?.push(body)
    if (body.questions?.review) reply.seen?.push(body.state)
    return answer(reply, body.state, body.questions ?? {})
  })
  return clock
}

export const SPAWN = {
  tool_use_id: 'call-1',
  prompt: 'Rename a variable in one file.',
  description: 'Rename a variable',
  subagentType: 'general-purpose',
  provider: CORE,
  parentModel: 'opus',
  background: true,
  fork: false,
} as const

export type Test = Parameters<Parameters<typeof test>[1]>[0]

export async function paneText($: Test, find: RegExp, bodyColumns = PANE.props.bodyColumns) {
  const ui = await $.ui.mount({ ...PANE, props: { ...PANE.props, bodyColumns }, surface: 'terminal' })
  const row = await ui.find({ type: 'Text', text: find })
  await ui.unmount()
  return row?.text
}

export async function paneLines($: Test, viewport = { columns: 60, rows: 40 }) {
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal', viewport })
  const lines = (await ui.findAll({ type: 'Text' })).map(row => row.text ?? '')
  await ui.unmount()
  return lines
}

// The app toggles at the top of the pane, as they read: `● guard` on, `○ check` off.
export async function appToggles($: Test) {
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  const shown = (await ui.findAll({ type: 'Button' })).filter(one => one.key?.startsWith('toggle:')).map(one => one.text)
  await ui.unmount()
  return shown
}

// Whether a pane Button is one of the controls at its top.
export const isControl = (key: string | undefined) => key !== undefined && /^(toggle|wait):/.test(key)

const CALL = { isRunning: false, isErrored: false, isInterrupted: false } as const
export const callOf = (id: string, command: string) => ({ tool_use_id: id, tool: 'Bash', input: { command }, ...CALL })
export const readCallOf = (id: string, file_path: string) => ({ tool_use_id: id, tool: 'Read', input: { file_path }, ...CALL })

type Props = Record<string, unknown>

export async function rowTree($: Test, target: object) {
  const ui = await $.ui.mount({ plugin: PLUGIN, surface: 'terminal', viewport: { columns: 100, rows: 30 }, ...target } as never)
  const texts = (await ui.findAll({ type: 'Text' })).map(row => row.text ?? '')
  const props = (await ui.findAll({ type: 'Text' })).map(row => row.props as Props)
  const boxes = (await ui.findAll({ type: 'Box' })).map(row => row.props as Props).filter(box => box.borderStyle !== undefined)
  await ui.unmount()
  // A card has a bright border; a frame, the dim one around the one-line form.
  return { texts, props, cards: boxes.filter(box => !box.borderDimColor), frames: boxes.filter(box => box.borderDimColor) }
}

export const rowTexts = async ($: Test, target: object) => (await rowTree($, target)).texts

export const toolRow = (props: { tool_use_id: string }, columns = 100) =>
  ({ component: 'ToolUse', requestId: props.tool_use_id, props, viewport: { columns, rows: 30 } }) as const

export const agentRow = (id: string, columns = 100) =>
  toolRow({ tool_use_id: id, tool: 'Agent', input: { description: 'Rename a variable' }, ...CALL }, columns)

export const groupRow = (calls: { tool_use_id: string }[], isExpanded: boolean) =>
  ({ component: 'ToolGroup', requestId: 'group-1', props: { calls, isActive: false, isExpanded } }) as const

export const userRow = (text: string) =>
  ({ component: 'UserMessage', requestId: 'message-1', props: { text, origin: { kind: 'composer' }, isExpanded: false } }) as const

export const assistantRow = (text: string, isFirstOfReply = false) =>
  ({ component: 'AssistantMessage', requestId: 'message-2', props: { text, isFirstOfReply } }) as const

export const bandRow = (bodyColumns = 100, maxRows = 10) =>
  ({
    component: 'AbovePrompt',
    requestId: 'band',
    props: { hasSurvey: false, isWorking: false, maxRows, bodyColumns, scroll: { offset: 0, bodyRows: maxRows }, view: {} },
  }) as const

export const vsrLines = (lines: string[]) => lines.filter(text => text.startsWith(' vSR   '))

// The decision log as last written, one parsed line each.
export const logLines = (reply: Reply) =>
  (reply.files?.[LOG_FILE] ?? '').split('\n').filter(line => line !== '').map(line => JSON.parse(line))

export const plain = (segments: Segment[]) => segments.map(one => one.text).join('')

export const PUSH = { tool: 'Bash', command: 'git push --force origin main', tool_use_id: 'call-9' } as const
export const YES = { decision: { behavior: 'allow' } } as const
export const NO = { decision: { behavior: 'deny' } } as const

export const DELETE = 'git push origin --delete feature/login'
export const FETCH = 'git fetch origin --prune'
export const BRANCHES = 'git branch -r'
export const CHAIN = `${DELETE} && ${FETCH} && ${BRANCHES}`
