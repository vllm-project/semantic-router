import { expect, test } from 'claude-code/testing'

import { LOGO_NAME } from './lib/logo'
import { HUES, MARK_ART } from './lib/mark-art'

import {
  agentRow,
  bandRow,
  readCallOf,
  userRow,
  CHAIN,
  callOf,
  DELETE,
  ENGINE_ROW,
  engine,
  HOME,
  groupRow,
  NO,
  PANE,
  PANE_ID,
  PLUGIN,
  paneLines,
  paneText,
  PUSH,
  type Reply,
  rowTexts,
  rowTree,
  SPAWN,
  toolRow,
  vsrLines,
  YES,
} from './kit'

test('a Bash row with an allowed decision keeps the engine row and adds one compact line', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { review: 0.76 })
  on('tool.call', () => ({ result: { stdout: 'ok' } }))
  await $.tool.call({ ...PUSH, command: 'git push origin main', tool_use_id: 'call-3' })

  const { texts, cards } = await rowTree($, toolRow(callOf('call-3', 'git push origin main')))

  expect(texts[0]).toBe(ENGINE_ROW)
  expect(texts.filter(text => /^› allowed \S/.test(text))).toEqual([
    '› allowed git push origin main  risk ▕█████████▏░┃░▏ 0.76 < 0.90',
  ])
  expect(cards).toEqual([])
})

test('a row with no decision is returned as the engine drew it', async ($, on) => {
  engine(on, { review: 0.76 })
  on('tool.call', () => ({ result: { stdout: 'ok' } }))
  await $.tool.call({ ...PUSH, command: 'git push origin main', tool_use_id: 'call-3' })

  const lines = await rowTexts($, toolRow(callOf('call-4', 'ls')))

  expect(lines).toEqual([ENGINE_ROW])
})

test('a held then declined call draws a DECLINED card with the part, the risk meter and you said no', async ($, on) => {
  engine(on, { review: 0.98 })
  on('classic.PermissionRequest', () => NO)
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call(PUSH)

  const { texts, cards } = await rowTree($, toolRow(callOf('call-9', PUSH.command)))

  expect(texts[0]).toBe(ENGINE_ROW)
  expect(cards).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'error', paddingX: 1, width: 76 })])
  expect(texts).toContain('✕  DECLINED  git push --force origin main')
  expect(texts).toContain('risk ▕██████████████████████┃█▌▏  0.98  holds at 0.90')
  expect(texts).toContain('you said no · looks destructive')
})

test('a held call that is still waiting draws a HELD card with the part', async ($, on) => {
  const clock = engine(on, { review: 0.98 })
  on('classic.PermissionRequest', () => YES)
  // The call runs until the test ends it: one still open when the test ends is failed with no handler, and that
  // failure fails the whole file.
  let finish = (_: { result: object }) => {}
  const running = new Promise<void>(ran =>
    on('tool.call', () => new Promise<{ result: object }>(resolve => ((finish = resolve), ran()))),
  )
  const call = $.tool.call(PUSH)
  await clock.advance(1)

  const { texts, cards } = await rowTree($, toolRow(callOf('call-9', PUSH.command)))

  expect(cards).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'warning' })])
  expect(texts).toContain('‖  HELD  git push --force origin main')
  expect(texts).toContain('waiting for you · looks destructive')
  await running
  finish({ result: { stdout: '' } })
  await call
})

test('a held call the person let run draws an APPROVED card', async ($, on) => {
  engine(on, { review: 0.98 })
  on('classic.PermissionRequest', () => YES)
  on('tool.call', () => ({ result: { stdout: 'pushed' } }))
  await $.tool.call(PUSH)

  const { texts, cards } = await rowTree($, toolRow(callOf('call-9', PUSH.command)))

  expect(cards).toEqual([expect.objectContaining({ borderColor: 'permission' })])
  expect(texts).toContain('✓  APPROVED  git push --force origin main')
  expect(texts).toContain('you let it run · looks destructive')
})

test('a chain held on one part draws the part and its step count in the card title', async ($, on) => {
  engine(on, { review: 0.1, parts: { [DELETE]: 0.96 } })
  await $.tool.check({ tool: 'Bash', input: { command: CHAIN }, tool_use_id: 'call-9' })

  const { texts } = await rowTree($, toolRow(callOf('call-9', CHAIN)))

  expect(texts).toContain(`‖  HELD  ${DELETE} (+2 steps)`)
  expect(texts).toContain('risk ▕██████████████████████┃█░▏  0.96  holds at 0.90')
})

test('the compact line gives up its bound, then clips the command, as the row narrows', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { review: 0.76 })
  on('tool.call', () => ({ result: { stdout: 'ok' } }))
  await $.tool.call({ ...PUSH, command: 'git push origin main', tool_use_id: 'call-3' })
  const row = callOf('call-3', 'git push origin main')
  const compact = async (columns: number) =>
    (await rowTree($, toolRow(row, columns))).texts.find(text => /^› allowed \S/.test(text))

  expect(await compact(80)).toBe('› allowed git push origin main  risk ▕█████████▏░┃░▏ 0.76 < 0.90')
  expect(await compact(66)).toBe('› allowed git push origin main  risk ▕█████████▏░┃░▏ 0.76')
  expect(await compact(60)).toBe('› allowed git push origin ma…  risk ▕█████████▏░┃░▏ 0.76')
})

test('a card on the narrowest row drops the caption and the score text before the meter', { options: { display: 'verbose' } }, async ($, on) => {
  engine(on, { review: 0.98, complexity: { score: 1.61, confidence: 0.8 } })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  on('classic.PermissionRequest', () => NO)
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.agent.spawn(SPAWN)
  await $.tool.call(PUSH)

  const risk = await rowTree($, toolRow(callOf('call-9', PUSH.command), 60))
  const route = await rowTree($, agentRow('call-1', 60))

  expect(risk.cards).toEqual([expect.objectContaining({ width: 52 })])
  expect(risk.texts).toContain('risk ▕██████████████████████┃█▌▏  0.98')
  expect(route.texts).toContain('easy ▕██████│████████████│█▍░░░░▏ hard  1.61')
})

test('at 50 columns every kind falls back to the one-line form', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { complexity: { score: 0.09, confidence: 0.86 }, review: 0.2 }
  engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: `agent-${e.tool_use_id}` }))
  on('classic.PermissionRequest', () => NO)
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'routed' })
  reply.complexity = { score: 1, confidence: 0.1 }
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'kept' })
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'allowed' })
  reply.review = 0.98
  await $.tool.call({ ...PUSH, tool_use_id: 'blocked' })
  reply.errors = true
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'off' })

  const rows = [
    agentRow('routed', 50),
    agentRow('kept', 50),
    toolRow(callOf('allowed', 'git status'), 50),
    toolRow(callOf('blocked', PUSH.command), 50),
    agentRow('off', 50),
  ]
  const seen: string[][] = []
  for (const row of rows) {
    const { texts, cards } = await rowTree($, row)
    expect(cards).toEqual([])
    seen.push(vsrLines(texts))
  }

  expect(seen.map(lines => lines.length)).toEqual([1, 1, 1, 1, 1])
  expect(seen.map(([line]) => line.split(/ {2,}/)[1])).toEqual(['ROUTED', 'KEPT', 'ALLOWED', 'DECLINED', 'SKIPPED'])
  expect(seen[2][0]).toBe(' vSR   ALLOWED  ran normally, not risky e…')
})

test('a folded group keeps its line and adds a card for the declined call and a compact line for the allowed one', { options: { display: 'verbose' } }, async ($, on) => {
  const reply: Reply = { review: 0.98 }
  engine(on, reply)
  on('classic.PermissionRequest', () => NO)
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call(PUSH)
  reply.review = 0.2
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'call-10' })
  const calls = [callOf('call-9', PUSH.command), callOf('call-10', 'git status'), callOf('call-11', 'ls')]

  const { texts, cards } = await rowTree($, groupRow(calls, false))

  expect(texts[0]).toBe(ENGINE_ROW)
  expect(cards).toEqual([expect.objectContaining({ borderStyle: 'round', borderColor: 'error' })])
  expect(texts.filter(text => text.startsWith('✕  DECLINED'))).toEqual(['✕  DECLINED  git push --force origin main'])
  expect(texts.filter(text => /^› allowed \S/.test(text))).toEqual([
    '› allowed git status  risk ▕██▍░░░░░░░░┃░▏ 0.20 < 0.90',
  ])
})

test('an expanded group adds nothing, and a folded group with no decision is untouched', async ($, on) => {
  engine(on, { review: 0.2 })
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'call-10' })
  const decided = [callOf('call-10', 'git status')]

  expect(await rowTexts($, groupRow(decided, true))).toEqual([ENGINE_ROW])
  expect(await rowTexts($, groupRow([callOf('call-11', 'ls')], false))).toEqual([ENGINE_ROW])
})

test('the pane is not opened at session start, and /vsr opens it', async ($, on) => {
  const opened: string[] = []
  engine(on, { opened })

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })
  expect(opened).toEqual([])

  await $.command.run({ command: 'vsr', args: '' })
  expect(opened).toEqual([PANE_ID])
})

test('the tally counts each kind the session decided, and splits the held ones', async ($, on) => {
  const reply: Reply = { complexity: { score: 0.09, confidence: 0.86 } }
  engine(on, reply)
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: `agent-${e.tool_use_id}` }))
  let answer: typeof YES | typeof NO = NO
  on('classic.PermissionRequest', () => answer)
  on('tool.call', () => ({ result: { stdout: '' } }))
  const check = (command: string) => $.tool.check({ tool: 'Bash', input: { command } })

  await $.agent.spawn({ ...SPAWN, tool_use_id: 'a' })
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'b' })
  reply.complexity = { score: 1, confidence: 0.1 }
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'c' })
  reply.review = 0.2
  await check('ls')
  await check('git status')
  await check('git push origin main')
  reply.review = 0.98
  await $.tool.call({ ...PUSH, tool_use_id: 'h1' })
  answer = YES
  await $.tool.call({ ...PUSH, tool_use_id: 'h2' })
  reply.fails = true
  await check('git push --force origin dev')

  expect(await paneText($, /routed/)).toBe('‖ 2 held (1 declined, 1 approved)  ⇢ 2 routed  ≈ 1 kept  › 3 allowed')
})

test('the tally colours each glyph in its app colour and leaves out what did not happen', async ($, on) => {
  engine(on, { review: 0.2 })
  await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })

  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  const glyph = (await ui.findAll({ type: 'Text' })).find(row => row.text === '›')
  const tally = afterLogo((await ui.findAll({ type: 'Text' })).map(row => row.text ?? ''))[0]
  await ui.unmount()

  expect(glyph?.props).toMatchObject({ color: 'success' })
  expect(tally).toBe('› 1 allowed')
})

test('the pane draws every decision, more than its rows hold, for the engine to scroll', { options: { check: true } }, async ($, on) => {
  engine(on, { review: 0.2 })
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) await $.tool.check({ tool: 'Bash', input: { command: `git push origin b${n}` } })

  // Eight entries of four rows each are more than 35 rows hold.
  const lines = await paneLines($, { columns: 60, rows: 35 })

  expect(lines).toContain(ANSWER_HINT)
  expect(afterLogo(lines)[0]).toBe('› 8 allowed')
  expect(lines).toContain('‖ guard · 8')
  expect(lines.filter(text => text.startsWith(' ALLOWED  git'))).toEqual([8, 7, 6, 5, 4, 3, 2, 1].map(n => ` ALLOWED  git push origin b${n}`))
})

test('a section header counts the rows it lists, up to the hundred decisions the pane keeps', async ($, on) => {
  engine(on, { review: 0.2 })
  for (let n = 1; n <= 105; n++) await $.tool.check({ tool: 'Bash', input: { command: `git push origin b${n}` } })

  const lines = await paneLines($, { columns: 60, rows: 30 })
  const allowed = lines.filter(text => text.startsWith(' ALLOWED  git'))

  expect(lines).toContain('‖ guard · 100')
  expect(allowed.length).toBe(100)
  expect(allowed.at(-1)).toBe(' ALLOWED  git push origin b6')
})

test('the pane starts empty', async ($, on) => {
  engine(on, {})

  expect(await paneText($, /No decisions/)).toBe('No decisions yet. Each one appears under the row it decided on.')
  expect(await paneText($, /routed/)).toBeUndefined()
})

const TICKET = 'From: Maria Gonzalez <maria.gonzalez@example.com>'
const TICKET_SPANS = [
  { label: 'PERSON', start: 6, end: 20, text: 'Maria Gonzalez', probability: 0.95 },
  { label: 'EMAIL_ADDRESS', start: 22, end: 48, text: 'maria.gonzalez@example.com', probability: 0.99 },
]
const readResult = () => ({
  result: { type: 'text', file: { filePath: '/repo/ticket.txt', content: TICKET, numLines: 1, startLine: 1, totalLines: 1 } },
})

async function severalApps($: Parameters<Parameters<typeof test>[1]>[0], on: Parameters<Parameters<typeof test>[1]>[1]) {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, review: 0.2, pii: TICKET_SPANS, jailbreak: 0.99, feedback: { WRONG_ANSWER: 0.92, SAT: 0.08 } })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  on('tool.call', () => readResult())
  await $.agent.spawn(SPAWN)
  await $.tool.call({ tool: 'Read', file_path: '/repo/ticket.txt', tool_use_id: 'r1' })
  await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
}

test('the band shows vSR as a capsule, v white, S yellow and R blue, then one item per app that decided, each glyph in its app colour', async ($, on) => {
  await severalApps($, on)

  const { texts, props } = await rowTree($, bandRow())

  expect(texts[0]).toBe(' vSR   ◆ 1 shielded  ▒ 2 hidden  ⇢ 1 routed  › 1 allowed')
  expect(['v', 'S', 'R'].map(letter => props[texts.indexOf(letter)]?.color)).toEqual(['#ffffff', '#fdb618', '#32a3ff'])
  expect(props[texts.indexOf('v')]).toMatchObject({ backgroundColor: '#2d333f' })
  expect(props[texts.indexOf('⇢')]).toMatchObject({ color: 'suggestion' })
  expect(props[texts.indexOf('▒')]).toMatchObject({ color: 'merged' })
  expect(props[texts.indexOf('◆')]).toMatchObject({ color: 'warning' })
})

test('the band sits in a dim frame, so it reads apart from the reply above it', async ($, on) => {
  await severalApps($, on)

  const { texts, frames, cards } = await rowTree($, bandRow())

  expect(cards).toEqual([])
  expect(frames).toEqual([expect.objectContaining({ borderStyle: 'round', borderDimColor: true })])
  expect(texts[0]).toBe(' vSR   ◆ 1 shielded  ▒ 2 hidden  ⇢ 1 routed  › 1 allowed')
})

test('the band drops its frame where the engine leaves it one row', async ($, on) => {
  await severalApps($, on)

  const { texts, frames } = await rowTree($, bandRow(100, 1))

  expect(frames).toEqual([])
  expect(texts[0]).toBe(' vSR   ◆ 1 shielded  ▒ 2 hidden  ⇢ 1 routed  › 1 allowed')
})

test('the band keeps to whole items within its columns', async ($, on) => {
  await severalApps($, on)

  const { texts } = await rowTree($, bandRow(36))

  expect(texts[0]).toBe(' vSR   ◆ 1 shielded  ▒ 2 hidden')
})

test('the band leaves the engine its own row before any decision, and while a survey holds it', async ($, on) => {
  engine(on, { review: 0.2 })
  expect((await rowTree($, bandRow())).texts).toEqual([ENGINE_ROW])

  await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  const survey = bandRow()
  expect((await rowTree($, { ...survey, props: { ...survey.props, hasSurvey: true } })).texts).toEqual([ENGINE_ROW])
})

test('the pane has one section per app in urgency order, each with its newest entries first', async ($, on) => {
  await severalApps($, on)
  await $.tool.check({ tool: 'Bash', input: { command: 'git status' } })

  const lines = await paneLines($, { columns: 60, rows: 50 })
  const headers = lines.filter(text => /^\S+ \w+ · \d+$/.test(text))

  expect(afterLogo(lines)[0]).toBe('◆ 1 shielded  ▒ 2 hidden  ⇢ 1 routed  › 2 allowed')
  expect(headers).toEqual(['‖ guard · 2', '◆ shield · 1', '▒ redact · 1', '⇢ router · 1'])
  const guard = lines.slice(lines.indexOf('‖ guard · 2'), lines.indexOf('◆ shield · 1'))
  expect(guard.filter(text => text.startsWith(' ALLOWED  '))).toEqual([' ALLOWED  git status', ' ALLOWED  ls'])
  expect(lines).toContain(' REDACTED  ticket.txt')
})

test('a short pane still lists every section in full, each newest first', async ($, on) => {
  engine(on, { review: 0.2, pii: TICKET_SPANS })
  on('tool.call', () => readResult())
  for (const n of [1, 2, 3, 4, 5, 6]) await $.tool.check({ tool: 'Bash', input: { command: `git push origin b${n}` } })
  await $.tool.call({ tool: 'Read', file_path: '/repo/ticket.txt', tool_use_id: 'r1' })

  const lines = await paneLines($, { columns: 60, rows: 22 })

  expect(lines).toContain(' REDACTED  ticket.txt')
  expect(lines).toContain('‖ guard · 6')
  expect(lines.filter(text => text.startsWith(' ALLOWED  git'))).toEqual([6, 5, 4, 3, 2, 1].map(n => ` ALLOWED  git push origin b${n}`))
})

test('at 50 columns the cards of every new kind fall back to the one-line form', async ($, on) => {
  await severalApps($, on)
  await $.prompt.submit({ text: 'No, that is wrong.', origin: { kind: 'composer' } } as never)

  const read = await rowTree($, toolRow(readCallOf('r1', '/repo/ticket.txt'), 50))
  const prompt = await rowTree($, { ...userRow('No, that is wrong.'), viewport: { columns: 50, rows: 30 } })

  expect(read.cards).toEqual([])
  expect(vsrLines(read.texts).map(line => line.split(/ {2,}/)[1])).toEqual(['REDACTED', 'SHIELDED'])
  expect(prompt.cards).toEqual([])
  expect(vsrLines(prompt.texts)).toHaveLength(1)
})

// The line above the answers, once the pane offers any.
const ANSWER_HINT = 'Answers tune the thresholds when you run /vsr:calibrate.'

// What the pane draws under its logo.
// The pane's lines past its top: the logo, and the controls, whose last row is guard's waits and its chosen pill.
const afterLogo = (lines: string[]) => lines.slice(lines.findIndex(line => line.startsWith('guard waits')) + 2)

type Test = Parameters<Parameters<typeof test>[1]>[0]
type On = Parameters<Parameters<typeof test>[1]>[1]

// A session in `env` with a force push the person refused: a BLOCKED card under its row. `read` answers the mark's
// file, base64; left out, nothing answers.
async function blockedPush($: Test, on: On, { env, read }: { env?: Record<string, string>; read?: string } = {}) {
  engine(on, { review: 0.98, ...(env ? { env } : {}) })
  on('classic.PermissionRequest', () => NO)
  on('tool.call', () => ({ result: { stdout: '' } }))
  if (read !== undefined) on('fs.read', () => ({ value: { base64: read } }))
  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })
  await $.tool.call(PUSH)
}

// A one-pixel PNG: what the engine takes for a picture.
const MARK_PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4u1UUAAR6AciUn+EZAAAAAElFTkSuQmCC'

async function imagesIn($: Test, target: object) {
  const ui = await $.ui.mount({ plugin: PLUGIN, surface: 'terminal', viewport: { columns: 100, rows: 40 }, ...target } as never)
  const images = (await ui.findAll({ type: 'Image' })).map(row => row.props)
  await ui.unmount()
  return images
}

// The rows of the mark as the terminal shows them.
const artText = (rows: 2 | 3 | 4, set: 'smooth' | 'blocks' = 'blocks') => MARK_ART[set][rows].map(cells => cells.map(([char]) => char).join(''))

// Every row of every size and set of the mark.
const ALL_ART = (['smooth', 'blocks'] as const).flatMap(set => ([2, 3, 4] as const).flatMap(rows => artText(rows, set)))

test('a card draws no mark: no rows of the V and no picture, only the vSR tag', async ($, on) => {
  await blockedPush($, on)
  const card = toolRow(callOf('call-9', PUSH.command))

  const { texts, props, cards } = await rowTree($, card)

  expect(texts).toContain('✕  DECLINED  git push --force origin main')
  expect(texts).toContain(' vSR  · guard')
  expect(texts.filter(text => ALL_ART.includes(text))).toEqual([])
  // Yellow and blue are left only to the tag's S and R.
  expect(props.filter(one => one.color === HUES.Y || one.color === HUES.B)).toHaveLength(2)
  expect(cards).toEqual([expect.objectContaining({ width: 76 })])
  expect(await imagesIn($, card)).toEqual([])
  // The pane of the same session draws the mark, so the logo is on: the card leaves it out.
  const lines = await paneLines($)
  for (const row of artText(4)) expect(lines).toContain(row)
})

test('the pane opens with the V four rows high and the name beside it', async ($, on) => {
  await blockedPush($, on)

  const lines = await paneLines($)

  for (const row of artText(4)) expect(lines).toContain(row)
  expect(lines).toContain(LOGO_NAME)
})

test('with the logo off, neither the cards nor the pane draw it', { options: { logo: 'off' } }, async ($, on) => {
  await blockedPush($, on, { env: { HOME, TERM_PROGRAM: 'ghostty' }, read: MARK_PNG })

  const { texts, cards } = await rowTree($, toolRow(callOf('call-9', PUSH.command)))
  const lines = await paneLines($)

  expect(texts).toContain(' vSR  · guard')
  expect(cards).toEqual([expect.objectContaining({ width: 76 })])
  expect([...texts, ...lines].filter(text => ALL_ART.includes(text) || text === LOGO_NAME)).toEqual([])
  expect(await imagesIn($, PANE)).toEqual([])
})

test('in Ghostty the pane draws the mark as a picture, in the box the blocks would take, and the card none', async ($, on) => {
  await blockedPush($, on, { env: { HOME, TERM_PROGRAM: 'ghostty' }, read: MARK_PNG })

  expect(await imagesIn($, PANE)).toEqual([{ source: { png: MARK_PNG }, columns: 8, rows: 4, alt: 'vLLM' }])
  expect(await paneLines($)).not.toContain(artText(4)[1])
  expect(await imagesIn($, toolRow(callOf('call-9', PUSH.command)))).toEqual([])
})

test('inside tmux the mark is blocks, since the environment cannot tell whether pictures pass through', async ($, on) => {
  await blockedPush($, on, { env: { HOME, KITTY_WINDOW_ID: '1', TMUX: '/tmp/tmux-1/default,1,0' }, read: MARK_PNG })

  expect(await imagesIn($, PANE)).toEqual([])
  expect(await paneLines($)).toContain(artText(4)[1])
})

test('a mark file that is not a PNG gets characters (smooth ones in Ghostty), so the card and the pane still draw', async ($, on) => {
  await blockedPush($, on, { env: { HOME, TERM_PROGRAM: 'ghostty' }, read: btoa('not a png') })

  const { texts } = await rowTree($, toolRow(callOf('call-9', PUSH.command)))
  const lines = await paneLines($)

  expect(await imagesIn($, PANE)).toEqual([])
  expect(texts).toContain('✕  DECLINED  git push --force origin main')
  for (const row of artText(4, 'smooth')) expect(lines).toContain(row)
})

for (const [where, env, set] of [
  ['GNOME Terminal (VTE 0.76)', { HOME, VTE_VERSION: '7600' }, 'smooth'],
  ['VS Code 1.108', { HOME, TERM_PROGRAM: 'vscode', TERM_PROGRAM_VERSION: '1.108.2' }, 'smooth'],
  ['VS Code 1.107', { HOME, TERM_PROGRAM: 'vscode', TERM_PROGRAM_VERSION: '1.107.1' }, 'blocks'],
  ['tmux inside GNOME Terminal', { HOME, VTE_VERSION: '7600', TMUX: '/tmp/tmux-1/default,1,0' }, 'blocks'],
  ['an unknown terminal', { HOME }, 'blocks'],
] as const) {
  test(`in ${where} the pane draws the mark in ${set} characters`, async ($, on) => {
    await blockedPush($, on, { env })

    const lines = await paneLines($)

    for (const row of artText(4, set)) expect(lines).toContain(row)
  })
}

test('the logo option picks the smooth characters whatever the terminal says', { options: { logo: 'smooth' } }, async ($, on) => {
  await blockedPush($, on)

  expect(await paneLines($)).toContain(artText(4, 'smooth')[1])
})

test('the pane says when a token goes to a remote runtime over plain http', { options: { runtime: 'remote', runtimeUrl: 'http://10.0.0.5:8790/v1/decisions', runtimeToken: 'tok' } }, async ($, on) => {
  engine(on, {})

  expect(await paneText($, /^model/, 120)).toBe('model  remote · http://10.0.0.5:8790/v1/decisions · token set, sent unencrypted')
})

test('by default a route draws one plain line, with no card and no scores', async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 }, steps: { score: 0.17, confidence: 0.74 } })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  await $.agent.spawn(SPAWN)

  const { texts, cards } = await rowTree($, agentRow('call-1'))

  expect(cards).toEqual([])
  expect(texts[1]).toBe(' vSR   ROUTED  subagent → Haiku · easy task')
})

test('by default an allowed command draws nothing under it, and a held one draws its card', async ($, on) => {
  const reply: Reply = { review: 0.2 }
  engine(on, reply)
  on('classic.PermissionRequest', () => NO)
  on('tool.call', () => ({ result: { stdout: '' } }))
  await $.tool.call({ ...PUSH, command: 'git status', tool_use_id: 'allowed' })
  reply.review = 0.98
  await $.tool.call({ ...PUSH, tool_use_id: 'held' })

  expect(await rowTexts($, toolRow(callOf('allowed', 'git status')))).toEqual([ENGINE_ROW])
  expect((await rowTree($, toolRow(callOf('held', PUSH.command)))).cards).toEqual([expect.objectContaining({ borderColor: 'error' })])
})

test('with display quiet a route draws nothing under its row', { options: { display: 'quiet' } }, async ($, on) => {
  engine(on, { complexity: { score: 0.09, confidence: 0.86 } })
  on('agent.spawn', (_, e) => ({ model: e.model ?? e.parentModel, agentId: 'agent-1' }))
  await $.agent.spawn(SPAWN)

  expect(await rowTexts($, agentRow('call-1'))).toEqual([ENGINE_ROW])
})
