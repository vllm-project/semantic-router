import type { App, Decision, Signal, Tag, Tiers } from '../../types'
import { APPS, type Color, LOOKS } from './apps'
import { difficultyMeter, holdMeter, type Segment } from './meters'
import { clip, fixed, modelName, oneLine, plural, RISK_WORDS } from './text'

// One meter row: `lead`, the cells, `close`, then the first of `fits` (score, caption) that fits the row.
export type Gauge = { lead: string; close: string; segments: Segment[]; fill: Color; fits: (readonly [string, string])[] }

export type Body = { kind: 'gauge'; gauge: Gauge } | { kind: 'chips'; chips: string[] }

// A card for the decisions worth noticing, a single compact line for an allowed command.
export type Inline =
  | { kind: 'card'; glyph: string; color: Color; title: string; tail?: string; body?: Body; note: string }
  | { kind: 'compact'; score: number; threshold: number }

// How much a decision asks of the person: `act`, look now (held, shielded, unverified, read in part with data
// hidden); `notice`, vSR changed something worth a line; `quiet`, all went as it would have, counted on the band.
export type Weight = 'act' | 'notice' | 'quiet'

export type Described = {
  app: App
  weight: Weight
  // The plain words a one-line row ends with: no scores, no lines.
  brief: string
  // The whole one-line row after the badge, what matters first: it names its own source, so a folded group adds nothing.
  line: string
  badge: string
  color: Color
  headline: string
  detail: string
  subject: string
  more?: string
  inline: Inline
}

// What a skipped call let through, in the words of what it would have done.
const SKIPPED: Record<App, (subject: string) => string> = {
  guard: () => 'command ran unchecked',
  check: () => 'answer not checked',
  shield: subject => `${subject} went to Claude unchecked for instructions`,
  redact: subject => `${subject} went to Claude unchecked for personal data`,
  router: () => 'subagent started without routing',
  escalate: () => 'pushback not checked',
}

// What a withheld output was not checked for.
const WITHHELD_FOR: Partial<Record<App, string>> = { redact: 'personal data', shield: 'instructions' }

type Look = Omit<Described, 'weight' | 'brief' | 'line'>
type Kind<D extends Decision> = {
  look: (d: D) => Look
  weight: Weight | ((d: D) => Weight)
  brief?: (d: D) => string
  line?: (d: D, brief: string) => string
}

const taskWord = (level: number, { easyBelow, hardFrom }: Tiers) =>
  level < easyBelow ? 'easy task' : level < hardFrom ? 'multi-step task' : 'hard task'

const riskColor = (score: number, line: number): Color => (score >= line ? 'error' : 'success')

const difficultyGauge = (value: number, tiers: Tiers, fill: Color): Body => ({
  kind: 'gauge',
  gauge: {
    lead: 'easy ▕',
    close: '▏ hard',
    segments: difficultyMeter(value, tiers),
    fill,
    fits: [[`${fixed(value)} of 2`, ` · ${taskWord(value, tiers)}`], [`${fixed(value)} of 2`, ''], [fixed(value), '']],
  },
})

// `word` names the line: the guard's holds a command, the shield's only warns.
const riskGauge = (label: string, value: number, line: number, word = 'holds at'): Body => ({
  kind: 'gauge',
  gauge: {
    lead: `${label} ▕`,
    close: '▏',
    segments: holdMeter(value, line),
    fill: riskColor(value, line),
    fits: [[fixed(value), `  ${word} ${fixed(line)}`], [fixed(value), '']],
  },
})

const unsupportedGauge = (value: number, line: number, fill: Color): Body => ({
  kind: 'gauge',
  gauge: {
    lead: 'unsupported ▕',
    close: '▏',
    segments: holdMeter(value, line),
    fill,
    fits: [[fixed(value), `  flags at ${fixed(line)}`], [fixed(value), '']],
  },
})

// No part scored: every one timed out or failed, so there is no score to show.
const isUnscored = (d: Extract<Decision, { kind: 'guard' }>) => d.unchecked !== undefined && d.unchecked >= d.parts

const reviewScore = (d: Extract<Decision, { kind: 'guard' }>) =>
  isUnscored(d) ? 'no step was scored' : `risk score ${fixed(d.review)} ${d.review >= d.threshold ? '≥' : '<'} ${fixed(d.threshold)}`

const stepsMore = (parts: number) =>
  parts > 1 ? ` (+${parts - 1} ${parts === 2 ? 'step' : 'steps'})` : undefined

const SIGNAL_WORDS: Record<Signal['kind'], string> = { wrong: 'said it was wrong', different: 'asked for something else', repeat: 'asked again' }

const signalText = (signal: Signal) => `${SIGNAL_WORDS[signal.kind]} ${fixed(signal.score)}`

export const TAGS: readonly Tag[] = ['name', 'email', 'phone', 'card', 'ip', 'date', 'ssn', 'iban', 'address', 'id', 'secret']

const hiddenCount = (counts: Partial<Record<Tag, number>>) => Object.values(counts).reduce((sum, n) => sum + (n ?? 0), 0)

const countChips = (counts: Partial<Record<Tag, number>>) =>
  TAGS.flatMap(tag => (counts[tag] ? [`${tag} ×${counts[tag]}`] : []))

const readWords = (files: number, commands: number) =>
  [files > 0 ? `${plural(files, 'file')} read` : '', commands > 0 ? plural(commands, 'command output') : '']
    .filter(Boolean)
    .join(' and ')

// Where a route's effort came from, short: the steps answer and what it did.
function effortNote(d: Extract<Decision, { kind: 'route' }>) {
  const steps = d.steps === null ? '' : fixed(d.steps.score)
  switch (d.effortFrom) {
    case 'haiku':
      return `${steps === '' ? '' : `reasoning steps ${steps} · `}Haiku runs at ${d.picked}`
    case 'steps':
      return `reasoning steps ${steps} → ${d.picked}`
    case 'unsure':
      return `reasoning steps ${d.steps === null ? 'unavailable' : `unsure (${fixed(d.steps.confidence)})`} → default ${d.picked}`
    case 'floor':
      return `${steps === '' ? '' : `reasoning steps ${steps} · `}keeps its own ${d.effort} effort`
    case 'escalated':
      return `${steps === '' ? '' : `reasoning steps ${steps} · `}one level up → ${d.picked}`
  }
}

// What Claude read, as a title names it: a file by its path, a command by its output.
const outputOf = (tool: string, source: string) => (tool === 'Read' ? source : `the output of ${source}`)

// One entry per decision kind: how it looks (the inline row, the folded group and the pane all draw it), how much it
// asks of the user, and the plain words its one-line row ends with (by default the card's note).
const KINDS: { [K in Decision['kind']]: Kind<Extract<Decision, { kind: K }>> } = {
  route: {
    look: d => {
      const title = `subagent → ${modelName(d.model)}${d.effort ? ` · ${d.effort} effort` : ''}`
      const alone = d.escalated && (d.escalated.alone === 'default' ? 'keep the default model' : `pick ${modelName(d.escalated.alone)}`)
      return {
        app: 'router',
        badge: 'ROUTED',
        color: APPS.router.color,
        headline: `subagent → ${modelName(d.model)}${d.effort ? `, ${d.effort} effort` : ''}`,
        detail: `${taskWord(d.level, d.tiers)} · difficulty ${fixed(d.level)} of 2 · confidence ${fixed(d.confidence)} · ${effortNote(d)}${alone ? ` · escalated, router alone would ${alone}` : ''}`,
        subject: oneLine(d.task),
        inline: {
          kind: 'card',
          glyph: APPS.router.glyph,
          color: APPS.router.color,
          title,
          body: difficultyGauge(d.level, d.tiers, APPS.router.color),
          note: alone ? `escalated: you pushed back · router alone would ${alone}` : `${effortNote(d)} · ${oneLine(d.task)}`,
        },
      }
    },
    weight: 'notice',
    brief: d => (d.escalated ? 'you pushed back' : taskWord(d.level, d.tiers)),
  },
  keep: {
    look: d => {
      const note =
        d.reason === 'quality'
          ? 'the routing goal is quality: never a smaller model'
          : d.reason === 'downgrade'
            ? `not sure enough for a smaller model · confidence ${fixed(d.confidence)} < ${fixed(d.minConfidence)}`
            : `model unsure · confidence ${fixed(d.confidence)} < ${fixed(d.minConfidence)}`
      return {
        app: 'router',
        badge: 'KEPT',
        color: LOOKS.kept.color,
        headline: `subagent stays on ${modelName(d.model)}`,
        detail: note,
        subject: oneLine(d.task),
        inline: {
          kind: 'card',
          ...LOOKS.kept,
          title: `subagent stays on ${modelName(d.model)}`,
          body: difficultyGauge(d.level, d.tiers, LOOKS.kept.color),
          note,
        },
      }
    },
    weight: 'notice',
    brief: d =>
      d.reason === 'quality' ? 'routing goal is quality' : d.reason === 'downgrade' ? 'not sure enough for a smaller model' : 'model unsure',
  },
  guard: {
    look: d => {
      const words = d.rule ? `always held: ${d.rule}` : d.unchecked ? `${d.unchecked} of ${plural(d.parts, 'step')} not checked` : RISK_WORDS[d.risk]
      const part = oneLine(d.part)
      const more = stepsMore(d.parts)
      const base = { app: 'guard' as const, subject: part, more }
      const asked = (badge: string, color: Color, glyph: string, headline: string, detail: string): Described => ({
        ...base,
        badge,
        color,
        headline,
        detail,
        inline: {
          kind: 'card',
          glyph,
          color,
          title: part,
          tail: more,
          ...(isUnscored(d) ? {} : { body: riskGauge('risk', d.review, d.threshold) }),
          note: headline,
        },
      })
      if (d.verdict === 'pass') {
        const riskiest = d.parts > 1 ? ` · riskiest of ${d.parts} steps` : ''
        return {
          ...base,
          badge: 'ALLOWED',
          color: 'success',
          headline: d.risk === 'safe' ? `${words}, ran normally` : 'ran normally, not risky enough to hold',
          detail: reviewScore(d) + riskiest,
          inline: { kind: 'compact', score: d.review, threshold: d.threshold },
        }
      }
      if (d.outcome === 'approved') return asked('APPROVED', 'permission', '✓', `you let it run · ${words}`, reviewScore(d))
      if (d.outcome === 'declined') return asked('DECLINED', 'error', '✕', `you said no · ${words}`, reviewScore(d))
      return asked('HELD', 'warning', APPS.guard.glyph, `waiting for you · ${words}`, `${reviewScore(d)} · ${part}`)
    },
    weight: d => (d.verdict === 'ask' ? 'act' : 'quiet'),
  },
  escalate: {
    look: d => {
      const headline = 'you pushed back · next subagent one step up'
      return {
        app: 'escalate',
        badge: 'ESCALATED',
        color: APPS.escalate.color,
        headline,
        detail: `${d.signals.map(signalText).join(' · ')} · told Claude to re-check its last answer`,
        subject: oneLine(d.prompt),
        inline: {
          kind: 'card',
          glyph: APPS.escalate.glyph,
          color: APPS.escalate.color,
          title: headline,
          body: { kind: 'chips', chips: d.signals.map(signalText) },
          note: 'told Claude to re-check its last answer',
        },
      }
    },
    weight: 'notice',
    brief: () => 'Claude re-checks its last answer',
  },
  redact: {
    look: d => {
      const chips = countChips(d.counts)
      const total = hiddenCount(d.counts)
      const first = TAGS.find(tag => d.counts[tag])
      const headline = `${plural(total, 'item')} hidden ${d.tool === 'Read' ? 'before Claude read' : 'from'} ${outputOf(d.tool, d.source)}`
      return {
        app: 'redact',
        badge: 'REDACTED',
        color: APPS.redact.color,
        headline,
        detail: chips.join(' · '),
        subject: d.source,
        inline: {
          kind: 'card',
          glyph: APPS.redact.glyph,
          color: APPS.redact.color,
          title: headline,
          body: { kind: 'chips', chips },
          note: d.unread
            ? `the last ${d.unread.toLocaleString('en-US')} characters went to Claude unchecked`
            : `replaced with [${first ?? 'name'}]-style tags · ${d.checkedOn ? `the originals went to ${d.checkedOn}` : 'the originals stayed on this machine'}`,
        },
      }
    },
    weight: d => (d.unread ? 'act' : 'notice'),
    brief: d => TAGS.filter(tag => d.counts[tag]).join(', '),
    line: (d, brief) => `${plural(hiddenCount(d.counts), 'item')} hidden: ${brief} · ${d.tool === 'Read' ? d.source : `output of ${d.source}`}`,
  },
  shield: {
    look: d => {
      const headline = `${outputOf(d.tool, d.source)} tried to give Claude instructions`
      return {
        app: 'shield',
        badge: 'SHIELDED',
        color: APPS.shield.color,
        headline,
        detail: `injection score ${fixed(d.score)} ≥ ${fixed(d.threshold)} · Claude was told to treat it as data`,
        subject: d.source,
        inline: {
          kind: 'card',
          glyph: APPS.shield.glyph,
          color: APPS.shield.color,
          title: headline,
          body: riskGauge('injection', d.score, d.threshold, 'warns at'),
          note: 'Claude was told to treat it as data, not instructions',
        },
      }
    },
    weight: 'act',
  },
  check: {
    look: d => {
      const isGrounded = d.verdict === 'grounded'
      const color: Color = isGrounded ? 'success' : 'warning'
      const headline = isGrounded ? 'answer matches what Claude read this turn' : 'some claims are not backed by what Claude read'
      const against = `checked against ${readWords(d.files, d.commands)} this turn`
      return {
        app: 'check',
        badge: isGrounded ? 'GROUNDED' : 'UNVERIFIED',
        color,
        headline,
        detail: `unsupported ${fixed(d.score)} ${isGrounded ? '<' : '≥'} ${fixed(d.threshold)} · ${against}`,
        subject: oneLine(d.prompt) || 'the answer',
        inline: {
          kind: 'card',
          glyph: isGrounded ? APPS.check.glyph : LOOKS.unverified.glyph,
          color,
          title: headline,
          body: unsupportedGauge(d.score, d.threshold, color),
          note: isGrounded ? against : 'look twice before you rely on it',
        },
      }
    },
    weight: d => (d.verdict === 'grounded' ? 'quiet' : 'act'),
  },
  unavailable: {
    look: d => {
      if (d.withheld) {
        const headline = `${oneLine(d.subject)} withheld from Claude: not checked for ${WITHHELD_FOR[d.app] ?? d.app}`
        return {
          app: d.app,
          badge: 'WITHHELD',
          color: LOOKS.withheld.color,
          headline,
          detail: `${d.why} · ${d.app} waits until done · /vsr:setup checks it`,
          subject: oneLine(d.subject),
          inline: { kind: 'card', ...LOOKS.withheld, title: headline, note: d.why },
        }
      }
      if (d.read) {
        const headline = `${d.app} read the first ${d.read.chars.toLocaleString('en-US')} of ${d.read.of.toLocaleString('en-US')} characters`
        const note = `the rest went to Claude unchecked · ${d.why}`
        return {
          app: d.app,
          badge: 'PARTIAL',
          color: APPS.off.color,
          headline,
          detail: note,
          subject: oneLine(d.subject),
          inline: { kind: 'card', glyph: APPS.off.glyph, color: APPS.off.color, title: headline, note },
        }
      }
      const headline = SKIPPED[d.app](oneLine(d.subject))
      return {
        app: d.app,
        badge: 'SKIPPED',
        color: APPS.off.color,
        headline,
        detail: `${d.why} · /vsr:setup checks it`,
        subject: oneLine(d.subject),
        inline: { kind: 'card', glyph: APPS.off.glyph, color: APPS.off.color, title: headline, note: d.why },
      }
    },
    // A withheld output is a card: Claude got nothing, and the person decides what to change.
    weight: d => (d.withheld ? 'act' : 'notice'),
    brief: d => (d.read ? 'the rest went to Claude unchecked' : d.why),
  },
}

export function describe(decision: Decision): Described {
  const kind = KINDS[decision.kind] as Kind<Decision>
  const look = kind.look(decision)
  const weight = typeof kind.weight === 'function' ? kind.weight(decision) : kind.weight
  const brief = kind.brief?.(decision) ?? (look.inline.kind === 'card' ? look.inline.note : look.detail)
  const line = kind.line?.(decision, brief) ?? (brief === '' ? look.headline : `${look.headline} · ${brief}`)
  return { ...look, weight, brief, line }
}

// A folded group hides its calls, so each of its lines names the call it judges.
const NAMED_SUBJECT_WIDTH = 48
export const named = (described: Described): Described => ({
  ...described,
  headline: `${clip(described.subject, NAMED_SUBJECT_WIDTH)} · ${described.headline}`,
})

export type TallyItem = { glyph: string; color: Color; text: string }

const heldOutcomes = (list: readonly Decision[]) => {
  const asked = list.flatMap(d => (d.kind === 'guard' && d.verdict === 'ask' ? [d.outcome ?? 'waiting'] : []))
  const inside = ([['waiting', 'waiting'], ['declined', 'declined'], ['approved', 'approved']] as const)
    .map(([outcome, word]) => [asked.filter(one => one === outcome).length, word] as const)
    .filter(([count]) => count > 0)
    .map(([count, word]) => `${count} ${word}`)
  return inside.length > 0 ? ` (${inside.join(', ')})` : ''
}

// Most urgent first, so a narrow band drops what went as usual before what needs a look; an item shows once its count is above zero.
const TALLIES: readonly { glyph: string; color: Color; word: string; count: (list: readonly Decision[]) => number; more?: (list: readonly Decision[]) => string }[] = [
  { ...APPS.guard, word: 'held', count: list => list.filter(d => d.kind === 'guard' && d.verdict === 'ask').length, more: heldOutcomes },
  { ...LOOKS.unverified, word: 'unverified', count: list => list.filter(d => d.kind === 'check' && d.verdict === 'unverified').length },
  { ...LOOKS.withheld, word: 'withheld', count: list => list.filter(d => d.kind === 'unavailable' && d.withheld).length },
  { ...APPS.shield, word: 'shielded', count: list => list.filter(d => d.kind === 'shield').length },
  { ...APPS.redact, word: 'hidden', count: list => list.reduce((sum, d) => sum + (d.kind === 'redact' ? hiddenCount(d.counts) : 0), 0) },
  { ...APPS.router, word: 'routed', count: list => list.filter(d => d.kind === 'route').length },
  { ...APPS.escalate, word: 'escalated', count: list => list.filter(d => d.kind === 'escalate').length },
  { ...LOOKS.kept, word: 'kept', count: list => list.filter(d => d.kind === 'keep').length },
  { ...LOOKS.allowed, word: 'allowed', count: list => list.filter(d => d.kind === 'guard' && d.verdict === 'pass').length },
  { ...APPS.check, word: 'grounded', count: list => list.filter(d => d.kind === 'check' && d.verdict === 'grounded').length },
  { ...APPS.off, word: 'skipped', count: list => list.filter(d => d.kind === 'unavailable' && !d.offline && !d.withheld).length },
]

export function tallyOf(list: readonly Decision[]): TallyItem[] {
  return TALLIES.flatMap(({ glyph, color, word, count, more }) => {
    const n = count(list)
    return n > 0 ? [{ glyph, color, text: `${n} ${word}${more?.(list) ?? ''}` }] : []
  })
}

export const tallyWidth = (items: readonly TallyItem[]) =>
  items.reduce((sum, item, index) => sum + (index === 0 ? 0 : 2) + item.glyph.length + 1 + item.text.length, 0)
