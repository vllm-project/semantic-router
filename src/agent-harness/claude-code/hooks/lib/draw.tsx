import type { EngineInterface } from 'claude-code'

import type { App, Calibration, Decision, Spend } from '../../types'
import { APP_NAMES, APPS, type Color, LOOKS, type Section, SECTIONS, sectionOf } from './apps'
import { type Body, type Described, describe, type Gauge, type TallyItem, tallyOf, tallyWidth, type Weight } from './describe'
import { ANSWERS, type Answer, answersFor } from './labels'
import { BRAND, brand, type Logo, wordmark, wordmarkHeight } from './logo'
import { holdMeter, type Segment, textWidth } from './meters'
import { choiceOf, type Display, isCleartextRemote, type Settings, WAIT_APPS, WAIT_CHOICES, type WaitApp, type WaitChoice } from './settings'
import { clip, fixed, modelName } from './text'

export type Elements = ReturnType<EngineInterface['ui']['resolve']>

const MIN_DETAIL_WIDTH = 8
// Below this the inline decision is the one-line form: a card would wrap.
const MIN_CARD_COLUMNS = 56
const MAX_CARD_WIDTH = 76
// A card's frame and padding take this many columns from its width.
const CARD_CHROME = 4
const COMPACT_RISK_CELLS = 12
// Below this many columns inside a frame, the one-line form goes unframed.
const MIN_FRAMED_COLUMNS = 24

// The detail gives way before the headline does.
function badgeLine(elements: Elements, { badge, color, headline, detail }: Described, columns: number) {
  const { Text } = elements
  const room = columns - `${BRAND}  ${badge}  `.length
  const head = clip(headline, Math.max(1, room))
  const roomLeft = room - head.length - 2
  return (
    <Text>
      {brand(elements)}
      {' '}
      <Text backgroundColor={color} color="inverseText">{` ${badge} `}</Text>
      {` ${head}`}
      {detail !== '' && roomLeft >= MIN_DETAIL_WIDTH ? <Text dimColor>{`  ${clip(detail, roomLeft)}`}</Text> : null}
    </Text>
  )
}

function meterCells({ Text }: Elements, segments: Segment[], fill: Color) {
  return segments.map(({ text, tone }) => (
    <Text color={tone === 'fill' ? fill : tone === 'hold' ? 'warning' : undefined} dimColor={tone === 'rest' || tone === 'rule'}>
      {text}
    </Text>
  ))
}

// The meter's own line: the score and its caption give way before the meter does.
function meterLine(elements: Elements, { lead, close, segments, fill, fits }: Gauge, room: number) {
  const { Text } = elements
  const used = lead.length + textWidth(segments) + close.length
  const [score, caption] = fits.find(([one, rest]) => used + 2 + one.length + rest.length <= room) ?? fits[fits.length - 1]
  return (
    <Text>
      <Text dimColor>{lead}</Text>
      {meterCells(elements, segments, fill)}
      <Text dimColor>{close}</Text>
      {'  '}
      <Text weight="bold">{score}</Text>
      {caption === '' ? null : <Text dimColor>{caption}</Text>}
    </Text>
  )
}

// Small badges in the card's color, as many as fit, then how many did not.
function chipsLine({ Text }: Elements, chips: string[], color: Color, room: number) {
  const shown: string[] = []
  let used = 0
  for (const [index, chip] of chips.entries()) {
    const width = chip.length + (shown.length > 0 ? ' · '.length : 0)
    const rest = chips.length - index - 1
    if (used + width + (rest > 0 ? ` +${rest}`.length : 0) > room && shown.length > 0) break
    shown.push(chip)
    used += width
  }
  const hidden = chips.length - shown.length
  return (
    <Text>
      {shown.flatMap((chip, index) => [index === 0 ? '' : <Text dimColor>{' · '}</Text>, <Text color={color}>{chip}</Text>])}
      {hidden > 0 ? <Text dimColor>{` +${hidden}`}</Text> : null}
    </Text>
  )
}

function bodyLine(elements: Elements, body: Body, color: Color, room: number) {
  return body.kind === 'gauge' ? meterLine(elements, body.gauge, room) : chipsLine(elements, body.chips, color, room)
}

// The app tag goes at the right of the header, else of the footnote, wherever it fits without clipping either;
// else it is left out. The logo's mark is drawn once, at the top of the pane, not on each card.
function card(
  elements: Elements,
  { app, badge }: Described,
  one: Extract<Described['inline'], { kind: 'card' }>,
  columns: number,
) {
  const { Box, Text } = elements
  const width = Math.min(columns - CARD_CHROME, MAX_CARD_WIDTH)
  const inner = width - CARD_CHROME
  const lead = one.glyph.length + 1 + badge.length + 2 + 1
  const tail = one.tail ?? ''
  const title = `${clip(one.title, Math.max(1, inner - lead - tail.length))}${tail}`
  const label = `${BRAND} · ${APPS[app].name}`
  const tag = (
    <Text>
      {brand(elements)}
      <Text dimColor>{` · ${APPS[app].name}`}</Text>
    </Text>
  )
  const tagRoom = label.length + 2
  const isHeaderTagged = lead + title.length + tagRoom <= inner
  const isNoteTagged = !isHeaderTagged && one.note.length + tagRoom <= inner
  return (
    <Box borderStyle="round" borderColor={one.color} paddingX={1} flexDirection="row" gap={1} width={width}>
      <Box flexDirection="column" flexGrow={1}>
        <Box flexDirection="row" justifyContent="space-between">
          <Text>
            <Text color={one.color}>{one.glyph}</Text>
            {' '}
            <Text backgroundColor={one.color} color="inverseText">{` ${badge} `}</Text>
            {` ${title}`}
          </Text>
          {isHeaderTagged ? tag : null}
        </Box>
        {one.body ? bodyLine(elements, one.body, one.color, inner) : null}
        <Box flexDirection="row" justifyContent="space-between">
          <Text dimColor>{clip(one.note, inner)}</Text>
          {isNoteTagged ? tag : null}
        </Box>
      </Box>
    </Box>
  )
}

const COMPACT_LEAD = `${LOOKS.allowed.glyph} allowed `

function compactLine(elements: Elements, { subject, more = '' }: Described, score: number, threshold: number, columns: number) {
  const { Text } = elements
  const segments = holdMeter(score, threshold, COMPACT_RISK_CELLS)
  const bar = textWidth(segments) + 'risk ▕'.length + '▏'.length
  const bound = ` < ${fixed(threshold)}`
  const spare = columns - COMPACT_LEAD.length - 2 - bar - ` ${fixed(score)}`.length - more.length
  const isBound = subject.length + bound.length <= spare
  const part = clip(subject, Math.max(1, spare))
  return (
    <Text>
      <Text color={LOOKS.allowed.color}>{COMPACT_LEAD}</Text>
      {`${part}${more}  `}
      <Text dimColor>risk ▕</Text>
      {meterCells(elements, segments, 'success')}
      <Text dimColor>▏</Text>
      {` ${fixed(score)}`}
      {isBound ? <Text dimColor>{bound}</Text> : null}
    </Text>
  )
}

// The one-line form in a dim frame of its colour, so it stands apart from Claude's words around it; unframed where
// the row is too narrow for one.
function framedLine(elements: Elements, described: Described, columns: number) {
  const { Box } = elements
  const room = columns - CARD_CHROME
  if (room < MIN_FRAMED_COLUMNS) return badgeLine(elements, described, columns)
  return (
    <Box borderStyle="round" borderColor={described.color} borderDimColor paddingX={1} alignSelf="flex-start">
      {badgeLine(elements, described, room)}
    </Box>
  )
}

// What a row shows of a decision under `display`: the full card or line, the one-line form, or nothing.
export function rowView(elements: Elements, described: Described, columns: number, narrow: Described, display: Display) {
  if (display === 'verbose' || described.weight === 'act') return inlineView(elements, described, columns, narrow)
  if (display === 'quiet' || described.weight === 'quiet') return null
  // The plain words go first when the row is short: a cut word reads worse than none.
  const room = columns - CARD_CHROME >= MIN_FRAMED_COLUMNS ? columns - CARD_CHROME : columns
  const fits = `${BRAND}  ${described.badge}  ${described.line}`.length <= room
  return framedLine(elements, { ...described, headline: fits ? described.line : described.headline, detail: '' }, columns)
}

// A card for a decision worth noticing, a compact line for an allowed command, the one-line form where the row is narrow.
export function inlineView(elements: Elements, described: Described, columns: number, narrow: Described = described) {
  if (columns < MIN_CARD_COLUMNS) return framedLine(elements, narrow, columns)
  const { inline } = described
  return inline.kind === 'card'
    ? card(elements, described, inline, columns)
    : compactLine(elements, described, inline.score, inline.threshold, columns)
}

function tallyText({ Text }: Elements, items: readonly TallyItem[]) {
  return items.flatMap((item, index) => [
    index === 0 ? '' : '  ',
    <Text color={item.color}>{item.glyph}</Text>,
    ` ${item.text}`,
  ])
}

// The whole items that fit `columns`, after a lead of `used` columns.
function itemsWithin(items: readonly TallyItem[], columns: number, used: number) {
  const kept: TallyItem[] = []
  for (const item of items) {
    if (used + tallyWidth([...kept, item]) > columns && kept.length > 0) break
    kept.push(item)
  }
  return kept
}

const BAND_LEAD = `${BRAND}  `

// Offline the decision model's checks step aside, so the band says which: only guard's fixed rules still hold.
// Offline, an app set to wait until done holds (the output, the command) instead of stepping aside.
export type Offline = { notChecking: readonly App[]; holding: readonly App[] }

const offlineItem = ({ notChecking, holding }: Offline): TallyItem => ({
  glyph: APPS.off.glyph,
  color: 'warning',
  text: [
    'model offline',
    ...(notChecking.length > 0 ? [`${notChecking.join(', ')} not checking`] : []),
    ...(holding.length > 0 ? [`${holding.join(', ')} holding`] : []),
  ].join(' · '),
})

// A frame takes the line above and below it.
const FRAME_ROWS = 3

// The tally above the prompt, in a dim frame so it reads apart from Claude's reply just above it; unframed where the
// engine leaves fewer rows than a frame takes or the columns are too few.
export function band(
  elements: Elements,
  list: readonly Decision[],
  columns: number,
  isOffline: boolean,
  maxRows = FRAME_ROWS,
  offline: Offline = { notChecking: [], holding: [] },
) {
  const { Box, Text } = elements
  const items = isOffline ? [offlineItem(offline), ...tallyOf(list)] : tallyOf(list)
  const isFramed = maxRows >= FRAME_ROWS && columns - CARD_CHROME >= MIN_FRAMED_COLUMNS
  const room = isFramed ? columns - CARD_CHROME : columns
  const line = (
    <Text wrap="truncate-end">
      {brand(elements)}
      {'  '}
      {tallyText(elements, itemsWithin(items, room, BAND_LEAD.length))}
    </Text>
  )
  return isFramed ? (
    <Box borderStyle="round" borderDimColor paddingX={1} alignSelf="flex-start">
      {line}
    </Box>
  ) : (
    line
  )
}

// What the pane's foot says of the setup: where the classifier runs, the lines in force, the last calibration (the
// controls at its top show the apps and the waits).
// `spend` is what routing cost against the session's model; `labeled`, the answers given this session.
export type Overview = { settings: Settings; calibration: Calibration; isOffline: boolean; spend: Spend; labeled: number }

const dollars = (value: number) => (value < 0.01 ? '<$0.01' : `$${value.toFixed(2)}`)

// What routing came to: an estimate, at list price, of the same tokens on the session's model.
function spendWords({ spend }: Overview) {
  const { sessionModel } = spend
  if (spend.turns === 0 || sessionModel === undefined || spend.baseline === 0) return undefined
  const change = 1 - spend.actual / spend.baseline
  const verdict = change >= 0 ? `saved ${Math.round(change * 100)}%` : `cost ${Math.round(-change * 100)}% more`
  return `${dollars(spend.actual)} for ${spend.turns} subagent ${spend.turns === 1 ? 'turn' : 'turns'} vs ${dollars(spend.baseline)} on ${modelName(sessionModel)} · ${verdict} · est. at list price`
}

const tokenWords = ({ url, token }: Settings['runtime']) =>
  token === '' ? '' : isCleartextRemote(url) ? ' · token set, sent unencrypted' : ' · token set'

function overviewRows(about: Overview) {
  const { settings: { runtime, thresholds: t, log }, calibration, isOffline, labeled } = about
  const routing = spendWords(about)
  const since = `${calibration.since} ${calibration.since === 1 ? 'decision' : 'decisions'} since`
  return [
    ...(routing === undefined ? [] : [['routing', routing] as const]),
    ['model', `${runtime.kind} · ${runtime.url}${tokenWords(runtime)}${isOffline ? ' · not answering' : ''}`],
    [
      'router',
      `easy below ${fixed(t.router.easyBelow)}, hard from ${fixed(t.router.hardFrom)} (of 2), sure from ${fixed(t.router.minConfidence)} · reasoning steps few below ${fixed(t.router.stepsLow)}, many from ${fixed(t.router.stepsHigh)}`,
    ],
    [
      'thresholds',
      [
        `guard holds at ${fixed(t.guard.review)}`,
        `check flags at ${fixed(t.check.unsupported)}`,
        `shield warns at ${fixed(t.shield.inject)}`,
        `redact hides at ${fixed(t.redact.pii)}`,
        `escalate at ${fixed(t.escalate.feedback)}`,
      ].join(' · '),
    ],
    ['calibrated', log.path === '' ? 'log off' : `${calibration.at ?? 'never'} · ${since}${labeled > 0 ? ` · ${labeled} answered here` : ''} · log ${log.path}`],
  ] as const
}

export function overview(elements: Elements, about: Overview) {
  const { Box, Text } = elements
  return (
    <Box flexDirection="column" marginTop={1}>
      {overviewRows(about).map(([label, value]) => (
        <Text>
          <Text dimColor>{`${label}  `}</Text>
          {value}
        </Text>
      ))}
    </Box>
  )
}

// The rows the overview takes at `columns`, its gap above included.
export const overviewHeight = (about: Overview, columns: number) =>
  1 + overviewRows(about).reduce((sum, [label, value]) => sum + Math.max(1, Math.ceil((label.length + 2 + value.length) / columns)), 0)

// What pressing a control at the top of the pane does: turn an app on or off, or pick how long one waits.
export type Controls = { toggle: (app: App) => void; wait: (app: WaitApp, choice: WaitChoice) => void }

// Three apps to a row: with each hotkey drawn before its name (`1: ● guard`), six would not fit 60 columns.
const APPS_PER_ROW = 3
const CONTROL_LABEL_WIDTH = 'redact waits'.length + 2
const WAIT_WORDS: Record<WaitChoice, string> = { '1': '1s', '2': '2s', '3': '3s', done: 'until done' }
const APP_ROWS = Array.from({ length: Math.ceil(APP_NAMES.length / APPS_PER_ROW) }, (_, row) =>
  APP_NAMES.slice(row * APPS_PER_ROW, (row + 1) * APPS_PER_ROW),
)

// The rows the controls take, their gap below included.
export const CONTROLS_HEIGHT = APP_ROWS.length + WAIT_APPS.length + 1

// The apps, each pressed (or its digit typed) to turn it on or off, then how long each protective app waits: the
// chosen wait a pill in the app's colour, the rest dim and as wide, so the row does not shift when the pick moves.
export function controls(elements: Elements, settings: Settings, act: Controls) {
  const { Box, Button, Text } = elements
  return (
    <Box flexDirection="column" marginBottom={1}>
      {APP_ROWS.map((row, index) => (
        <Box flexDirection="row">
          <Text>{(index === 0 ? 'apps' : '').padEnd(CONTROL_LABEL_WIDTH)}</Text>
          <Box flexDirection="row" gap={2}>
            {row.map(app => {
              const isOn = settings.apps[app]
              return (
                <Button key={`toggle:${app}`} plain hotkey={String(APP_NAMES.indexOf(app) + 1)} dimColor={!isOn} onPress={() => act.toggle(app)}>
                  {isOn ? <Text color={APPS[app].color}>{`● ${app}`}</Text> : `○ ${app}`}
                </Button>
              )
            })}
          </Box>
        </Box>
      ))}
      {WAIT_APPS.map(app => {
        const isOn = settings.apps[app]
        const chosen = choiceOf(settings.waits[app])
        return (
          <Box flexDirection="row">
            <Text dimColor={!isOn}>{`${app} waits`.padEnd(CONTROL_LABEL_WIDTH)}</Text>
            <Box flexDirection="row" gap={1}>
              {WAIT_CHOICES.map(choice => {
                const words = ` ${WAIT_WORDS[choice]} `
                return (
                  <Button key={`wait:${app}:${choice}`} plain dimColor={!isOn || choice !== chosen} onPress={() => act.wait(app, choice)}>
                    {choice === chosen ? (
                      <Text backgroundColor={APPS[app].color} color="inverseText" dimColor={!isOn}>
                        {words}
                      </Text>
                    ) : (
                      words
                    )}
                  </Button>
                )
              })}
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}

type Entry = { decision: Decision; described: Described; second: string; rows: number }

// The answers given in the pane, and what pressing one does; absent where answers cannot be kept (the log is off).
export type Labeling = { given: Readonly<Record<string, Answer>>; answer: (decision: Decision, answer: Answer) => void }

// Each section gets its header and newest entry first, then older entries go round the sections while rows are left.
function fitSections(sections: { section: Section; entries: Entry[] }[], budget: number) {
  const shown = sections.map(() => 0)
  let used = sections.reduce((sum, { entries }) => sum + 1 + entries[0].rows, 0)
  sections.forEach((_, index) => (shown[index] = 1))
  for (let isGrowing = true; isGrowing; ) {
    isGrowing = false
    sections.forEach(({ entries }, index) => {
      const next = entries[shown[index]]
      if (next !== undefined && used + next.rows <= budget) {
        shown[index] += 1
        used += next.rows
        isGrowing = true
      }
    })
  }
  return sections.map(({ section, entries }, index) => ({ section, count: entries.length, entries: entries.slice(0, shown[index]) }))
}

const WEIGHT_ORDER: Record<Weight, number> = { act: 0, notice: 1, quiet: 2 }

// Said once above the answers, so a person knows what answering does.
const ANSWER_HINT = 'Answers tune the thresholds when you run /vsr:calibrate.'

export function dashboard(
  elements: Elements,
  logo: Logo,
  list: readonly Decision[],
  columns: number,
  rows: number,
  about: Overview,
  act: Controls,
  labeling?: Labeling,
) {
  const { Box, Button, Text } = elements
  const tally = tallyOf(list)
  const tallyRows = Math.max(1, Math.ceil(tallyWidth(tally) / columns))
  const hint = labeling !== undefined && list.some(decision => answersFor(decision) !== undefined)
  const sections = SECTIONS.flatMap(section => {
    const entries = list
      .filter(decision => sectionOf(decision) === section)
      .reverse()
      .map(decision => {
        const described = describe(decision)
        const second = `${described.headline} · ${described.detail}`
        const answers = labeling && answersFor(decision) ? 1 : 0
        return { decision, described, second, rows: 1 + Math.ceil(second.length / columns) + answers }
      })
      // What needed a look (a held command, a withheld output) before what only went on, each newest first, so a
      // run of ALLOWED rows does not push a HELD one out of view.
      .sort((a, b) => WEIGHT_ORDER[a.described.weight] - WEIGHT_ORDER[b.described.weight])
    return entries.length === 0 ? [] : [{ section, entries }]
  })
  return (
    <Box flexDirection="column">
      {wordmark(elements, logo)}
      {controls(elements, about.settings, act)}
      <Text>{tallyText(elements, tally)}</Text>
      {hint ? <Text dimColor>{ANSWER_HINT}</Text> : null}
      {fitSections(sections, rows - wordmarkHeight(logo) - CONTROLS_HEIGHT - tallyRows - (hint ? Math.ceil(ANSWER_HINT.length / columns) : 0) - overviewHeight(about, columns)).map(({ section, count, entries }) => (
        <Box flexDirection="column">
          <Text color={APPS[section].color} weight="bold">{`${APPS[section].glyph} ${APPS[section].name} · ${count}`}</Text>
          {entries.map(({ decision, described: { badge, color, subject, more = '', headline, detail } }) => {
            const kind = labeling ? answersFor(decision) : undefined
            return (
              <Box flexDirection="column">
                <Text>
                  <Text backgroundColor={color} color="inverseText">{` ${badge} `}</Text>
                  {` ${clip(subject, Math.max(1, columns - badge.length - 3 - more.length))}${more}`}
                </Text>
                <Text>
                  {headline}
                  {detail === '' ? null : <Text dimColor>{` · ${detail}`}</Text>}
                </Text>
                {labeling && kind ? (
                  <Box flexDirection="row" gap={1}>
                    {ANSWERS[kind].map(({ answer, label }) => {
                      const isChosen = labeling.given[decision.id] === answer
                      return (
                        <Button
                          key={`${decision.id}:${answer}`}
                          label={isChosen ? `✓ ${label}` : label}
                          variant={isChosen ? 'primary' : 'secondary'}
                          plain
                          onPress={() => labeling.answer(decision, answer)}
                        >
                          {isChosen ? (
                            <Text backgroundColor="suggestion" color="inverseText">{` ✓ ${label} `}</Text>
                          ) : (
                            <Text dimColor>
                              {' '}
                              <Text dimColor underline>
                                {label}
                              </Text>{' '}
                            </Text>
                          )}
                        </Button>
                      )
                    })}
                  </Box>
                ) : null}
              </Box>
            )
          })}
        </Box>
      ))}
      {overview(elements, about)}
    </Box>
  )
}
