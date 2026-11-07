import type { EngineInterface } from 'claude-code'

import type { Calibration, Decision, Spend } from '../../types'
import { APPS, type Color, type Section, SECTIONS, sectionOf } from './apps'
import { type Body, type Described, describe, type Gauge, type TallyItem, tallyOf, tallyWidth } from './describe'
import { ANSWERS, type Answer, answersFor } from './labels'
import { brand, type Logo, mark, markColumns, wordmark, wordmarkHeight } from './logo'
import { holdMeter, type Segment, textWidth } from './meters'
import { APP_NAMES, type Display, isCleartextRemote, type Settings } from './settings'
import { clip, fixed, modelName } from './text'

export type Elements = ReturnType<EngineInterface['ui']['resolve']>

const MIN_DETAIL_WIDTH = 8
// Below this the inline decision is the one-line form: a card would wrap.
const MIN_CARD_COLUMNS = 56
const MAX_CARD_WIDTH = 76
// A card's frame and padding take this many columns from its width.
const CARD_CHROME = 4
const COMPACT_RISK_CELLS = 12

// The detail gives way before the headline does.
function srLine(elements: Elements, { badge, color, headline, detail }: Described, columns: number) {
  const { Text } = elements
  const room = columns - `vSR  ${badge}  `.length
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
    const width = chip.length + 2 + (shown.length > 0 ? 1 : 0)
    const rest = chips.length - index - 1
    if (used + width + (rest > 0 ? ` +${rest}`.length : 0) > room && shown.length > 0) break
    shown.push(chip)
    used += width
  }
  const hidden = chips.length - shown.length
  return (
    <Text>
      {shown.flatMap((chip, index) => [
        index === 0 ? '' : ' ',
        <Text backgroundColor={color} color="inverseText">{` ${chip} `}</Text>,
      ])}
      {hidden > 0 ? <Text dimColor>{` +${hidden}`}</Text> : null}
    </Text>
  )
}

function bodyLine(elements: Elements, body: Body, color: Color, room: number) {
  return body.kind === 'gauge' ? meterLine(elements, body.gauge, room) : chipsLine(elements, body.chips, color, room)
}

// The logo's mark stands at the right of the card, as tall as its lines. The app tag goes at the right of the
// header, else of the footnote, wherever it fits without clipping either; else it is left out.
function card(
  elements: Elements,
  logo: Logo,
  { app, badge }: Described,
  one: Extract<Described['inline'], { kind: 'card' }>,
  columns: number,
) {
  const { Box, Text } = elements
  const rows = one.body ? 3 : 2
  const markRoom = markColumns(logo, rows) === 0 ? 0 : markColumns(logo, rows) + 1
  const width = Math.min(columns - CARD_CHROME, MAX_CARD_WIDTH + markRoom)
  const inner = width - CARD_CHROME - markRoom
  const lead = one.glyph.length + 1 + badge.length + 2 + 1
  const tail = one.tail ?? ''
  const title = `${clip(one.title, Math.max(1, inner - lead - tail.length))}${tail}`
  const label = `vSR · ${APPS[app].name}`
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
      {mark(elements, logo, rows)}
    </Box>
  )
}

const COMPACT_LEAD = '✓ allowed '

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
      <Text color="success">{COMPACT_LEAD}</Text>
      {`${part}${more}  `}
      <Text dimColor>risk ▕</Text>
      {meterCells(elements, segments, 'success')}
      <Text dimColor>▏</Text>
      {` ${fixed(score)}`}
      {isBound ? <Text dimColor>{bound}</Text> : null}
    </Text>
  )
}

// What a row shows of a decision under `display`: the full card or line, the one-line form, or nothing.
export function rowView(elements: Elements, logo: Logo, described: Described, columns: number, narrow: Described, display: Display) {
  if (display === 'verbose' || described.weight === 'act') return inlineView(elements, logo, described, columns, narrow)
  if (display === 'quiet' || described.weight === 'quiet') return null
  // The plain words go first when the row is short: a cut word reads worse than none.
  const fits = `vSR  ${described.badge}  ${described.line}`.length <= columns
  return srLine(elements, { ...described, headline: fits ? described.line : described.headline, detail: '' }, columns)
}

// A card for a decision worth noticing, a compact line for an allowed command, the one-line form where the row is narrow.
export function inlineView(elements: Elements, logo: Logo, described: Described, columns: number, narrow: Described = described) {
  if (columns < MIN_CARD_COLUMNS) return srLine(elements, narrow, columns)
  const { inline } = described
  return inline.kind === 'card'
    ? card(elements, logo, described, inline, columns)
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

const BAND_LEAD = 'vSR  '

const OFFLINE: TallyItem = { glyph: APPS.off.glyph, color: APPS.off.color, text: 'classifier offline' }

export function band(elements: Elements, list: readonly Decision[], columns: number, isOffline: boolean) {
  const { Text } = elements
  const items = isOffline ? [OFFLINE, ...tallyOf(list)] : tallyOf(list)
  return (
    <Text wrap="truncate-end">
      {brand(elements)}
      {'  '}
      {tallyText(elements, itemsWithin(items, columns, BAND_LEAD.length))}
    </Text>
  )
}

// What the pane's foot says of the setup: where the classifier runs, the apps on, the lines in force, the last calibration.
// `spend` and `sessionModel` describe what routing cost against the session's model; `labeled`, the answers given this session.
export type Overview = { settings: Settings; calibration: Calibration; isOffline: boolean; spend: Spend; sessionModel?: string; labeled: number }

const dollars = (value: number) => (value < 0.01 ? '<$0.01' : `$${value.toFixed(2)}`)

// What routing came to: an estimate, at list price, of the same tokens on the session's model.
function spendWords({ spend, sessionModel }: Overview) {
  if (spend.turns === 0 || sessionModel === undefined || spend.baseline === 0) return undefined
  const change = 1 - spend.actual / spend.baseline
  const verdict = change >= 0 ? `saved ${Math.round(change * 100)}%` : `cost ${Math.round(-change * 100)}% more`
  return `${dollars(spend.actual)} for ${spend.turns} subagent ${spend.turns === 1 ? 'turn' : 'turns'} vs ${dollars(spend.baseline)} on ${modelName(sessionModel)} · ${verdict} · est. at list price`
}

const tokenWords = ({ url, token }: Settings['runtime']) =>
  token === '' ? '' : isCleartextRemote(url) ? ' · token set, sent unencrypted' : ' · token set'

function overviewRows(about: Overview) {
  const { settings: { runtime, apps, thresholds: t, log }, calibration, isOffline, labeled } = about
  const routing = spendWords(about)
  const on = APP_NAMES.filter(app => apps[app])
  const off = APP_NAMES.filter(app => !apps[app])
  const since = `${calibration.since} ${calibration.since === 1 ? 'decision' : 'decisions'} since`
  return [
    ...(routing === undefined ? [] : [['routing', routing] as const]),
    ['classifier', `${runtime.kind} · ${runtime.url}${tokenWords(runtime)}${isOffline ? ' · not answering' : ''}`],
    ['apps', [...on, ...off.map(app => `${app} off`)].join(' · ')],
    [
      'lines',
      [
        `router ${fixed(t.router.easyBelow)} / ${fixed(t.router.hardFrom)}, confidence ${fixed(t.router.minConfidence)}, steps ${fixed(t.router.stepsLow)} / ${fixed(t.router.stepsHigh)}, confidence ${fixed(t.router.stepsMinConfidence)}`,
        `escalate ${fixed(t.escalate.feedback)}, reask ${fixed(t.escalate.reask)}`,
        `redact ${fixed(t.redact.pii)}`,
        `shield ${fixed(t.shield.inject)}`,
        `guard ${fixed(t.guard.review)}`,
        `check ${fixed(t.check.unsupported)}`,
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

export function dashboard(
  elements: Elements,
  logo: Logo,
  list: readonly Decision[],
  columns: number,
  rows: number,
  about: Overview,
  labeling?: Labeling,
) {
  const { Box, Button, Text } = elements
  const tally = tallyOf(list)
  const tallyRows = Math.max(1, Math.ceil(tallyWidth(tally) / columns))
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
    return entries.length === 0 ? [] : [{ section, entries }]
  })
  return (
    <Box flexDirection="column">
      {wordmark(elements, logo)}
      <Text>{tallyText(elements, tally)}</Text>
      {fitSections(sections, rows - wordmarkHeight(logo) - tallyRows - overviewHeight(about, columns)).map(({ section, count, entries }) => (
        <Box flexDirection="column">
          <Text color={APPS[section].color} weight="bold">{`${APPS[section].glyph} ${APPS[section].name} · ${count}`}</Text>
          {entries.map(({ decision, described: { badge, color, subject, more = '' }, second }) => {
            const kind = labeling ? answersFor(decision) : undefined
            return (
              <Box flexDirection="column">
                <Text>
                  <Text backgroundColor={color} color="inverseText">{` ${badge} `}</Text>
                  {` ${clip(subject, Math.max(1, columns - badge.length - 3 - more.length))}${more}`}
                </Text>
                <Text dimColor>{second}</Text>
                {labeling && kind ? (
                  <Box flexDirection="row" gap={1}>
                    {ANSWERS[kind].map(({ answer, label }) => {
                      const isChosen = labeling.given[decision.id] === answer
                      return (
                        <Button
                          key={`${decision.id}:${answer}`}
                          label={isChosen ? `✓ ${label}` : label}
                          variant={isChosen ? 'primary' : 'secondary'}
                          onPress={() => labeling.answer(decision, answer)}
                        />
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
