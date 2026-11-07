import type { EngineInterface, On } from 'claude-code'

import type { Decision } from '../types'
import { readOffline } from './lib/classify'
import { describe, named } from './lib/describe'
import { band, dashboard, type Elements, overview, rowView } from './lib/draw'
import { type Answer, giveLabel, labelsFile, readLabels } from './lib/labels'
import { markCalibrated, readCalibration } from './lib/log'
import { BLOCKS_LOGO, loadLogo, type Logo, NO_LOGO, wordmark } from './lib/logo'
import type { Display, Settings } from './lib/settings'
import { readDecisions } from './lib/state'
import { stamp } from './lib/env'
import { answerKey } from './lib/text'
import { readSpend } from './router'
import type { App } from '../types'

const PANE = 'vsr-decisions'
const PANE_TITLE = 'vSR · decisions'
const FALLBACK_COLUMNS = 100
const INLINE_INDENT = 2
// Rows the pane's frame takes from the surface's height.
const PANE_CHROME_ROWS = 4
// How much of a block's end must match the end of a checked answer.
const BLOCK_PROBE_CHARS = 120

const inlineColumns = (e: { viewport?: { columns: number } }) =>
  (e.viewport?.columns ?? FALLBACK_COLUMNS) - INLINE_INDENT - 2

// The logo the cards and the pane draw, loaded at session start (a reload is one).
let logo: Logo = BLOCKS_LOGO

export async function loadLook($: EngineInterface, settings: Settings) {
  logo = settings.logo === 'off' ? NO_LOGO : await loadLogo($, settings.logo)
}

const callIdOf = (decision: Decision) => (decision.kind === 'guard' ? decision.id : 'toolUseId' in decision ? decision.toolUseId : undefined)

const forCall = (list: readonly Decision[], toolUseId: string | undefined) =>
  toolUseId === undefined ? [] : list.filter(decision => callIdOf(decision) === toolUseId)

// The newest decision made on this prompt.
const forPrompt = (list: readonly Decision[], text: string) =>
  [...list].reverse().find(d => (d.kind === 'escalate' || d.kind === 'unavailable') && d.prompt !== undefined && d.prompt === text)

// The newest check whose answer ends the way this block does.
function forAnswerBlock(list: readonly Decision[], text: string) {
  const probe = answerKey(text).slice(-BLOCK_PROBE_CHARS)
  if (probe === '') return undefined
  return [...list].reverse().find(d => (d.kind === 'check' || d.kind === 'unavailable') && d.answerTail !== undefined && answerKey(d.answerTail).endsWith(probe))
}

const tagsOf = (counts: Partial<Record<string, number>>) => Object.keys(counts).sort().join(',')

// Said already this session: the same kinds of data hidden from the same source, the same app skipped the same way.
function isRepeat(decision: Decision, list: readonly Decision[]) {
  const earlier = list.slice(0, list.indexOf(decision))
  if (decision.kind === 'redact') {
    return earlier.some(one => one.kind === 'redact' && one.source === decision.source && tagsOf(one.counts) === tagsOf(decision.counts))
  }
  if (decision.kind === 'unavailable' && !decision.read) {
    return earlier.some(one => one.kind === 'unavailable' && one.app === decision.app && one.why === decision.why)
  }
  return false
}

// The row, and under it what `display` shows of the decisions made on it.
function under(elements: Elements, row: unknown, found: readonly Decision[], columns: number, view: { all: readonly Decision[]; display: Display }, isNamed = false) {
  const { Box } = elements
  const shown = found.flatMap(one => {
    // A runtime that could not be reached is said once, on the band; its skipped calls stay off the rows.
    if (one.kind === 'unavailable' && one.offline) return []
    const described = describe(one)
    if (view.display !== 'verbose' && described.weight === 'notice' && isRepeat(one, view.all)) return []
    const drawn = rowView(elements, described, columns, isNamed ? named(described) : described, view.display)
    return drawn === null ? [] : [<Box paddingLeft={INLINE_INDENT}>{drawn}</Box>]
  })
  if (shown.length === 0) return row
  return (
    <Box flexDirection="column">
      {row}
      {shown}
    </Box>
  )
}

export async function registerCommand($: EngineInterface) {
  await $.command.register({
    name: 'vsr',
    description: 'Show what the vSR decision model decided and how it is set up, in a pane',
    argumentHint: '[calibrated]',
  })
}

// The apps that stop checking while the decision model is offline (guard's fixed rules still hold).
const OFFLINE_APPS: readonly App[] = ['guard', 'redact', 'shield', 'check']

export function registerView(on: On, settings: Settings) {
  // Blocks until the session start loads the logo; a reload starts over here.
  logo = settings.logo === 'off' ? NO_LOGO : BLOCKS_LOGO

  // `/vsr calibrated` is what the calibration skill runs once it applied new thresholds.
  on('command.run', { command: 'vsr' }, async ($, e) => {
    if (e.args.trim() === 'calibrated') {
      const at = await stamp($)
      await markCalibrated($, at)
      return { text: `vSR: calibrated at ${at}.` }
    }
    await $.ui.open({ id: PANE, title: PANE_TITLE })
    return { text: 'vSR pane opened.' }
  })

  on('ui.render', { component: 'ToolUse' }, async ($, e, next) => {
    const all = await readDecisions($)
    const found = forCall(all, e.props.tool_use_id)
    if (found.length === 0) return next(e)
    return under($.ui.resolve(e), await next(e), found, inlineColumns(e), { all, display: settings.display })
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'ToolGroup' }, async ($, e, next) => {
    if (e.props.isExpanded) return next(e)
    const list = await readDecisions($)
    const found = e.props.calls.flatMap(call => forCall(list, call.tool_use_id))
    if (found.length === 0) return next(e)
    return under($.ui.resolve(e), await next(e), found, inlineColumns(e), { all: list, display: settings.display }, true)
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'UserMessage' }, async ($, e, next) => {
    const all = await readDecisions($)
    const found = forPrompt(all, e.props.text)
    if (found === undefined) return next(e)
    return under($.ui.resolve(e), await next(e), [found], inlineColumns(e), { all, display: settings.display })
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'AssistantMessage' }, async ($, e, next) => {
    if (e.props.isSummary) return next(e)
    const all = await readDecisions($)
    const found = forAnswerBlock(all, e.props.text)
    if (found === undefined) return next(e)
    return under($.ui.resolve(e), await next(e), [found], inlineColumns(e), { all, display: settings.display })
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const list = await readDecisions($)
    const isOffline = await readOffline($)
    if (e.props.hasSurvey || e.props.maxRows < 1 || (list.length === 0 && !isOffline)) return next(e)
    return band($.ui.resolve(e), list, e.props.bodyColumns, isOffline, e.props.maxRows, OFFLINE_APPS.filter(app => settings.apps[app]))
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const elements = $.ui.resolve(e)
    const list = await readDecisions($)
    const given = await readLabels($)
    const about = {
      settings,
      calibration: await readCalibration($),
      isOffline: await readOffline($),
      spend: await readSpend($),
      labeled: Object.keys(given).length,
    }
    if (list.length === 0) {
      const { Box, Text } = elements
      return (
        <Box flexDirection="column">
          {wordmark(elements, logo)}
          <Text dimColor>No decisions yet. Each one appears under the row it decided on.</Text>
          {overview(elements, about)}
        </Box>
      )
    }
    // Answers given here are labels /vsr:calibrate fits the thresholds to; with the log off nothing keeps them.
    const labeling =
      (await labelsFile($, settings.log)) === undefined ? undefined : { given, answer: (d: Decision, answer: Answer) => void giveLabel($, settings.log, d, answer) }
    return dashboard(elements, logo, list, e.props.bodyColumns, (e.viewport?.rows ?? 24) - PANE_CHROME_ROWS, about, labeling)
  })
}
