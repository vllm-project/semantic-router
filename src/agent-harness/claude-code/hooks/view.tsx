import type { ConfigValue, EngineInterface, On, PluginOptions } from 'claude-code'

import type { Decision } from '../types'
import { readOffline } from './lib/classify'
import { describe, named } from './lib/describe'
import { APP_NAMES, appOf } from './lib/apps'
import { band, type Controls, controls, dashboard, type Elements, overview, rowView, turnsWords } from './lib/draw'
import { type Answer, giveLabel, labelsFile, readLabels } from './lib/labels'
import { markCalibrated, readCalibration } from './lib/log'
import { BLOCKS_LOGO, loadLogo, type Logo, NO_LOGO, wordmark } from './lib/logo'
import { follow, pin, unpinAll } from './lib/profile'
import { choiceOf, type Display, type Settings, WAIT_APPS, WAIT_CHOICES, type WaitApp, type WaitChoice } from './lib/settings'
import { readDecisions } from './lib/state'
import { stamp } from './lib/env'
import { answerKey } from './lib/text'
import { readSpend } from './router'
import { readTurns, resumeTurns } from './turns'
import type { App } from '../types'

const PANE = 'vsr-decisions'
const PANE_TITLE = 'vSR · decisions'
const FALLBACK_COLUMNS = 100
const INLINE_INDENT = 2
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

// The decisions made on this prompt: a pushback, where the turn it started runs, a call skipped. A row names its
// prompt by text alone, so a prompt sent twice shows the newest of each kind, not both turns' stacked.
const ON_PROMPT = new Set<Decision['kind']>(['escalate', 'unavailable', 'route', 'keep'])
function forPrompt(list: readonly Decision[], text: string) {
  const found = list.filter(d => ON_PROMPT.has(d.kind) && 'prompt' in d && d.prompt === text)
  return found.filter((d, i) => !found.slice(i + 1).some(later => appOf(later) === appOf(d)))
}

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
    description:
      'Show what the vSR decision model decided and how it is set up, in a pane; /vsr <app> on|off turns an app on or off, /vsr turns on|off routes your own turns too, /vsr <redact|shield|guard> wait 1|2|3|done sets how long it waits',
    argumentHint: '[<app> on|off | turns [on|off] | <redact|shield|guard> wait 1|2|3|done]',
  })
}

// Changes one of vSR's options as the /config menu would; the plugin reloads with it. Says why in a toast when it
// cannot, and returns whether it did.
export async function setOption($: EngineInterface, field: string, value: ConfigValue) {
  let why: string | undefined
  try {
    const rows = await $.config.list()
    const row = rows.find(one => one.key === `vsr.${field}`) ?? rows.find(one => one.key.endsWith(`.${field}`))
    why = row === undefined ? 'no such setting' : (await $.config.set({ key: row.key, value })).deny
  } catch (error) {
    why = String(error instanceof Error ? error.message : error)
  }
  if (why === undefined) {
    // The person's own choice: the listener's profile leaves it alone from now on.
    await pin($, field)
    return true
  }
  $.ui.toast(`vSR: could not change ${field}: ${why}`)
  return false
}

const waitWords = (choice: WaitChoice) => (choice === 'done' ? 'waits until done' : `waits ${choice} s`)

const USAGE = `vSR: /vsr opens the pane; /vsr <app> on|off (${APP_NAMES.join(', ')}); /vsr turns [on|off]; /vsr <${WAIT_APPS.join('|')}> wait ${WAIT_CHOICES.join('|')}; /vsr follow.`

// `/vsr <app> on|off` and `/vsr <app> wait <choice>`: the option it changes and the words that say so, or undefined.
function changeOf(words: readonly string[]): { field: string; value: ConfigValue; said: string } | undefined {
  const [app = '', verb, choice] = words
  const isApp = APP_NAMES.some(one => one === app)
  if (words.length === 2 && isApp && (verb === 'on' || verb === 'off')) return { field: app, value: verb === 'on', said: `${app} ${verb}` }
  const waitApp = WAIT_APPS.find(one => one === app)
  const waitChoice = WAIT_CHOICES.find(one => one === choice)
  if (words.length === 3 && waitApp && verb === 'wait' && waitChoice) {
    return { field: `${waitApp}Wait`, value: waitChoice, said: `${waitApp} ${waitWords(waitChoice)}` }
  }
  return undefined
}

// `/vsr turns on|off`. On, it also hands back turns paused by a /model pick; it needs the router app on.
// A change the person makes: saved like one made in /config, pinned, and applied over the listener's profile now,
// since an option that already holds the value saves nothing and reloads nothing.
async function change($: EngineInterface, settings: Settings, options: PluginOptions, field: string, value: ConfigValue) {
  if (!(await setOption($, field, value))) return false
  await follow($, settings, options)
  return true
}

async function turnsCommand($: EngineInterface, settings: Settings, options: PluginOptions, isOn: boolean) {
  if (isOn && !settings.apps.router) return 'vSR: turn on the router first: /vsr router on.'
  if (settings.routerTurns !== isOn && !(await change($, settings, options, 'routerTurns', isOn))) return 'vSR: turns not changed.'
  if (isOn) await resumeTurns($, settings)
  return isOn ? 'vSR: the router picks the model of your turns from now on.' : 'vSR: your turns run on the model you picked.'
}

// What the pane's controls do: each press changes one option.
function controlsFor($: EngineInterface, settings: Settings, options: PluginOptions): Controls {
  return {
    toggle: app => void change($, settings, options, app, !settings.apps[app]),
    wait: (app: WaitApp, choice: WaitChoice) => {
      if (choice !== choiceOf(settings.waits[app])) void change($, settings, options, `${app}Wait`, choice)
    },
  }
}

// The apps that stop checking while the decision model is offline (guard's fixed rules still hold).
const OFFLINE_APPS: readonly App[] = ['guard', 'redact', 'shield', 'check']

// Which apps stop checking while the model is offline, and which hold instead (set to wait until done).
function offlineOf(settings: Settings) {
  const on = OFFLINE_APPS.filter(app => settings.apps[app])
  const isHolding = (app: App) => (app === 'redact' || app === 'shield' || app === 'guard') && settings.waits[app] === 'done'
  return { notChecking: on.filter(app => !isHolding(app)), holding: on.filter(isHolding) }
}

export function registerView(on: On, settings: Settings, options: PluginOptions) {
  // Blocks until the session start loads the logo; a reload starts over here.
  logo = settings.logo === 'off' ? NO_LOGO : BLOCKS_LOGO

  // `/vsr calibrated` is what the calibration skill runs once it applied new thresholds.
  on('command.run', { command: 'vsr' }, async ($, e) => {
    const words = e.args.trim().toLowerCase().split(/\s+/).filter(Boolean)
    if (words.length === 1 && words[0] === 'calibrated') {
      const at = await stamp($)
      await markCalibrated($, at)
      return { text: `vSR: calibrated at ${at}.` }
    }
    if (words.length === 0) {
      await $.ui.open({ id: PANE, title: PANE_TITLE })
      return { text: 'vSR pane opened.' }
    }
    if (words.length === 1 && words[0] === 'follow') {
      await unpinAll($)
      const changed = await follow($, settings, options)
      return { text: `vSR: follows the listener's profile${changed.length === 0 ? '' : `: ${changed.join(', ')}`}.` }
    }
    if (words.length === 1 && words[0] === 'turns') {
      return { text: `vSR: your turns ${turnsWords({ settings, turns: await readTurns($) }) ?? 'run on the model you picked: the router app is off · /vsr router on'}.` }
    }
    if (words.length === 2 && words[0] === 'turns' && (words[1] === 'on' || words[1] === 'off')) return { text: await turnsCommand($, settings, options, words[1] === 'on') }
    const asked = changeOf(words)
    if (asked === undefined) return { text: USAGE }
    return { text: (await change($, settings, options, asked.field, asked.value)) ? `vSR: ${asked.said}.` : `vSR: ${asked.said.split(' ')[0]} not changed.` }
  })

  // A vSR row the person changes in the /config menu is theirs too.
  on('config.set', async ($, e, next) => {
    const done = await next(e)
    if (e.origin?.kind === 'composer' && e.key.startsWith('vsr.') && done.deny === undefined) {
      await pin($, e.key.slice('vsr.'.length))
      await follow($, settings, options)
    }
    return done
  }).catch(($, e, next) => next(e))

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
    if (found.length === 0) return next(e)
    return under($.ui.resolve(e), await next(e), found, inlineColumns(e), { all, display: settings.display })
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
    return band($.ui.resolve(e), list, e.props.bodyColumns, isOffline, e.props.maxRows, offlineOf(settings))
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const elements = $.ui.resolve(e)
    const act = controlsFor($, settings, options)
    const list = await readDecisions($)
    const given = await readLabels($)
    const about = {
      settings,
      calibration: await readCalibration($),
      isOffline: await readOffline($),
      spend: await readSpend($),
      turns: await readTurns($),
      labeled: Object.keys(given).length,
    }
    if (list.length === 0) {
      const { Box, Text } = elements
      return (
        <Box flexDirection="column">
          {wordmark(elements, logo)}
          {controls(elements, settings, act)}
          <Text dimColor>No decisions yet. Each one appears under the row it decided on.</Text>
          {overview(elements, about)}
        </Box>
      )
    }
    // Answers given here are labels /vsr:calibrate fits the thresholds to; with the log off nothing keeps them.
    const labeling =
      (await labelsFile($, settings.log)) === undefined ? undefined : { given, answer: (d: Decision, answer: Answer) => void giveLabel($, settings.log, d, answer) }
    return dashboard(elements, logo, list, e.props.bodyColumns, about, act, labeling)
  })
}
