import type { Outcome } from '../types'
import type { EngineInterface, Register, ToolCallInput, ToolCallResult } from 'claude-code'

import { checkTurn, noteRead, startTurnReads } from './check'
import { escalatePrompt, holdEscalation, releaseEscalation } from './escalate'
import { checkGuardModel, guardCheck, isRefusal, settleHeld } from './guard'
import { checkRuntime } from './lib/classify'
import { loadCalibration } from './lib/log'
import { APP_NAMES } from './lib/apps'
import { parseSettings, type Settings } from './lib/settings'
import { inspectOutput, writeBackOf } from './output'
import { checkRouterModel, countSpend, routeSpawn, routeStep } from './router'
import { askTurn, endTurn, showTurns, startTurn, stepTurn } from './turns'
import { loadLook, registerCommand, registerView } from './view'

// A tool's result as Claude gets it: redacted and shielded, then kept for check.
async function inspected($: EngineInterface, e: ToolCallInput, ran: ToolCallResult, settings: Settings) {
  const given = await inspectOutput($, e, ran, settings)
  await noteRead($, e, given, settings)
  return given
}

// Each event is hooked once, here; what each application does with it lives in its own module.
export const register: Register = (on, options) => {
  const settings = parseSettings(options)

  on('session.start', async ($, e, next) => {
    await registerCommand($)
    await loadLook($, settings)
    await loadCalibration($, settings.log)
    // The guard's check of its model asks only a runtime that answered, and a session that ends first drops it.
    if (APP_NAMES.some(app => settings.apps[app])) {
      void checkRuntime($, settings.runtime)
        .then(isUp => (isUp && settings.apps.guard ? checkGuardModel($, settings) : undefined))
        .catch(() => undefined)
    }
    if (settings.apps.router) void checkRouterModel($, settings.runtime)
    await showTurns($, settings)
    return next(e)
  })

  on('prompt.submit', ($, e, next) => {
    askTurn($, e, settings)
    return escalatePrompt($, e, next, settings)
  })

  on('turn.start', async ($, e, next) => {
    if (e.agentId === undefined) {
      await holdEscalation($, e)
      await startTurnReads($, e)
      await startTurn($, e, settings)
    }
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    if (e.agentId === undefined) {
      await releaseEscalation($, e)
      await endTurn($, e)
      await checkTurn($, e, settings)
    } else {
      await countSpend($, e)
    }
    return done
  })

  on('agent.spawn', ($, e, next) => routeSpawn($, e, next, settings))
  on('turn.step', async function* ($, e, next) {
    return yield* e.agentId === undefined ? stepTurn($, e, next, settings) : routeStep($, e, next, settings)
  })

  on('tool.check', { tool: 'Bash' }, ($, e, next) => guardCheck($, e, next, settings)).catch(($, e, next) => next(e))

  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    let outcome: Outcome = 'declined'
    try {
      const ran = await next(e)
      outcome = isRefusal(ran) || next.signal.aborted ? 'declined' : 'approved'
      return await inspected($, e, ran, settings)
    } finally {
      await settleHeld($, e.tool_use_id, outcome, settings)
    }
  })
  on('tool.call', { tool: 'Read' }, async ($, e, next) => inspected($, e, await next(e), settings))
  on('tool.call', { tool: 'WebFetch' }, async ($, e, next) => inspectOutput($, e, await next(e), settings))
  on('tool.call', { tool: ['Write', 'Edit'] }, async ($, e, next) => {
    const refusal = await writeBackOf($, e, settings)
    return refusal === undefined ? next(e) : { deny: refusal }
  }).catch(($, e, next) => next(e))

  registerView(on, settings)
}
