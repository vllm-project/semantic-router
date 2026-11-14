import type { Listener, ListenerAPI, VSRProfile } from '../types/config'
import type { ConfigData } from './configPageSupport'

export type VSRSwitch = Exclude<keyof VSRProfile, 'router_goal'>
export type VSRGoal = NonNullable<VSRProfile['router_goal']>

// What the vSR agent plugins do when the listener leaves a field out.
export const VSR_PLUGIN_DEFAULTS: Required<VSRProfile> = {
  redact: true,
  shield: true,
  guard: true,
  check: false,
  escalate: true,
  router: true,
  router_turns: false,
  router_goal: 'balanced',
}

export interface VSRSwitchInfo {
  key: VSRSwitch
  label: string
  hint: string
}

// The apps in the order the panel shows them, grouped by the plugins that read
// them: the Codex plugin reads guard, redact and shield only.
export const VSR_GROUPS: Array<{ title: string; agents: string; switches: VSRSwitchInfo[] }> = [
  {
    title: 'Protect',
    agents: 'Claude Code and Codex',
    switches: [
      {
        key: 'redact',
        label: 'Hide personal data',
        hint: 'Replaces personal data in tool output with tags before the agent reads it.',
      },
      {
        key: 'shield',
        label: 'Shield prompt injection',
        hint: 'Tells the agent to treat output that tries to give it orders as data.',
      },
      {
        key: 'guard',
        label: 'Guard risky commands',
        hint: 'Stops a risky shell command for the person: Claude Code asks first, Codex leaves it to them.',
      },
    ],
  },
  {
    title: 'Check and route',
    agents: 'Claude Code',
    switches: [
      {
        key: 'check',
        label: 'Check answers',
        hint: 'Marks an answer that is not backed by what the agent read that turn.',
      },
      {
        key: 'escalate',
        label: 'Escalate after pushback',
        hint: 'After pushback, moves the next subagent, or a routed turn, one model up.',
      },
      {
        key: 'router',
        label: 'Route subagents',
        hint: 'Gives each general-purpose subagent a model and effort that fit its task.',
      },
      {
        key: 'router_turns',
        label: 'Route your own turns',
        hint: 'Picks the model for each prompt the person sends as well.',
      },
    ],
  },
]

export const VSR_SWITCHES: VSRSwitchInfo[] = VSR_GROUPS.flatMap((group) => group.switches)

export const VSR_GOALS: VSRGoal[] = ['quality', 'balanced', 'save']

export const VSR_GOAL_HINT =
  'Quality never picks a smaller model. Save picks one whenever it can answer.'

/** The profile with one field set, or left to the plugin when value is
 * undefined; a profile with nothing left is undefined. */
export function withVSRField<K extends keyof VSRProfile>(
  vsr: VSRProfile | undefined,
  key: K,
  value: VSRProfile[K] | undefined,
): VSRProfile | undefined {
  const next: VSRProfile = { ...vsr, [key]: value }
  if (value === undefined) delete next[key]
  return Object.keys(next).length > 0 ? next : undefined
}

/** The listeners that serve the systemone API, with their index. */
export function systemOneListeners(
  config: ConfigData | null,
): Array<{ index: number; listener: Listener; vsr: VSRProfile | undefined }> {
  return (config?.listeners ?? []).flatMap((listener, index) => {
    const systemone = listener.apis?.systemone
    return systemone ? [{ index, listener, vsr: systemone.vsr }] : []
  })
}

/** The config with one listener's systemone vsr profile replaced, or removed
 * when vsr is undefined. Every other field and API is kept as it is. */
export function withListenerVSR(
  config: ConfigData,
  index: number,
  vsr: VSRProfile | undefined,
): ConfigData {
  const listeners = (config.listeners ?? []).map((listener, listenerIndex) => {
    const systemone = listener.apis?.systemone
    if (listenerIndex !== index || !systemone) return listener
    const next: ListenerAPI = { ...systemone, vsr }
    if (!vsr) delete next.vsr
    return { ...listener, apis: { ...listener.apis, systemone: next } }
  })
  return { ...config, listeners }
}
