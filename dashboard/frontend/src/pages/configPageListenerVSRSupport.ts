import type { Listener, ListenerAPI, VSRProfile } from '../types/config'
import type { ConfigData } from './configPageSupport'

export type VSRSwitch = Exclude<keyof VSRProfile, 'router_goal' | 'rules'>
export type VSRGoal = NonNullable<VSRProfile['router_goal']>

// What the vSR agent plugins do when the listener leaves a field out.
export const VSR_PLUGIN_DEFAULTS: Required<Omit<VSRProfile, 'rules'>> = {
  redact: true,
  shield: true,
  guard: true,
  check: false,
  escalate: true,
  router: true,
  router_turns: false,
  router_goal: 'balanced',
}

export const VSR_SWITCHES: Array<{ key: VSRSwitch; label: string }> = [
  { key: 'redact', label: 'Hide personal data (PII)' },
  { key: 'shield', label: 'Shield prompt injection' },
  { key: 'guard', label: 'Guard risky commands' },
  { key: 'check', label: 'Check answers' },
  { key: 'escalate', label: 'Escalate after pushback' },
  { key: 'router', label: 'Route subagents' },
  { key: 'router_turns', label: 'Route your own turns' },
]

export const VSR_GOALS: VSRGoal[] = ['quality', 'balanced', 'save']

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
