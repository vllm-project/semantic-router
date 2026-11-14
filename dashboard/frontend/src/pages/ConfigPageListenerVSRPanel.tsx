import { useState } from 'react'

import type { VSRProfile } from '../types/config'
import styles from './ConfigPageRouterConfigSection.module.css'
import { Choices, Row } from './ConfigPageListenerVSRChoices'
import { ListenerVSRRules } from './ConfigPageListenerVSRRules'
import {
  VSR_GOAL_HINT,
  VSR_GOALS,
  VSR_GROUPS,
  VSR_PLUGIN_DEFAULTS,
  systemOneListeners,
  withListenerVSR,
  withVSRField,
} from './configPageListenerVSRSupport'
import type { ConfigData } from './configPageSupport'

interface ListenerVSRControlsProps {
  vsr: VSRProfile | undefined
  disabled: boolean
  onChange: (vsr: VSRProfile | undefined) => void
}

const onOff = (on: boolean) => (on ? 'on' : 'off')
const title = (goal: string) => goal[0].toUpperCase() + goal.slice(1)

// Each app is Default, On or Off. Default leaves the field out, so the plugin
// keeps its own option; On and Off write only that field.
export function ListenerVSRControls({ vsr, disabled, onChange }: ListenerVSRControlsProps) {
  const set = <K extends keyof VSRProfile>(key: K, value: VSRProfile[K] | undefined) => () =>
    onChange(withVSRField(vsr, key, value))
  return (
    <div className={styles.vsrGroups}>
      {VSR_GROUPS.map((group) => (
        <section key={group.title} className={styles.vsrGroup} aria-label={group.title}>
          <header className={styles.vsrGroupHead}>
            <span className={styles.vsrGroupTitle}>{group.title}</span>
            <span className={styles.vsrGroupAgents}>{group.agents}</span>
          </header>
          {group.switches.map(({ key, label, hint }) => (
            <Row key={key} label={label} hint={hint}>
              <Choices
                label={label}
                disabled={disabled}
                choices={[
                  {
                    text: 'Default',
                    note: onOff(VSR_PLUGIN_DEFAULTS[key]),
                    pressed: vsr?.[key] === undefined,
                    pick: set(key, undefined),
                  },
                  { text: 'On', pressed: vsr?.[key] === true, pick: set(key, true) },
                  { text: 'Off', pressed: vsr?.[key] === false, pick: set(key, false) },
                ]}
              />
            </Row>
          ))}
          {group.agents === 'Claude Code' ? (
            <Row label="Router goal" hint={VSR_GOAL_HINT}>
              <Choices
                label="Router goal"
                disabled={disabled}
                choices={[
                  {
                    text: 'Default',
                    note: VSR_PLUGIN_DEFAULTS.router_goal,
                    pressed: vsr?.router_goal === undefined,
                    pick: set('router_goal', undefined),
                  },
                  ...VSR_GOALS.map((goal) => ({
                    text: title(goal),
                    pressed: vsr?.router_goal === goal,
                    pick: set('router_goal', goal),
                  })),
                ]}
              />
            </Row>
          ) : null}
        </section>
      ))}
    </div>
  )
}

interface ConfigPageListenerVSRPanelProps {
  config: ConfigData | null
  isReadonly: boolean
  saveConfig: (config: ConfigData) => Promise<void>
}

// The vSR agent plugin profile of every listener that serves systemone.
export default function ConfigPageListenerVSRPanel({
  config,
  isReadonly,
  saveConfig,
}: ConfigPageListenerVSRPanelProps) {
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listeners = systemOneListeners(config)
  if (!config || listeners.length === 0) return null

  const save = async (index: number, vsr: VSRProfile | undefined) => {
    setSaving(true)
    setError(null)
    try {
      await saveConfig(withListenerVSR(config, index, vsr))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save the vSR profile')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section id="global-layer-vsr" className={styles.sectionGroup}>
      <header className={styles.groupHeader}>
        <div>
          <h3 className={styles.groupTitle}>vSR agent plugins</h3>
          <p className={styles.groupDescription}>
            What the Claude Code and Codex plugins do when they connect to a System One listener.
            Default leaves the choice to each plugin. On and Off set the listener&apos;s choice,
            which a person&apos;s own setting in their plugin still overrides.
          </p>
        </div>
      </header>
      <div className={styles.sectionList}>
        {error ? <div className={styles.rawError}>{error}</div> : null}
        {listeners.map(({ index, listener, vsr }) => {
          // The listener's rules are not a plugin setting: a reset keeps them.
          const set = Object.keys(vsr ?? {}).filter((key) => key !== 'rules').length
          const bound = (vsr?.rules?.guard?.length ?? 0) + (vsr?.rules?.redact?.length ?? 0)
          const state = [set === 0 ? 'All plugin defaults' : `${set} set here`]
          if (bound > 0) state.push(`${bound} ${bound === 1 ? 'rule' : 'rules'} bound`)
          return (
            <article key={index} className={styles.vsrListener}>
              <header className={styles.vsrListenerHead}>
                <h4 className={styles.cardTitle}>{`${listener.name} · port ${listener.port}`}</h4>
                <span className={styles.vsrListenerState}>
                  {saving ? 'Saving…' : state.join(' · ')}
                </span>
                {set > 0 ? (
                  <button
                    type="button"
                    className={styles.vsrReset}
                    disabled={isReadonly || saving}
                    onClick={() => void save(index, vsr?.rules ? { rules: vsr.rules } : undefined)}
                  >
                    Reset to plugin defaults
                  </button>
                ) : null}
              </header>
              <ListenerVSRControls
                vsr={vsr}
                disabled={isReadonly || saving}
                onChange={(next) => void save(index, next)}
              />
              <ListenerVSRRules
                config={config}
                listener={listener}
                vsr={vsr}
                disabled={isReadonly || saving}
                onChange={(next) => void save(index, next)}
              />
              <code className={styles.sectionKey}>
                {`Saved to listeners.${listener.name}.apis.systemone.vsr`}
              </code>
            </article>
          )
        })}
      </div>
    </section>
  )
}
