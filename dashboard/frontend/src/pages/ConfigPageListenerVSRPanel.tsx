import { useState } from 'react'

import type { VSRProfile } from '../types/config'
import styles from './ConfigPageRouterConfigSection.module.css'
import {
  VSR_GOALS,
  VSR_PLUGIN_DEFAULTS,
  VSR_SWITCHES,
  systemOneListeners,
  withListenerVSR,
  type VSRGoal,
} from './configPageListenerVSRSupport'
import type { ConfigData } from './configPageSupport'

interface ListenerVSRControlsProps {
  vsr: VSRProfile | undefined
  disabled: boolean
  onChange: (vsr: VSRProfile | undefined) => void
}

// A field left out of vsr shows the plugin's default, marked as one. A change
// writes only that field, so a field nobody touched stays with the plugin.
export function ListenerVSRControls({ vsr, disabled, onChange }: ListenerVSRControlsProps) {
  const shown = { ...VSR_PLUGIN_DEFAULTS, ...vsr }
  const defaultHint = (key: keyof VSRProfile) =>
    vsr?.[key] === undefined ? <span className={styles.vsrDefault}>default</span> : null
  return (
    <div className={styles.summaryGrid}>
      {VSR_SWITCHES.map(({ key, label }) => (
        <label key={key} className={`${styles.summaryItem} ${styles.vsrOption}`}>
          <input
            type="checkbox"
            checked={shown[key]}
            disabled={disabled}
            onChange={(event) => onChange({ ...vsr, [key]: event.target.checked })}
          />
          <span>{label}</span>
          {defaultHint(key)}
        </label>
      ))}
      <label className={`${styles.summaryItem} ${styles.vsrOption}`}>
        <span>Router goal</span>
        {defaultHint('router_goal')}
        <select
          value={shown.router_goal}
          disabled={disabled}
          onChange={(event) => onChange({ ...vsr, router_goal: event.target.value as VSRGoal })}
        >
          {VSR_GOALS.map((goal) => (
            <option key={goal} value={goal}>
              {goal}
            </option>
          ))}
        </select>
      </label>
      {Object.keys(vsr ?? {}).some((key) => key !== 'rules') ? (
        <button
          type="button"
          className={styles.editButton}
          disabled={disabled}
          // Clear leaves the switches to the plugins; the listener's rules stay.
          onClick={() => onChange(vsr?.rules ? { rules: vsr.rules } : undefined)}
        >
          Clear
        </button>
      ) : null}
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
            What the Claude Code and Codex plugins do when they connect to a System One listener. A
            field the listener leaves out keeps the plugin&apos;s own option.
          </p>
        </div>
      </header>
      <div className={styles.sectionList}>
        {error ? <div className={styles.rawError}>{error}</div> : null}
        {listeners.map(({ index, listener, vsr }) => (
          <article key={index} className={styles.systemRow}>
            <h4 className={styles.cardTitle}>{`${listener.name} · port ${listener.port}`}</h4>
            <ListenerVSRControls
              vsr={vsr}
              disabled={isReadonly || saving}
              onChange={(next) => void save(index, next)}
            />
            <code
              className={styles.sectionKey}
            >{`listeners.${listener.name}.apis.systemone.vsr`}</code>
          </article>
        ))}
      </div>
    </section>
  )
}
