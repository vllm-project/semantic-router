import { useState } from 'react'
import { Bot, RotateCcw } from 'lucide-react'

import type { VSRProfile } from '../types/config'
import { Alert, Button, Card, CardDescription, CardTitle, IconTile, cn } from '../ui'
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
  const set =
    <K extends keyof VSRProfile>(key: K, value: VSRProfile[K] | undefined) =>
    () =>
      onChange(withVSRField(vsr, key, value))
  return (
    <div className="grid grid-cols-1 gap-3 2xl:grid-cols-2">
      {VSR_GROUPS.map((group) => (
        <section
          key={group.title}
          className="rounded-lg border border-border bg-surface-subtle/40 px-3.5 dark:bg-surface-subtle/30"
          aria-label={group.title}
        >
          <header className="flex items-center justify-between gap-3 border-b border-border py-2.5">
            <span className="text-[13px] font-semibold">{group.title}</span>
            <span className="text-xs text-subtle-foreground">{group.agents}</span>
          </header>
          <div className="divide-y divide-border">
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
          </div>
        </section>
      ))}
    </div>
  )
}

interface ConfigPageListenerVSRPanelProps {
  config: ConfigData | null
  isReadonly: boolean
  saveConfig: (config: ConfigData) => Promise<void>
  /** Why the controls are off, shown above them. */
  notice?: string
  highlight?: boolean
}

// The vSR agent plugin profile of every listener that serves systemone.
export default function ConfigPageListenerVSRPanel({
  config,
  isReadonly,
  saveConfig,
  notice,
  highlight = false,
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
    <Card
      id="global-layer-vsr"
      className={cn(
        'scroll-mt-6 transition-shadow duration-700',
        highlight && 'shadow-[0_0_0_3px_var(--ring)]',
      )}
    >
      <div className="flex items-start gap-3 px-5 pb-4 pt-4">
        <IconTile icon={Bot} tone="slate" size="sm" className="mt-0.5" />
        <div className="flex min-w-0 flex-col gap-1">
          <CardTitle>vSR agent plugins</CardTitle>
          <CardDescription className="leading-relaxed">
            What the Claude Code and Codex plugins do when they connect to a System One listener.
            Default leaves the choice to each plugin. On and Off set the listener&apos;s choice,
            which a person&apos;s own setting in their plugin still overrides. Changes here save
            right away.
          </CardDescription>
        </div>
      </div>
      <div className="flex flex-col gap-5 border-t border-border px-5 pb-5 pt-5">
        {notice ? <Alert tone="info">{notice}</Alert> : null}
        {error ? (
          <Alert tone="danger" title="The vSR profile was not saved">
            {error}
          </Alert>
        ) : null}
        {listeners.map(({ index, listener, vsr }) => {
          // The listener's rules are not a plugin setting: a reset keeps them.
          const set = Object.keys(vsr ?? {}).filter((key) => key !== 'rules').length
          const bound = (vsr?.rules?.guard?.length ?? 0) + (vsr?.rules?.redact?.length ?? 0)
          const state = [set === 0 ? 'All plugin defaults' : `${set} set here`]
          if (bound > 0) state.push(`${bound} ${bound === 1 ? 'rule' : 'rules'} bound`)
          return (
            <article key={index} className="flex flex-col gap-3">
              <header className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <h4 className="font-mono text-[13px] font-semibold">{`${listener.name} · port ${listener.port}`}</h4>
                <span className="text-xs text-muted-foreground">
                  {saving ? 'Saving…' : state.join(' · ')}
                </span>
                {set > 0 ? (
                  <Button
                    variant="ghost"
                    size="xs"
                    className="ml-auto"
                    disabled={isReadonly || saving}
                    onClick={() => void save(index, vsr?.rules ? { rules: vsr.rules } : undefined)}
                  >
                    <RotateCcw />
                    Reset to plugin defaults
                  </Button>
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
              <code className="font-mono text-[11px] text-subtle-foreground">
                {`Saved to listeners.${listener.name}.apis.systemone.vsr`}
              </code>
            </article>
          )
        })}
      </div>
    </Card>
  )
}
