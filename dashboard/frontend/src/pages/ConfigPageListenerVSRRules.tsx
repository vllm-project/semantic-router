import { useState } from 'react'

import type { Listener, VSRProfile } from '../types/config'
import styles from './ConfigPageRouterConfigSection.module.css'
import { Choices, Row } from './ConfigPageListenerVSRChoices'
import {
  describeTry,
  type VSRRuleApp,
  type VSRRuleBinding,
  vsrRuleBinding,
  vsrRuleBlocked,
  vsrRuleChoices,
  vsrRuleRecipes,
  withVSRRule,
  withVSRRulesRecipe,
} from './configPageListenerVSRSupport'
import type { ConfigData } from './configPageSupport'

const RULE_APPS: Array<{ app: VSRRuleApp; title: string; hint: string; sample: string }> = [
  {
    app: 'guard',
    title: 'Guard risky commands',
    hint: 'A command a bound rule matches is held for the person, whatever the model scores.',
    sample: 'A command to try, such as psql --host=prod-db.internal',
  },
  {
    app: 'redact',
    title: 'Hide personal data',
    hint: 'What a bound rule finds in tool output is replaced with [vsr:<rule>] before the agent reads it.',
    sample: 'Tool output to try',
  },
]

const BINDINGS: Array<{ binding: VSRRuleBinding; text: string }> = [
  { binding: 'off', text: 'Off' },
  { binding: 'advisory', text: 'Advisory' },
  { binding: 'required', text: 'Required' },
]

interface ListenerVSRRulesProps {
  config: ConfigData | null
  listener: Listener
  vsr: VSRProfile | undefined
  disabled: boolean
  onChange: (vsr: VSRProfile | undefined) => void
}

// The rules of one recipe the listener binds to guard and redact. Advisory
// steps aside when the Router cannot check it; Required holds the command or
// withholds the output instead.
export function ListenerVSRRules({ config, listener, vsr, disabled, onChange }: ListenerVSRRulesProps) {
  const recipes = vsrRuleRecipes(config)
  const recipe = vsr?.rules?.recipe ?? ''
  if (recipes.length === 0) return null
  return (
    <section className={styles.vsrGroup} aria-label="Rules of your own">
      <header className={styles.vsrGroupHead}>
        <span className={styles.vsrGroupTitle}>Rules of your own</span>
        <span className={styles.vsrGroupAgents}>Claude Code and Codex</span>
      </header>
      <Row label="Recipe" hint="Whose keyword and System One rules guard and redact apply; an entrypoint must reach it.">
        <select
          className={styles.vsrSelect}
          aria-label="Recipe of the rules"
          value={recipe}
          disabled={disabled}
          onChange={(event) => {
            const bound = (vsr?.rules?.guard?.length ?? 0) + (vsr?.rules?.redact?.length ?? 0)
            // Another recipe has other rules: the bound ones go, so ask first.
            if (bound > 0 && !window.confirm(`Bind the rules of ${event.target.value || 'no recipe'}? The ${bound} rules bound now are let go.`)) return
            onChange(withVSRRulesRecipe(vsr, event.target.value))
          }}
        >
          <option value="">None</option>
          {recipes.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </Row>
      {recipe
        ? RULE_APPS.map(({ app, title, hint, sample }) => {
            const choices = vsrRuleChoices(config, recipe, app, vsr)
            return (
              <div key={app} className={styles.vsrRuleApp}>
                <div className={styles.vsrRowText}>
                  <span className={styles.vsrRowLabel}>{title}</span>
                  <span className={styles.vsrRowHint}>{hint}</span>
                </div>
                {choices.length === 0 ? (
                  <span className={styles.vsrRowHint}>{`No rule of ${recipe} fits ${app}.`}</span>
                ) : (
                  choices.map((choice) => (
                    <Row key={`${choice.type}:${choice.name}`} label={choice.name} hint={choice.description || `${choice.type} rule`}>
                      <Choices
                        label={`${title}: ${choice.name}`}
                        disabled={disabled}
                        choices={BINDINGS.map(({ binding, text }) => ({
                          text,
                          pressed: vsrRuleBinding(vsr, app, choice) === binding,
                          pick: () => onChange(withVSRRule(vsr, recipe, app, choice, binding)),
                          blocked: binding === 'off' ? undefined : vsrRuleBlocked(vsr, app, choice),
                        }))}
                      />
                    </Row>
                  ))
                )}
                <TryRules listener={listener.name} app={app} placeholder={sample} />
              </div>
            )
          })
        : null}
    </section>
  )
}

// Asks the listener's rules about a sample as the plugins ask them, through
// the Dashboard, which adds the listener's key on the server.
function TryRules({ listener, app, placeholder }: { listener: string; app: VSRRuleApp; placeholder: string }) {
  const [text, setText] = useState('')
  const [said, setSaid] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true)
    try {
      const response = await fetch('/api/vsr/rules/try', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ listener, request: { app, parts: [{ id: 'try', text }] } }),
      })
      const answer: unknown = await response.json()
      const error = (answer as { error?: { message?: string } } | null)?.error?.message
      setSaid(response.ok ? describeTry(app, text, answer) : (error ?? `The Router answered HTTP ${response.status}.`))
    } catch (err) {
      setSaid(err instanceof Error ? err.message : 'The Router could not be reached.')
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className={styles.vsrTry}>
      <textarea
        className={styles.vsrTryText}
        aria-label={`Try the ${app} rules`}
        placeholder={placeholder}
        value={text}
        rows={2}
        onChange={(event) => setText(event.target.value)}
      />
      <button type="button" className={styles.vsrReset} disabled={busy || text.trim() === ''} onClick={() => void run()}>
        {busy ? 'Trying…' : 'Try it'}
      </button>
      <span className={styles.vsrRowHint} role="status">
        {said ?? 'Asks the rules as the Router runs them now; save a change first.'}
      </span>
    </div>
  )
}
