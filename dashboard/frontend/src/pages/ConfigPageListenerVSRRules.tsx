import { useState } from 'react'

import type { Listener, VSRProfile } from '../types/config'
import {
  Button,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Textarea,
} from '../ui'
import { Choices, Row } from './ConfigPageListenerVSRChoices'
import {
  describeTry,
  type VSRRuleApp,
  type VSRRuleBinding,
  type VSRRuleChoice,
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

const NO_RECIPE = '__no_recipe__'

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
export function ListenerVSRRules({
  config,
  listener,
  vsr,
  disabled,
  onChange,
}: ListenerVSRRulesProps) {
  const recipes = vsrRuleRecipes(config)
  const recipe = vsr?.rules?.recipe ?? ''
  if (recipes.length === 0) return null
  return (
    <section
      className="rounded-lg border border-border bg-surface-subtle/40 px-3.5 dark:bg-surface-subtle/30"
      aria-label="Rules of your own"
    >
      <header className="flex items-center justify-between gap-3 border-b border-border py-2.5">
        <span className="text-[13px] font-semibold">Rules of your own</span>
        <span className="text-xs text-subtle-foreground">Claude Code and Codex</span>
      </header>
      <div className="divide-y divide-border">
        <Row
          label="Recipe"
          hint="Whose keyword and System One rules guard and redact apply; an entrypoint must reach it."
        >
          <Select
            value={recipe || NO_RECIPE}
            disabled={disabled}
            onValueChange={(picked) => {
              const next = picked === NO_RECIPE ? '' : picked
              const bound = (vsr?.rules?.guard?.length ?? 0) + (vsr?.rules?.redact?.length ?? 0)
              // Another recipe has other rules: the bound ones go, so ask first.
              if (
                bound > 0 &&
                !window.confirm(
                  `Bind the rules of ${next || 'no recipe'}? The ${bound} rules bound now are let go.`,
                )
              )
                return
              onChange(withVSRRulesRecipe(vsr, next))
            }}
          >
            <SelectTrigger
              aria-label="Recipe of the rules"
              className="min-w-44 font-mono text-[12.5px]"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NO_RECIPE}>None</SelectItem>
              {recipes.map((name) => (
                <SelectItem key={name} value={name} className="font-mono text-[12.5px]">
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Row>
        {recipe
          ? RULE_APPS.map(({ app, title, hint, sample }) => {
              const choices = vsrRuleChoices(config, recipe, app, vsr)
              return (
                <div key={app} className="flex flex-col py-3">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <span className="text-[13px] font-medium">{title}</span>
                    <span className="text-xs leading-relaxed text-muted-foreground">{hint}</span>
                  </div>
                  {choices.length === 0 ? (
                    <span className="pt-2 text-xs text-subtle-foreground">{`No rule of ${recipe} fits ${app}.`}</span>
                  ) : (
                    <div className="divide-y divide-border">
                      <RuleAppChoices
                        app={app}
                        title={title}
                        recipe={recipe}
                        choices={choices}
                        vsr={vsr}
                        disabled={disabled}
                        onChange={onChange}
                      />
                    </div>
                  )}
                  <TryRules listener={listener.name} app={app} placeholder={sample} />
                </div>
              )
            })
          : null}
      </div>
    </section>
  )
}

// One app's choices. Once a rule is bound, the rules left Off fold behind one
// button, so a recipe with many rules keeps the panel short.
function RuleAppChoices({
  app,
  title,
  recipe,
  choices,
  vsr,
  disabled,
  onChange,
}: {
  app: VSRRuleApp
  title: string
  recipe: string
  choices: VSRRuleChoice[]
  vsr: VSRProfile | undefined
  disabled: boolean
  onChange: (vsr: VSRProfile | undefined) => void
}) {
  const [open, setOpen] = useState(false)
  const bound = choices.filter((choice) => vsrRuleBinding(vsr, app, choice) !== 'off')
  const folded = bound.length > 0 && !open ? choices.length - bound.length : 0
  const shown = folded > 0 ? bound : choices
  return (
    <>
      {shown.map((choice) => (
        <Row
          key={`${choice.type}:${choice.name}`}
          label={choice.name}
          hint={choice.description || `${choice.type} rule`}
        >
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
      ))}
      {folded > 0 || (open && bound.length > 0 && bound.length < choices.length) ? (
        <div className="py-2">
          <Button variant="ghost" size="xs" onClick={() => setOpen(!open)}>
            {folded > 0
              ? `Show ${folded} more ${folded === 1 ? 'rule' : 'rules'} of ${recipe}`
              : 'Show only the bound rules'}
          </Button>
        </div>
      ) : null}
    </>
  )
}

// Asks the listener's rules about a sample as the plugins ask them, through
// the Dashboard, which adds the listener's key on the server.
function TryRules({
  listener,
  app,
  placeholder,
}: {
  listener: string
  app: VSRRuleApp
  placeholder: string
}) {
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
      setSaid(
        response.ok
          ? describeTry(app, text, answer)
          : (error ?? `The Router answered HTTP ${response.status}.`),
      )
    } catch (err) {
      setSaid(err instanceof Error ? err.message : 'The Router could not be reached.')
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="mt-2 flex flex-col gap-2 rounded-lg border border-dashed border-border p-2.5">
      <Textarea
        aria-label={`Try the ${app} rules`}
        placeholder={placeholder}
        value={text}
        rows={2}
        onChange={(event) => setText(event.target.value)}
        className="min-h-14 font-mono text-[12.5px]"
      />
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="outline"
          size="xs"
          disabled={busy || text.trim() === ''}
          onClick={() => void run()}
        >
          {busy ? 'Trying…' : 'Try it'}
        </Button>
        <span className="text-xs text-muted-foreground" role="status">
          {said ?? 'Asks the rules as the Router runs them now; save a change first.'}
        </span>
      </div>
    </div>
  )
}
