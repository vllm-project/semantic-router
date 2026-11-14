import type { Listener, ListenerAPI, SystemOneSignal, VSRProfile, VSRRules } from '../types/config'
import type { ConfigData, ConfigSignals, KeywordSignal } from './configPageSupport'

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

export type VSRRuleApp = 'guard' | 'redact'

/** A rule of the recipe the panel can bind to an app, as the Router checks at
 * load: guard takes a rule that says whether a command matches, redact one
 * that finds text. */
export interface VSRRuleChoice {
  type: 'keyword' | 'systemone'
  name: string
  description: string
}

const DEFAULT_RECIPE = 'default'

/** The recipes the listener can bind rules of: the flat routing as the default
 * recipe, then the named ones. */
export function vsrRuleRecipes(config: ConfigData | null): string[] {
  const named = (config?.recipes ?? []).map((recipe) => recipe.name)
  const flat = config?.default_recipe?.name ?? DEFAULT_RECIPE
  return config?.routing?.signals && !named.includes(flat) ? [flat, ...named] : named
}

function recipeSignals(config: ConfigData | null, recipe: string): ConfigSignals | undefined {
  const named = config?.recipes?.find((one) => one.name === recipe)
  if (named) return named.routing?.signals
  return recipe === (config?.default_recipe?.name ?? DEFAULT_RECIPE) ? config?.routing?.signals : undefined
}

// As the Router reads it (config.KeywordRuleFindsText): an OR rule of literal
// keywords or method regex, in any case, with no fuzzy match, count or density.
const findsText = (rule: KeywordSignal) =>
  rule.operator === 'OR' &&
  (!rule.method || rule.method.toLowerCase() === 'regex') &&
  !rule.fuzzy_match &&
  !rule.count &&
  !rule.density

const fitsApp = (rule: SystemOneSignal, app: VSRRuleApp) =>
  (rule.stage ?? 'request') === 'request' &&
  (app === 'redact'
    ? rule.question.type === 'span'
    : rule.question.type === 'noul' || (rule.question.type === 'choice' && (rule.labels?.length ?? 0) > 0))

/** The most rules an app binds, as the Router allows (MaxVSRRulesPerApp). */
export const VSR_RULES_PER_APP = 16

/** The rules of a recipe that fit an app, then any rule the listener binds
 * that the recipe no longer offers, so it can still be let go. */
export function vsrRuleChoices(
  config: ConfigData | null,
  recipe: string,
  app: VSRRuleApp,
  vsr?: VSRProfile,
): VSRRuleChoice[] {
  const fitting = fittingRules(config, recipe, app)
  const missing = (vsr?.rules?.[app] ?? [])
    .filter((rule) => !fitting.some((one) => one.type === rule.type && one.name === rule.name))
    .map((rule) => ({ type: rule.type, name: rule.name, description: `not a ${app} rule of ${recipe}; the Router refuses it until it is let go` }))
  return [...fitting, ...missing]
}

/** Why a choice cannot be bound now: the app is full, or another bound rule
 * has its name (the plugins name a rule by its name alone). */
export function vsrRuleBlocked(vsr: VSRProfile | undefined, app: VSRRuleApp, choice: VSRRuleChoice): string | undefined {
  if (vsrRuleBinding(vsr, app, choice) !== 'off') return undefined
  const bound = vsr?.rules?.[app] ?? []
  if (bound.some((rule) => rule.name === choice.name)) return `another rule named ${choice.name} is bound`
  if (bound.length >= VSR_RULES_PER_APP) return `${app} binds ${VSR_RULES_PER_APP} rules at most`
  return undefined
}

function fittingRules(config: ConfigData | null, recipe: string, app: VSRRuleApp): VSRRuleChoice[] {
  const signals = recipeSignals(config, recipe)
  return [
    ...(signals?.keywords ?? [])
      .filter((rule) => app === 'guard' || findsText(rule))
      .map((rule) => ({ type: 'keyword' as const, name: rule.name, description: rule.description ?? '' })),
    ...(signals?.systemone ?? [])
      .filter((rule) => fitsApp(rule, app))
      .map((rule) => ({ type: 'systemone' as const, name: rule.name, description: rule.description ?? '' })),
  ]
}

export type VSRRuleBinding = 'off' | 'advisory' | 'required'

/** How a rule is bound to an app now. */
export function vsrRuleBinding(vsr: VSRProfile | undefined, app: VSRRuleApp, choice: VSRRuleChoice): VSRRuleBinding {
  const bound = vsr?.rules?.[app]?.find((rule) => rule.type === choice.type && rule.name === choice.name)
  if (!bound) return 'off'
  return bound.enforce === 'required' ? 'required' : 'advisory'
}

/** The profile with one rule bound, rebound or let go; a profile whose rules
 * bind nothing keeps no rules. */
export function withVSRRule(
  vsr: VSRProfile | undefined,
  recipe: string,
  app: VSRRuleApp,
  choice: VSRRuleChoice,
  binding: VSRRuleBinding,
): VSRProfile | undefined {
  const rules: VSRRules = { recipe, guard: vsr?.rules?.guard, redact: vsr?.rules?.redact }
  const others = (rules[app] ?? []).filter((rule) => !(rule.type === choice.type && rule.name === choice.name))
  const bound =
    binding === 'off'
      ? others
      : [...others, { type: choice.type, name: choice.name, ...(binding === 'required' ? { enforce: 'required' as const } : {}) }]
  rules[app] = bound.length > 0 ? bound : undefined
  if (!rules.guard) delete rules.guard
  if (!rules.redact) delete rules.redact
  return withVSRField(vsr, 'rules', rules.guard || rules.redact ? rules : undefined)
}

/** The profile bound to another recipe, with no rule of the old one. */
export function withVSRRulesRecipe(vsr: VSRProfile | undefined, recipe: string): VSRProfile | undefined {
  return withVSRField(vsr, 'rules', recipe === '' ? undefined : { recipe })
}

/** What a Try it answer says, in words: the guard rules the sample matches,
 * or the text the redact rules would hide, and the rules left unchecked. */
export function describeTry(app: VSRRuleApp, text: string, answer: unknown): string {
  const part = (answer as { parts?: unknown[] } | null)?.parts?.[0] as
    | { matches?: string[]; spans?: Array<{ start: number; end: number; rule: string }>; unchecked?: Array<{ rule: string; reason: string }> }
    | undefined
  if (!part) return 'The Router gave no answer for the sample.'
  const points = Array.from(text)
  const found =
    app === 'guard'
      ? (part.matches ?? []).length > 0
        ? `Held by ${(part.matches ?? []).join(', ')}.`
        : 'No rule holds it.'
      : (part.spans ?? []).length > 0
        ? `Hides ${(part.spans ?? []).map((span) => `"${points.slice(span.start, span.end).join('')}" (${span.rule})`).join(', ')}.`
        : 'No rule finds anything to hide.'
  const unchecked = (part.unchecked ?? []).map((one) => `${one.rule} (${one.reason})`)
  return unchecked.length > 0 ? `${found} Not checked: ${unchecked.join(', ')}.` : found
}
