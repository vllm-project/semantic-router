import { countSignalsInProfile, type RoutingScope } from '../utils/routingScopes'

const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`

/** What a recipe routes with, such as "14 decisions · 18 signals". */
export function routingScopeSummary(scope: RoutingScope): string {
  const decisions = Array.isArray(scope.routing?.decisions) ? scope.routing.decisions.length : 0
  const signals = countSignalsInProfile(scope.routing).total
  return `${plural(decisions, 'decision')} · ${plural(signals, 'signal')}`
}

/** The trigger's hover text: the recipe's public model names, else what it is for. */
export function routingScopeDetail(scope: RoutingScope): string {
  return scope.entrypointModelNames.length > 0
    ? scope.entrypointModelNames.join(', ')
    : scope.description || 'Top-level automatic routing'
}
