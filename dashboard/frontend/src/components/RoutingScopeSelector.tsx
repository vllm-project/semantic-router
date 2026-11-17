import { useId } from 'react'
import { ChevronDown, Layers } from 'lucide-react'

import { countSignalsInProfile, type RoutingScope } from '../utils/routingScopes'

interface RoutingScopeSelectorProps {
  label?: string
  onChange: (scopeId: string) => void
  scopes: RoutingScope[]
  value: string
}

const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`

function routingScopeOptionLabel(scope: RoutingScope): string {
  const decisions = Array.isArray(scope.routing?.decisions) ? scope.routing.decisions.length : 0
  const signals = countSignalsInProfile(scope.routing).total
  return [scope.label, plural(decisions, 'decision'), plural(signals, 'signal')].join(' · ')
}

/**
 * Recipe scope picker drawn like the design-system Select trigger. A native
 * <select> covers the trigger so keyboard, screen reader and form automation
 * all operate on the platform control.
 */
export default function RoutingScopeSelector({
  label = 'Routing profile',
  onChange,
  scopes,
  value,
}: RoutingScopeSelectorProps) {
  const selectId = useId()
  const selected = scopes.find((scope) => scope.id === value) ?? scopes[0]
  if (!selected) return null
  const detail =
    selected.entrypointModelNames.length > 0
      ? selected.entrypointModelNames.join(', ')
      : selected.description || 'Top-level automatic routing'

  return (
    <div
      title={detail}
      className="relative inline-flex h-9 min-w-[200px] max-w-[22rem] items-center gap-2 rounded-md border border-input bg-surface px-3 text-[13px] shadow-xs transition-[border-color,box-shadow] hover:border-border-strong has-[select:disabled]:opacity-50 has-[select:focus-visible]:border-ring has-[select:focus-visible]:ring-[3px] has-[select:focus-visible]:ring-ring/40"
    >
      <Layers className="size-4 shrink-0 text-chart-1" aria-hidden="true" />
      <span className="shrink-0 text-subtle-foreground" aria-hidden="true">
        Recipe
      </span>
      <span className="min-w-0 flex-1 truncate font-medium text-foreground" aria-hidden="true">
        {selected.label}
      </span>
      <ChevronDown className="size-4 shrink-0 text-subtle-foreground" aria-hidden="true" />
      <select
        id={selectId}
        aria-label={label}
        value={selected.id}
        onChange={(event) => onChange(event.target.value)}
        className="absolute inset-0 size-full cursor-pointer appearance-none rounded-md opacity-0 outline-none [&>option]:bg-popover [&>option]:text-popover-foreground"
      >
        {scopes.map((scope) => (
          <option key={scope.id} value={scope.id}>
            {routingScopeOptionLabel(scope)}
          </option>
        ))}
      </select>
    </div>
  )
}
