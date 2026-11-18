import { Layers } from 'lucide-react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, cn } from '../ui'
import type { RoutingScope } from '../utils/routingScopes'
import { routingScopeDetail, routingScopeSummary } from './routingScopeSelectorSupport'

interface RoutingScopeSelectorProps {
  label?: string
  onChange: (scopeId: string) => void
  scopes: RoutingScope[]
  value: string
  /** Repeat the label inside the trigger; off where a visible heading already names the control. */
  showLabel?: boolean
  className?: string
}

/** Recipe scope picker: the recipe's name in the trigger, its routing counts under each option. */
export default function RoutingScopeSelector({
  label = 'Recipe',
  onChange,
  scopes,
  value,
  showLabel = true,
  className,
}: RoutingScopeSelectorProps) {
  const selected = scopes.find((scope) => scope.id === value) ?? scopes[0]
  if (!selected) return null

  return (
    <Select value={selected.id} onValueChange={onChange}>
      <SelectTrigger
        aria-label={label}
        title={routingScopeDetail(selected)}
        className={cn('min-w-[200px] max-w-[22rem]', className)}
      >
        <Layers className="size-4 text-chart-1" aria-hidden="true" />
        {showLabel ? <span className="shrink-0 text-subtle-foreground">{label}</span> : null}
        <span className="min-w-0 flex-1 truncate text-left font-medium text-foreground">
          <SelectValue>{selected.label}</SelectValue>
        </span>
      </SelectTrigger>
      <SelectContent align="start" className="w-auto max-w-[min(24rem,calc(100vw-2rem))]">
        {scopes.map((scope) => (
          <SelectItem
            key={scope.id}
            value={scope.id}
            textValue={scope.label}
            className="items-start py-2 [&>span:first-child]:top-2.5"
          >
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="truncate font-medium">{scope.label}</span>
              <span className="truncate text-xs text-muted-foreground">
                {routingScopeSummary(scope)}
              </span>
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
