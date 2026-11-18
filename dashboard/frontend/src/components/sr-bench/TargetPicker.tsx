import { useState } from 'react'
import { Layers, Search } from 'lucide-react'
import { Badge, Checkbox, InputGroup, ProviderLogo, cn } from '../../ui'
import { targetProvider } from './runPresentation'
import { targetLabel } from './targetPresentation'
import type { Target } from './types'

const priced = (target: Target) =>
  !!target.prices &&
  Object.keys(target.prices).length > 0 &&
  (target.kind !== 'single' || Object.prototype.hasOwnProperty.call(target.prices, target.model))

/** Configured targets as a checklist: recipes routed by the Router, then models called directly. */
export default function TargetPicker({
  targets,
  selected,
  onChange,
  unavailable,
}: {
  targets: Target[]
  selected: Target[]
  onChange: (next: Target[]) => void
  unavailable: (target: Target) => string | null
}) {
  const [query, setQuery] = useState('')
  const visible = targets.filter((target) =>
    `${target.id} ${target.model}`.toLowerCase().includes(query.trim().toLowerCase()),
  )
  const groups: Array<[title: string, items: Target[]]> = [
    ['Recipes · routed by the Router', visible.filter((target) => target.kind === 'mom')],
    ['Models · called directly', visible.filter((target) => target.kind !== 'mom')],
  ]
  const isSelected = (target: Target) => selected.some((item) => item.id === target.id)
  const toggle = (target: Target, checked: boolean) =>
    onChange(
      checked ? [...selected, { ...target }] : selected.filter((item) => item.id !== target.id),
    )
  return (
    <div className="flex min-w-0 flex-col gap-2.5">
      {targets.length > 8 ? (
        <InputGroup
          type="search"
          aria-label="Search targets"
          placeholder="Search models and recipes…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          leading={<Search aria-hidden="true" />}
          className="h-8"
        />
      ) : null}
      <div
        className="overflow-hidden rounded-lg border border-border"
        role="group"
        aria-label="Targets"
      >
        {groups.map(([title, items]) =>
          items.length ? (
            <div key={title}>
              <div className="border-b border-border bg-surface-subtle/70 px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                {title}
              </div>
              {items.map((target) => {
                const checked = isSelected(target)
                const reason = checked ? null : unavailable(target)
                return (
                  <label
                    key={target.id}
                    className={cn(
                      'flex items-center gap-3 border-b border-border px-3 py-2.5 transition-colors last:border-b-0',
                      reason
                        ? 'cursor-not-allowed opacity-60'
                        : 'cursor-pointer hover:bg-surface-subtle',
                      checked && 'bg-primary-soft/40',
                    )}
                  >
                    <Checkbox
                      checked={checked}
                      disabled={!!reason}
                      aria-label={targetLabel(target)}
                      data-target-id={target.id}
                      onCheckedChange={(value) => toggle(target, value === true)}
                    />
                    {target.kind === 'mom' ? (
                      <Layers aria-hidden="true" className="size-4 shrink-0 text-chart-1" />
                    ) : (
                      <ProviderLogo provider={targetProvider(target)} size={16} />
                    )}
                    <span className="flex min-w-0 flex-1 flex-col">
                      <span
                        className={cn(
                          'truncate text-[13px] font-medium',
                          target.kind === 'mom' && 'font-mono text-[12.5px]',
                        )}
                      >
                        {targetLabel(target)}
                      </span>
                      <span className="truncate text-xs text-subtle-foreground">
                        {reason ??
                          (target.kind === 'mom'
                            ? target.capture_recipe
                              ? 'Mixture of models · recipe snapshot captured'
                              : 'Mixture of models'
                            : 'Single model')}
                      </span>
                    </span>
                    <Badge tone={priced(target) ? 'outline' : 'warning'} size="sm">
                      {priced(target) ? 'Priced' : 'No prices'}
                    </Badge>
                  </label>
                )
              })}
            </div>
          ) : null,
        )}
        {!visible.length ? (
          <p className="px-3 py-4 text-center text-[13px] text-muted-foreground">
            No targets match “{query}”.
          </p>
        ) : null}
      </div>
    </div>
  )
}
