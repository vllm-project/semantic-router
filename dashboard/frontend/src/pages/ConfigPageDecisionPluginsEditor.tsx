import * as React from 'react'
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Plus,
  Puzzle,
  Trash2,
} from 'lucide-react'

import { getPluginFieldSchema } from '../lib/dslSchemas'
import {
  Badge,
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
  IconTile,
} from '../ui'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import { JsonInput } from './ConfigPageSchemaFieldsEditorControls'
import { PLUGINS, pluginLabel } from './configPageDecisionPresentation'
import type { DecisionPluginConfig } from './configPageSupport'

interface ConfigPageDecisionPluginsEditorProps {
  value: DecisionPluginConfig[]
  onChange: (value: DecisionPluginConfig[]) => void
  readOnly?: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function configurationSummary(configuration: Record<string, unknown>): string {
  const entries = Object.entries(configuration).filter(([key]) => key !== 'enabled')
  const shown = entries
    .slice(0, 3)
    .map(([key, value]) =>
      value && typeof value === 'object'
        ? key
        : `${key}: ${Array.isArray(value) ? value.length : String(value)}`,
    )
  return [...shown, entries.length > 3 ? `${entries.length - 3} more` : '']
    .filter(Boolean)
    .join(' · ')
}

function enabledState(configuration: Record<string, unknown>): 'on' | 'off' | 'default' {
  if (configuration.enabled === false) return 'off'
  if (configuration.enabled === true) return 'on'
  return 'default'
}

function PluginCard({
  plugin,
  index,
  count,
  readOnly,
  open,
  onOpenChange,
  onChange,
  onMove,
  onRemove,
}: {
  plugin: DecisionPluginConfig
  index: number
  count: number
  readOnly: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onChange: (plugin: DecisionPluginConfig) => void
  onMove: (delta: number) => void
  onRemove: () => void
}) {
  const schema = getPluginFieldSchema(plugin.type)
  const known = PLUGINS.some((candidate) => candidate.type === plugin.type)
  const configuration = isRecord(plugin.configuration) ? plugin.configuration : {}
  const description = PLUGINS.find((candidate) => candidate.type === plugin.type)?.description
  const state = enabledState(configuration)
  return (
    <li className="rounded-xl border border-border bg-surface shadow-xs">
      <Collapsible open={open} onOpenChange={onOpenChange}>
        <div className="flex items-center gap-2.5 px-3 py-2.5">
          <CollapsibleTrigger asChild>
            <button
              type="button"
              aria-label={`${open ? 'Collapse' : 'Expand'} ${plugin.type}`}
              className="flex min-w-0 flex-1 items-center gap-2.5 rounded-md text-left focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
            >
              {open ? (
                <ChevronDown className="size-3.5 shrink-0 text-subtle-foreground" />
              ) : (
                <ChevronRight className="size-3.5 shrink-0 text-subtle-foreground" />
              )}
              <IconTile icon={Puzzle} tone="rose" size="sm" />
              <span className="flex min-w-0 flex-col">
                <span className="flex items-center gap-2">
                  <span className="truncate text-[13px] font-medium">
                    {pluginLabel(plugin.type)}
                  </span>
                  {state === 'off' ? (
                    <Badge size="sm" tone="neutral">
                      Off
                    </Badge>
                  ) : null}
                  {!known ? (
                    <Badge size="sm" tone="warning">
                      Not in the contract
                    </Badge>
                  ) : null}
                </span>
                <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
                  {plugin.type}
                  {configurationSummary(configuration) ? (
                    <span className="font-sans"> · {configurationSummary(configuration)}</span>
                  ) : null}
                </span>
              </span>
            </button>
          </CollapsibleTrigger>
          {readOnly ? null : (
            <span className="flex shrink-0 items-center">
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label={`Move ${plugin.type} up`}
                disabled={index === 0}
                onClick={() => onMove(-1)}
              >
                <ArrowUp />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label={`Move ${plugin.type} down`}
                disabled={index === count - 1}
                onClick={() => onMove(1)}
              >
                <ArrowDown />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label={`Remove ${plugin.type}`}
                onClick={onRemove}
              >
                <Trash2 />
              </Button>
            </span>
          )}
        </div>
        <CollapsibleContent>
          <div className="flex flex-col gap-3 border-t border-border px-3.5 py-3.5">
            {description ? (
              <p className="text-xs leading-relaxed text-muted-foreground">{description}</p>
            ) : null}
            {known && schema.length > 0 ? (
              <ConfigPageSchemaFieldsEditor
                schema={schema}
                value={configuration}
                readOnly={readOnly}
                path={['plugins', String(index), 'configuration']}
                onChange={(next) => onChange({ ...plugin, configuration: next })}
              />
            ) : (
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-muted-foreground">
                  No form is registered for this plugin type; its configuration is edited as JSON
                  and saved as written.
                </span>
                <JsonInput
                  value={configuration}
                  disabled={readOnly}
                  onChange={(next) =>
                    onChange({ ...plugin, configuration: isRecord(next) ? next : {} })
                  }
                />
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </li>
  )
}

/** A decision's plugins, in order, each configured from its Router schema. */
export default function ConfigPageDecisionPluginsEditor({
  value,
  onChange,
  readOnly = false,
}: ConfigPageDecisionPluginsEditorProps) {
  const plugins = Array.isArray(value) ? value : []
  const [open, setOpen] = React.useState<Set<number>>(
    () => new Set(plugins.length === 1 ? [0] : []),
  )
  const used = new Set(plugins.map((plugin) => plugin.type))
  const move = (index: number, delta: number) => {
    const target = index + delta
    if (target < 0 || target >= plugins.length) return
    const next = [...plugins]
    ;[next[index], next[target]] = [next[target], next[index]]
    onChange(next)
    setOpen(new Set())
  }
  return (
    <div className="flex flex-col gap-2">
      {plugins.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border-strong px-4 py-4 text-[13px] text-muted-foreground">
          No plugins. Plugins change the request or the answer after this decision matches: caching,
          system prompts, guards and more.
        </p>
      ) : (
        <ol className="flex flex-col gap-2">
          {plugins.map((plugin, index) => (
            <PluginCard
              key={`${plugin.type}-${index}`}
              plugin={plugin}
              index={index}
              count={plugins.length}
              readOnly={readOnly}
              open={open.has(index)}
              onOpenChange={(next) =>
                setOpen((current) => {
                  const updated = new Set(current)
                  if (next) updated.add(index)
                  else updated.delete(index)
                  return updated
                })
              }
              onChange={(next) =>
                onChange(plugins.map((entry, position) => (position === index ? next : entry)))
              }
              onMove={(delta) => move(index, delta)}
              onRemove={() => {
                onChange(plugins.filter((_, position) => position !== index))
                setOpen(new Set())
              }}
            />
          ))}
        </ol>
      )}
      {readOnly ? null : (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button type="button" variant="outline" size="sm" className="w-fit">
              <Plus />
              Add plugin
              <ChevronDown className="-mr-1 opacity-70" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="start"
            className="max-h-[min(460px,var(--radix-dropdown-menu-content-available-height))] w-80 overflow-y-auto"
          >
            <DropdownMenuLabel>Plugin type</DropdownMenuLabel>
            {PLUGINS.map((plugin) => (
              <DropdownMenuItem
                key={plugin.type}
                className="items-start"
                onSelect={() => {
                  onChange([...plugins, { type: plugin.type, configuration: {} }])
                  setOpen(new Set([plugins.length]))
                }}
              >
                <Puzzle className="mt-0.5 text-chart-5" />
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="flex items-center gap-2">
                    <span className="font-medium">{plugin.label}</span>
                    {used.has(plugin.type) ? (
                      <span className="inline-flex items-center gap-0.5 text-[11px] text-subtle-foreground">
                        <Check className="size-3" />
                        added
                      </span>
                    ) : null}
                  </span>
                  <span className="text-xs leading-snug text-muted-foreground">
                    {plugin.description}
                  </span>
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </div>
  )
}
