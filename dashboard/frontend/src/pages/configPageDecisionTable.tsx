import { Split } from 'lucide-react'

import { Badge, IconTile, MiniBar, Tooltip, type Column } from '../ui'
import { DecisionConditionChips, DecisionModelsCell } from './ConfigPageDecisionsSectionCells'
import { algorithmMeta, decisionKind, pluginLabel } from './configPageDecisionPresentation'
import type { DecisionConfig, NormalizedModel } from './configPageSupport'

export interface DecisionColumnContext {
  models: readonly NormalizedModel[]
  /** Requests Router Replay recorded per decision; absent when replay is unavailable. */
  traffic?: ReadonlyMap<string, number>
}

/** Columns of the decisions list; every value comes from the decision as written. */
export function decisionColumns({
  models,
  traffic,
}: DecisionColumnContext): Column<DecisionConfig>[] {
  const maxTraffic = traffic ? Math.max(1, ...traffic.values()) : 1
  const columns: Column<DecisionConfig>[] = [
    {
      id: 'name',
      header: 'Decision',
      sortValue: (row) => row.name,
      cell: (row) => {
        const kind = decisionKind(row)
        return (
          <div className="flex min-w-0 items-center gap-3">
            <IconTile
              icon={Split}
              tone={kind === 'guardrail' ? 'rose' : kind === 'fallback' ? 'slate' : 'violet'}
              size="sm"
            />
            <div className="flex min-w-0 flex-col">
              <span className="max-w-[300px] truncate font-mono text-[13px] font-medium">
                {row.name}
              </span>
              {row.description ? (
                <span className="max-w-[300px] truncate text-xs text-subtle-foreground">
                  {row.description}
                </span>
              ) : null}
            </div>
          </div>
        )
      },
    },
    {
      id: 'priority',
      header: 'Priority',
      sortValue: (row) => row.priority ?? 0,
      cell: (row) => (
        <span className="flex items-center gap-1">
          <span className="rounded-md bg-secondary px-1.5 py-0.5 font-mono text-xs font-medium tabular">
            P{row.priority ?? 0}
          </span>
          {row.tier !== undefined ? (
            <Tooltip content="Decision tier">
              <span className="font-mono text-[11px] text-subtle-foreground">T{row.tier}</span>
            </Tooltip>
          ) : null}
        </span>
      ),
    },
    {
      id: 'when',
      header: 'When',
      hideBelow: 'xl',
      cell: (row) => <DecisionConditionChips decision={row} />,
    },
    {
      id: 'models',
      header: 'Models',
      cell: (row) => <DecisionModelsCell decision={row} models={models} />,
    },
    {
      id: 'algorithm',
      header: 'Algorithm',
      hideBelow: 'xl',
      sortValue: (row) => String(row.algorithm?.type ?? ''),
      cell: (row) => {
        const meta = algorithmMeta(row.algorithm?.type)
        return meta ? (
          <Badge tone="outline">{meta.label}</Badge>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        )
      },
    },
    {
      id: 'plugins',
      header: 'Plugins',
      hideBelow: '2xl',
      sortValue: (row) => row.plugins?.length ?? 0,
      cell: (row) =>
        row.plugins?.length ? (
          <Tooltip content={row.plugins.map((plugin) => pluginLabel(plugin.type)).join(', ')}>
            <span className="text-xs">
              <span className="font-medium tabular">{row.plugins.length}</span>{' '}
              <span className="text-muted-foreground">
                {row.plugins.length === 1 ? 'plugin' : 'plugins'}
              </span>
            </span>
          </Tooltip>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
  ]
  if (traffic) {
    columns.push({
      id: 'traffic',
      header: 'Recorded',
      align: 'right',
      sortValue: (row) => traffic.get(row.name) ?? 0,
      cell: (row) => (
        <div className="ml-auto flex w-24 flex-col items-end gap-1">
          <span className="font-mono text-xs tabular">
            {(traffic.get(row.name) ?? 0).toLocaleString()}
          </span>
          <MiniBar
            value={traffic.get(row.name) ?? 0}
            max={maxTraffic}
            tone="violet"
            className="h-1"
          />
        </div>
      ),
    })
  }
  return columns
}
