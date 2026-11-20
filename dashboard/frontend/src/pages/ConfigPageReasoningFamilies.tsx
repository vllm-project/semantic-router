import { useMemo, useState } from 'react'
import { ArrowLeft, Brain, Search } from 'lucide-react'

import {
  Badge,
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  EmptyState,
  IconTile,
  KeyValueGrid,
  SectionLabel,
  useRetainedValue,
} from '../ui'
import { ModelLogo } from './ConfigPageModelLogo'
import type { NormalizedModel, ReasoningFamily } from './configPageSupport'

interface FamilyRow {
  name: string
  family: ReasoningFamily
  controls: string[]
  defaultLevel: string
  models: NormalizedModel[]
}

const monogram = (name: string) =>
  name
    .split(/[-_\s.]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toLocaleUpperCase())
    .join('') || 'R'

function familyRows(
  families: Record<string, ReasoningFamily>,
  models: NormalizedModel[],
): FamilyRow[] {
  return Object.entries(families).map(([name, family]) => ({
    name,
    family,
    controls: [...(family.levels ?? []), ...(family.modes ?? [])],
    defaultLevel: family.default || family.default_mode || '',
    models: models.filter((model) => model.reasoning_family === name),
  }))
}

function filterFamilyRows(rows: FamilyRow[], search: string): FamilyRow[] {
  const query = search.trim().toLocaleLowerCase()
  if (!query) return rows
  return rows.filter((row) =>
    [row.name, row.family.type, row.family.parameter, ...row.controls].some((value) =>
      value?.toLocaleLowerCase().includes(query),
    ),
  )
}

function FamilyMark({ name }: { name: string }) {
  return (
    <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-chart-3/12 text-[11px] font-semibold text-chart-3 ring-1 ring-inset ring-chart-3/20">
      {monogram(name)}
    </span>
  )
}

function FamilyList({ rows, onOpen }: { rows: FamilyRow[]; onOpen: (name: string) => void }) {
  const [search, setSearch] = useState('')
  const visible = filterFamilyRows(rows, search)
  return (
    <div className="flex flex-col gap-3">
      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-subtle-foreground" />
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search family, type or parameter…"
          aria-label="Search reasoning families"
          className="h-8 w-full rounded-md border border-input bg-surface pl-8 pr-3 text-[13px] shadow-xs outline-none transition-[border-color,box-shadow] placeholder:text-subtle-foreground hover:border-border-strong focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
        />
      </div>
      {visible.length ? (
        <ul className="flex flex-col overflow-hidden rounded-xl border border-border">
          {visible.map((row) => (
            <li key={row.name} className="border-b border-border last:border-0">
              <button
                type="button"
                onClick={() => onOpen(row.name)}
                className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle focus-visible:outline-none"
              >
                <FamilyMark name={row.name} />
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate font-mono text-[13px] font-medium">{row.name}</span>
                  <span className="truncate text-xs text-subtle-foreground">
                    {row.family.type.replace(/_/g, ' ')} · {row.family.parameter}
                  </span>
                </span>
                <span className="shrink-0 font-mono text-xs tabular text-subtle-foreground">
                  {row.models.length
                    ? `${row.models.length} ${row.models.length === 1 ? 'model' : 'models'}`
                    : '—'}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <div className="rounded-xl border border-dashed border-border-strong">
          <EmptyState
            compact
            icon={Brain}
            tone="violet"
            title={rows.length ? 'No families match' : 'No built-in reasoning families'}
            description={
              rows.length ? 'Try another search.' : 'The model catalog lists no reasoning families.'
            }
          />
        </div>
      )}
      <p className="px-1 text-xs text-subtle-foreground">
        A custom model picks one of these, or describes its reasoning inline, in its Reasoning
        settings.
      </p>
    </div>
  )
}

function FamilyDetail({
  row,
  onOpenModel,
}: {
  row: FamilyRow
  onOpenModel: (name: string) => void
}) {
  return (
    <div className="flex flex-col gap-5">
      <KeyValueGrid
        items={[
          { label: 'Type', value: row.family.type.replace(/_/g, ' ') },
          { label: 'Parameter', value: row.family.parameter, mono: true },
          {
            label: 'Activation parameter',
            value: row.family.activation_parameter || '—',
            mono: Boolean(row.family.activation_parameter),
          },
          { label: 'Default', value: row.defaultLevel || 'Model-selected', mono: true },
          { label: 'Levels', value: row.family.levels?.join(', ') || '—', mono: true },
          { label: 'Modes', value: row.family.modes?.join(', ') || '—', mono: true },
          {
            label: 'Disabled value',
            value: row.family.disabled || '—',
            mono: Boolean(row.family.disabled),
          },
          {
            label: 'Effort flags',
            value:
              Object.entries(row.family.effort_flags ?? {})
                .map(([effort, flag]) => `${effort} → ${flag}`)
                .join(', ') || '—',
            mono: true,
          },
        ]}
      />
      <section className="flex flex-col gap-2">
        <SectionLabel>Models using this family · {row.models.length}</SectionLabel>
        {row.models.length ? (
          row.models.map((model) => (
            <button
              key={model.name}
              type="button"
              onClick={() => onOpenModel(model.name)}
              className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle"
            >
              <ModelLogo model={model} size={16} />
              <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] font-medium">
                {model.name}
              </span>
            </button>
          ))
        ) : (
          <p className="text-xs text-muted-foreground">No backend model uses it yet.</p>
        )}
      </section>
    </div>
  )
}

/**
 * The built-in reasoning families in a sheet: a searchable list, and one
 * family's request parameters and the models that use it.
 */
export default function ConfigPageReasoningFamilies({
  open,
  families,
  models,
  activeName,
  onOpen,
  onBack,
  onClose,
  onOpenModel,
}: {
  open: boolean
  families: Record<string, ReasoningFamily>
  models: NormalizedModel[]
  activeName: string | null
  onOpen: (name: string) => void
  onBack: () => void
  onClose: () => void
  onOpenModel: (name: string) => void
}) {
  const rows = useMemo(() => familyRows(families, models), [families, models])
  const active = rows.find((row) => row.name === activeName) ?? null
  // The closing sheet keeps showing the view it had, list or family.
  const view = useRetainedValue(open ? { active } : null)
  const detail = view?.active ?? null

  return (
    <DetailSheet open={open} onOpenChange={(next) => (next ? null : onClose())} size="md">
      {detail ? (
        <DetailSheetHeader
          icon={
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onBack}
              aria-label="All reasoning families"
            >
              <ArrowLeft />
            </Button>
          }
          title={<span className="font-mono">{detail.name}</span>}
          badges={<Badge tone="outline">built-in</Badge>}
          description={
            <>
              <span>{detail.family.type.replace(/_/g, ' ')}</span>
              <span className="text-subtle-foreground">·</span>
              <span className="font-mono">{detail.family.parameter}</span>
            </>
          }
        />
      ) : (
        <DetailSheetHeader
          icon={<IconTile icon={Brain} tone="violet" size="lg" />}
          title="Reasoning families"
          badges={
            <Badge size="sm" tone="neutral" className="tabular">
              {rows.length}
            </Badge>
          }
          description="How each built-in family turns a reasoning level into request parameters."
        />
      )}
      <DetailSheetBody className="px-5 py-5">
        {detail ? (
          <FamilyDetail row={detail} onOpenModel={onOpenModel} />
        ) : (
          <FamilyList rows={rows} onOpen={onOpen} />
        )}
      </DetailSheetBody>
      <DetailSheetFooter>
        <span className="text-xs text-muted-foreground">Defined by the model catalog</span>
        <Button variant="ghost" size="sm" className="ml-auto" onClick={onClose}>
          Close
        </Button>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
