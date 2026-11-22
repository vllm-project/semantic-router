import { motion } from 'framer-motion'
import { ArrowDown, ArrowUp, Plus, Trash2 } from 'lucide-react'

import {
  getProjectionFieldSchema,
  PROJECTION_INPUT_TYPES,
  projectionValueSources,
} from '../lib/projectionSchemas'
import type { FieldSchema } from '../lib/dslSchemas'
import { Button, ChipInput, Slider, cn, toneFill } from '../ui'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import {
  BoundsInput,
  NumberInput,
  OptionSelect,
  SuggestInput,
  SuggestionChips,
} from './ConfigPageSchemaFieldsEditorControls'
import { useSchemaFormHints } from './ConfigPageSchemaFieldsEditorSupport'
import { SignalChip } from './configPageRoutingScopeViews'
import { projectionInputErrors, projectionOutputErrors } from './configPageProjectionFormSupport'
import { bandGeometry } from './configPageProjectionTableSupport'
import { signalTypeIcon, signalTypeLabel } from './configPageSignalCatalog'
import type {
  ProjectionMappingCalibration,
  ProjectionMappingOutput,
  ProjectionScoreInput,
} from './configPageSupport'

const ease = [0.16, 1, 0.3, 1] as const

function nested(collection: 'scores' | 'mappings', key: string): FieldSchema[] {
  return getProjectionFieldSchema(collection).find((field) => field.key === key)?.fields ?? []
}

function nestedField(
  collection: 'scores' | 'mappings',
  parent: string,
  key: string,
): FieldSchema | undefined {
  return nested(collection, parent).find((field) => field.key === key)
}

const signed = (value: number) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`

/** Diverging weight bars: positive weights grow right, negative ones left. */
export function WeightBars({
  inputs,
  compact = false,
}: {
  inputs: readonly ProjectionScoreInput[]
  compact?: boolean
}) {
  const weights = inputs.map((input) => (Number.isFinite(input.weight) ? input.weight : 0))
  const max = Math.max(0.05, ...weights.map((weight) => Math.abs(weight)))
  const grid = compact
    ? 'grid-cols-[minmax(0,190px)_1fr_48px]'
    : 'grid-cols-[minmax(0,220px)_1fr_56px]'
  if (inputs.length === 0) return <p className="text-xs text-muted-foreground">No inputs yet.</p>
  return (
    <div className="flex flex-col gap-1.5">
      {inputs.map((input, index) => {
        const weight = weights[index]
        return (
          <div
            key={`${input.type}:${input.name}:${index}`}
            className={cn('grid items-center gap-3', grid)}
          >
            <SignalChip
              type={input.type}
              name={input.name ?? ''}
              className="max-w-full justify-self-start"
            />
            <div className="relative h-5">
              <span aria-hidden className="absolute inset-y-0 left-1/2 w-px bg-border-strong" />
              <motion.span
                className="absolute top-1/2 h-2.5 -translate-y-1/2 rounded-full"
                style={{
                  background: weight >= 0 ? toneFill.teal : toneFill.rose,
                  ...(weight >= 0 ? { left: '50%' } : { right: '50%' }),
                }}
                initial={{ width: 0 }}
                animate={{ width: `${(Math.abs(weight) / max) * 50}%` }}
                transition={{ duration: 0.6, delay: Math.min(index, 8) * 0.03, ease }}
              />
            </div>
            <span
              className={cn(
                'text-right font-mono text-xs tabular',
                weight >= 0 ? 'text-foreground' : 'text-chart-5',
              )}
            >
              {signed(weight)}
            </span>
          </div>
        )
      })}
      <div className={cn('grid gap-3 font-mono text-[10.5px] text-subtle-foreground', grid)}>
        <span />
        <span className="flex justify-between">
          <span>−{max.toFixed(2)}</span>
          <span>0</span>
          <span>+{max.toFixed(2)}</span>
        </span>
        <span />
      </div>
    </div>
  )
}

/** The bands a mapping cuts its score into, on one axis. */
export function BandStrip({
  outputs,
  method,
}: {
  outputs: readonly ProjectionMappingOutput[]
  method?: string
}) {
  const { low, high, bands } = bandGeometry(outputs)
  const span = high - low || 1
  const position = (value: number) => ((Math.min(high, Math.max(low, value)) - low) / span) * 100
  if (bands.length === 0) return <p className="text-xs text-muted-foreground">No bands yet.</p>
  const ticks = [
    ...new Set(
      bands.flatMap((band) => [band.from, band.to]).filter((value) => value > low && value < high),
    ),
  ].sort((a, b) => a - b)
  if (method === 'multi_emit') {
    return (
      <div className="flex flex-col gap-1">
        {bands.map((band) => (
          <div key={band.name} className="grid grid-cols-[minmax(0,140px)_1fr] items-center gap-3">
            <span className="truncate font-mono text-[11.5px]">{band.name}</span>
            <span className="relative h-2.5 rounded-full bg-muted">
              <span
                className="absolute inset-y-0 rounded-full bg-chart-4/70"
                style={{
                  left: `${position(band.from)}%`,
                  width: `${Math.max(1.5, position(band.to) - position(band.from))}%`,
                }}
              />
            </span>
          </div>
        ))}
        <div className="flex justify-between pl-[152px] font-mono text-[10.5px] text-subtle-foreground">
          <span>{low.toFixed(2)}</span>
          <span>{high.toFixed(2)}</span>
        </div>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative h-8 overflow-hidden rounded-lg border border-border bg-surface-subtle/60">
        {bands.map((band, index) => (
          <div
            key={band.name}
            title={`${band.name} · ${band.label}`}
            className={cn(
              'absolute inset-y-0 flex min-w-0 items-center justify-center px-1.5 font-mono text-[11px] text-muted-foreground',
              index > 0 && 'border-l border-border',
            )}
            style={{
              left: `${position(band.from)}%`,
              width: `${Math.max(0, position(band.to) - position(band.from))}%`,
              background: `color-mix(in oklch, var(--chart-4) ${8 + index * 7}%, transparent)`,
            }}
          >
            <span className="truncate">{band.name}</span>
          </div>
        ))}
      </div>
      <div className="relative h-3 font-mono text-[10.5px] text-subtle-foreground">
        <span className="absolute left-0">{low.toFixed(2)}</span>
        {ticks.map((tick) => (
          <span
            key={tick}
            className="absolute -translate-x-1/2"
            style={{ left: `${position(tick)}%` }}
          >
            {tick.toFixed(2)}
          </span>
        ))}
        <span className="absolute right-0">{high.toFixed(2)}</span>
      </div>
    </div>
  )
}

function inputTypeOption(type: string) {
  const Icon = signalTypeIcon(type)
  return (
    <span className="flex items-center gap-2">
      <Icon className={cn('size-3.5', type === 'projection' ? 'text-chart-4' : 'text-chart-2')} />
      <span className="font-sans">{signalTypeLabel(type)}</span>
    </span>
  )
}

/** Weighted inputs of a score: a signal or an earlier projection, its weight and how its value is read. */
export function ProjectionInputsEditor({
  value,
  onChange,
  readOnly = false,
  projectionNames = [],
}: {
  value: unknown
  onChange?: (value: ProjectionScoreInput[]) => void
  readOnly?: boolean
  projectionNames?: readonly string[]
}) {
  const hints = useSchemaFormHints()
  const inputs = (Array.isArray(value) ? value : []) as ProjectionScoreInput[]
  const update = (index: number, patch: Partial<ProjectionScoreInput>) =>
    onChange?.(
      inputs.map((input, position) => {
        if (position !== index) return input
        const next = { ...input, ...patch } as ProjectionScoreInput & Record<string, unknown>
        for (const key of Object.keys(patch))
          if (next[key] === undefined || next[key] === '') delete next[key]
        return next
      }),
    )
  const move = (index: number, delta: number) => {
    const target = index + delta
    if (target < 0 || target >= inputs.length) return
    const next = [...inputs]
    ;[next[index], next[target]] = [next[target], next[index]]
    onChange?.(next)
  }
  const label = (key: string) => nestedField('scores', 'inputs', key)?.label ?? key
  return (
    <div className="flex flex-col gap-2">
      {inputs.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border-strong px-3 py-3 text-xs text-muted-foreground">
          No inputs. A score adds up weighted signals and earlier projections.
        </p>
      ) : (
        <ol className="flex flex-col gap-2">
          {inputs.map((input, index) => {
            const errors = projectionInputErrors({
              ...input,
              type: input.type ?? '',
              weight: typeof input.weight === 'number' ? input.weight : Number.NaN,
            })
            const names =
              input.type === 'projection'
                ? projectionNames
                : (hints.signalNames?.(input.type) ?? [])
            return (
              <li
                key={index}
                className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 shadow-xs"
              >
                <div className="grid grid-cols-1 items-end gap-2 sm:grid-cols-[170px_minmax(0,1fr)_96px_auto]">
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">Input type</span>
                    <OptionSelect
                      ariaLabel="Input type"
                      value={input.type}
                      required
                      options={PROJECTION_INPUT_TYPES}
                      renderOption={inputTypeOption}
                      disabled={readOnly}
                      className="h-8"
                      onChange={(type) =>
                        update(index, {
                          type: type ?? '',
                          name: undefined,
                          value_source: undefined,
                        })
                      }
                    />
                  </label>
                  <label className="flex min-w-0 flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">
                      {input.type === 'projection' ? 'Projection' : 'Signal'}
                    </span>
                    <SuggestInput
                      ariaLabel="Signal or projection name"
                      value={input.name ?? ''}
                      suggestions={names}
                      mono
                      placeholder="name or name:label"
                      disabled={readOnly}
                      onChange={(name) => update(index, { name })}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">{label('weight')}</span>
                    <NumberInput
                      ariaLabel="Weight"
                      value={input.weight}
                      disabled={readOnly}
                      className="text-center"
                      onChange={(weight) => update(index, { weight: weight ?? Number.NaN })}
                    />
                  </label>
                  {readOnly ? (
                    <span />
                  ) : (
                    <span className="flex items-center pb-0.5">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Move input ${index + 1} up`}
                        disabled={index === 0}
                        onClick={() => move(index, -1)}
                      >
                        <ArrowUp />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Move input ${index + 1} down`}
                        disabled={index === inputs.length - 1}
                        onClick={() => move(index, 1)}
                      >
                        <ArrowDown />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Remove input ${index + 1}`}
                        onClick={() =>
                          onChange?.(inputs.filter((_, position) => position !== index))
                        }
                      >
                        <Trash2 />
                      </Button>
                    </span>
                  )}
                </div>
                {readOnly ? null : (
                  <Slider
                    aria-label={`Weight of ${input.name || `input ${index + 1}`}`}
                    value={Number.isFinite(input.weight) ? input.weight : 0}
                    min={-Math.max(1, Math.abs(Number.isFinite(input.weight) ? input.weight : 0))}
                    max={Math.max(1, Math.abs(Number.isFinite(input.weight) ? input.weight : 0))}
                    step={0.01}
                    tone={(Number.isFinite(input.weight) ? input.weight : 0) < 0 ? 'rose' : 'teal'}
                    onValueChange={(weight) =>
                      update(index, { weight: Math.round(weight * 100) / 100 })
                    }
                  />
                )}
                <div className="grid grid-cols-1 items-end gap-2 sm:grid-cols-[170px_120px_120px]">
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">Value source</span>
                    <OptionSelect
                      ariaLabel="Value source"
                      value={input.value_source}
                      options={['', ...projectionValueSources(input.type)]}
                      disabled={readOnly}
                      className="h-8"
                      onChange={(source) => update(index, { value_source: source })}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">{label('match')}</span>
                    <NumberInput
                      ariaLabel="Match value"
                      value={input.match}
                      placeholder="1"
                      disabled={readOnly}
                      className="h-8"
                      onChange={(match) => update(index, { match })}
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">{label('miss')}</span>
                    <NumberInput
                      ariaLabel="Miss value"
                      value={input.miss}
                      placeholder="0"
                      disabled={readOnly}
                      className="h-8"
                      onChange={(miss) => update(index, { miss })}
                    />
                  </label>
                </div>
                {!readOnly && errors.length ? (
                  <p className="text-xs text-destructive-foreground">{errors[0]}</p>
                ) : null}
              </li>
            )
          })}
        </ol>
      )}
      {readOnly ? null : (
        <Button
          type="button"
          variant="outline"
          size="xs"
          className="w-fit"
          onClick={() => onChange?.([...inputs, { type: 'keyword', name: '', weight: 0.1 }])}
        >
          <Plus />
          Add input
        </Button>
      )}
    </div>
  )
}

/** Named bands of a mapping with their lower and upper bounds. */
export function ProjectionOutputsEditor({
  value,
  onChange,
  readOnly = false,
}: {
  value: unknown
  onChange?: (value: ProjectionMappingOutput[]) => void
  readOnly?: boolean
}) {
  const outputs = (Array.isArray(value) ? value : []) as ProjectionMappingOutput[]
  const set = (index: number, next: ProjectionMappingOutput) =>
    onChange?.(outputs.map((output, position) => (position === index ? next : output)))
  const move = (index: number, delta: number) => {
    const target = index + delta
    if (target < 0 || target >= outputs.length) return
    const next = [...outputs]
    ;[next[index], next[target]] = [next[target], next[index]]
    onChange?.(next)
  }
  return (
    <div className="flex flex-col gap-2">
      {outputs.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border-strong px-3 py-3 text-xs text-muted-foreground">
          No bands. Add a band for each range of the score that decisions should match.
        </p>
      ) : (
        <ol className="flex flex-col gap-2">
          {outputs.map((output, index) => {
            const errors = projectionOutputErrors(output)
            const { name, ...bounds } = output
            return (
              <li
                key={index}
                className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 shadow-xs"
              >
                <div className="grid grid-cols-1 items-end gap-3 sm:grid-cols-[minmax(0,180px)_minmax(0,1fr)_auto]">
                  <label className="flex min-w-0 flex-col gap-1">
                    <span className="text-xs text-subtle-foreground">Band name</span>
                    <SuggestInput
                      ariaLabel="Band name"
                      value={name ?? ''}
                      mono
                      disabled={readOnly}
                      placeholder="support_fast"
                      onChange={(next) => set(index, { ...output, name: next })}
                    />
                  </label>
                  <BoundsInput
                    idPrefix={`band-${index}`}
                    value={bounds as Record<string, unknown>}
                    disabled={readOnly}
                    onChange={(next) =>
                      set(index, { name, ...(next ?? {}) } as ProjectionMappingOutput)
                    }
                  />
                  {readOnly ? (
                    <span />
                  ) : (
                    <span className="flex items-center pb-0.5">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Move band ${index + 1} up`}
                        disabled={index === 0}
                        onClick={() => move(index, -1)}
                      >
                        <ArrowUp />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Move band ${index + 1} down`}
                        disabled={index === outputs.length - 1}
                        onClick={() => move(index, 1)}
                      >
                        <ArrowDown />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`Remove band ${index + 1}`}
                        onClick={() =>
                          onChange?.(outputs.filter((_, position) => position !== index))
                        }
                      >
                        <Trash2 />
                      </Button>
                    </span>
                  )}
                </div>
                {!readOnly && errors.length ? (
                  <p className="text-xs text-destructive-foreground">{errors[0]}</p>
                ) : null}
              </li>
            )
          })}
        </ol>
      )}
      {readOnly ? null : (
        <Button
          type="button"
          variant="outline"
          size="xs"
          className="w-fit"
          onClick={() => onChange?.([...outputs, { name: '' }])}
        >
          <Plus />
          Add band
        </Button>
      )}
    </div>
  )
}

/** Optional calibration of a mapping's band confidence, from the Router schema. */
export function ProjectionCalibrationEditor({
  value,
  onChange,
  readOnly = false,
}: {
  value: unknown
  onChange?: (value: ProjectionMappingCalibration | undefined) => void
  readOnly?: boolean
}) {
  return (
    <ConfigPageSchemaFieldsEditor
      schema={nested('mappings', 'calibration')}
      value={value && typeof value === 'object' ? (value as Record<string, unknown>) : {}}
      readOnly={readOnly}
      showAllByDefault
      path={['projections', 'mappings', 'calibration']}
      onChange={(next) =>
        onChange?.(
          Object.keys(next).length ? (next as unknown as ProjectionMappingCalibration) : undefined,
        )
      }
    />
  )
}

/** Domain signals a partition coordinates. */
export function ProjectionMembersEditor({
  value,
  onChange,
  readOnly = false,
}: {
  value: unknown
  onChange?: (value: string[]) => void
  readOnly?: boolean
}) {
  const hints = useSchemaFormHints()
  const members = (Array.isArray(value) ? value : []).filter(
    (member): member is string => typeof member === 'string',
  )
  if (readOnly) {
    return (
      <div className="flex flex-wrap gap-1.5">
        {members.map((member) => (
          <SignalChip key={member} type="domain" name={member} />
        ))}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-1.5">
      <ChipInput
        aria-label="Members"
        value={members}
        caseSensitive
        placeholder="Add a domain signal"
        onChange={(next) => onChange?.(next)}
      />
      <SuggestionChips
        value={members}
        suggestions={hints.signalNames?.('domain') ?? []}
        onAdd={(next) => onChange?.(next)}
      />
    </div>
  )
}
