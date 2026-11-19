import { ArrowDown, ArrowUp, BrainCircuit, Plus, Trash2 } from 'lucide-react'

import { DECISION_MODEL_REF_SCHEMA } from '../lib/decisionSchemas'
import type { FieldSchema } from '../lib/dslSchemas'
import {
  Badge,
  Button,
  ProviderLogo,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
  Tooltip,
  cn,
} from '../ui'
import { SchemaField } from './ConfigPageSchemaFieldsEditor'
import { NumberInput, OptionSelect, SuggestInput } from './ConfigPageSchemaFieldsEditorControls'
import { modelDisplayName, modelLogo } from './configPageDecisionPresentation'
import {
  defaultReasoningEnabled,
  reasoningFamilyForModel,
  reasoningFamilyIsAlwaysOn,
} from './configPageReasoningControlSupport'
import type { DecisionModelRef, NormalizedModel, ReasoningFamily } from './configPageSupport'

type ModelRef = DecisionModelRef & Record<string, unknown>

// These ModelRef fields get model-aware controls; any other field the Router
// schema adds renders through the generic schema field.
const SPECIAL_FIELDS = new Set([
  'model',
  'use_reasoning',
  'reasoning_mode',
  'reasoning_effort',
  'lora_name',
  'weight',
])
const UNLISTED = '__unlisted__'

function fieldLabel(key: string): string {
  return DECISION_MODEL_REF_SCHEMA.find((field) => field.key === key)?.label ?? key
}

function withField(ref: ModelRef, key: string, value: unknown): ModelRef {
  const next = { ...ref }
  if (value === undefined || value === '') delete next[key]
  else next[key] = value
  return next
}

/** Picks a configured model; a name the catalog does not list stays selectable. */
export function ModelPicker({
  value,
  models,
  onChange,
  disabled,
  ariaLabel = 'Model',
  className,
}: {
  value: string
  models: readonly NormalizedModel[]
  onChange: (value: string) => void
  disabled?: boolean
  ariaLabel?: string
  className?: string
}) {
  const known = models.some((model) => model.name === value)
  return (
    <Select
      value={value || UNLISTED}
      onValueChange={(next) => onChange(next === UNLISTED ? '' : next)}
      disabled={disabled}
    >
      <SelectTrigger aria-label={ariaLabel} className={cn('w-full', className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {!value ? (
          <SelectItem value={UNLISTED}>
            <span className="text-muted-foreground">Choose a model…</span>
          </SelectItem>
        ) : null}
        {models.map((model) => (
          <SelectItem key={model.name} value={model.name}>
            <span className="flex min-w-0 items-center gap-2">
              <span className="inline-flex in-data-[slot=select-trigger]:hidden">
                <ProviderLogo provider={modelLogo(model)} size={14} />
              </span>
              <span className="truncate">{modelDisplayName(model, model.name)}</span>
              {modelDisplayName(model, model.name) !== model.name ? (
                <span className="truncate font-mono text-[11px] text-subtle-foreground">
                  {model.name}
                </span>
              ) : null}
            </span>
          </SelectItem>
        ))}
        {value && !known ? (
          <>
            <SelectSeparator />
            <SelectItem value={value}>
              <span className="font-mono text-[12.5px]">{value}</span>
              <span className="ml-2 text-xs text-subtle-foreground">not a configured model</span>
            </SelectItem>
          </>
        ) : null}
      </SelectContent>
    </Select>
  )
}

function ReasoningControls({
  value,
  family,
  onChange,
  readOnly,
}: {
  value: ModelRef
  family: ReasoningFamily | undefined
  onChange: (value: ModelRef) => void
  readOnly: boolean
}) {
  const alwaysOn = reasoningFamilyIsAlwaysOn(family)
  const efforts = family?.levels ?? []
  const modes = family?.modes ?? []
  const state = value.use_reasoning === undefined ? 'default' : value.use_reasoning ? 'on' : 'off'
  const configurable = Boolean(family) || value.use_reasoning !== undefined
  const setReasoning = (next: 'default' | 'on' | 'off') => {
    let updated = withField(value, 'use_reasoning', next === 'default' ? undefined : next === 'on')
    if (next === 'off') {
      updated = withField(updated, 'reasoning_effort', undefined)
      if (updated.reasoning_mode !== 'disabled')
        updated = withField(updated, 'reasoning_mode', undefined)
    }
    onChange(updated)
  }
  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1.5">
        <span className="flex items-center gap-1.5 text-[13px] font-medium">
          <BrainCircuit className="size-3.5 text-subtle-foreground" />
          {fieldLabel('use_reasoning')}
          <span className="font-mono text-[11px] font-normal text-subtle-foreground">
            use_reasoning
          </span>
        </span>
        {alwaysOn ? (
          <Tooltip content="This model's reasoning family cannot turn thinking off">
            <span>
              <Badge tone="info">Always reasons</Badge>
            </span>
          </Tooltip>
        ) : (
          <Tooltip
            content={
              configurable
                ? `Default follows the ${family ? 'reasoning family' : 'provider'}${family && defaultReasoningEnabled(family) ? ', which reasons' : ''}`
                : 'Assign a reasoning family to the model to configure thinking'
            }
          >
            <span>
              <Segmented
                size="sm"
                aria-label="Use reasoning"
                value={state}
                onValueChange={setReasoning}
                className={cn((!configurable || readOnly) && 'pointer-events-none opacity-60')}
                options={[
                  { value: 'default', label: 'Default' },
                  { value: 'on', label: 'On' },
                  { value: 'off', label: 'Off' },
                ]}
              />
            </span>
          </Tooltip>
        )}
      </div>
      {efforts.length > 0 || value.reasoning_effort ? (
        <div className="flex min-w-[150px] flex-col gap-1.5">
          <span className="text-[13px] font-medium">{fieldLabel('reasoning_effort')}</span>
          <OptionSelect
            ariaLabel="Reasoning effort"
            value={value.reasoning_effort}
            options={['', ...efforts]}
            disabled={readOnly || value.use_reasoning === false}
            className="h-8"
            onChange={(next) => onChange(withField(value, 'reasoning_effort', next))}
          />
        </div>
      ) : null}
      {(modes.length > 0 && (efforts.length === 0 || modes.length > 1)) || value.reasoning_mode ? (
        <div className="flex min-w-[150px] flex-col gap-1.5">
          <span className="text-[13px] font-medium">{fieldLabel('reasoning_mode')}</span>
          <OptionSelect
            ariaLabel="Thinking mode"
            value={value.reasoning_mode}
            options={['', ...(modes.length ? modes : ['enabled', 'disabled', 'adaptive'])]}
            disabled={readOnly}
            className="h-8"
            onChange={(next) => {
              let updated = withField(value, 'reasoning_mode', next)
              if (next === 'disabled')
                updated = withField(
                  withField(updated, 'use_reasoning', false),
                  'reasoning_effort',
                  undefined,
                )
              else if (next) updated = withField(updated, 'use_reasoning', true)
              onChange(updated)
            }}
          />
        </div>
      ) : null}
      {family?.default || family?.default_mode ? (
        <span className="pb-1.5 text-xs text-subtle-foreground">
          family default {[family.default, family.default_mode].filter(Boolean).join(' · ')}
        </span>
      ) : null}
    </div>
  )
}

function ModelRefCard({
  value,
  index,
  count,
  models,
  families,
  readOnly,
  onChange,
  onMove,
  onRemove,
}: {
  value: ModelRef
  index: number
  count: number
  models: readonly NormalizedModel[]
  families: Record<string, ReasoningFamily>
  readOnly: boolean
  onChange: (value: ModelRef) => void
  onMove: (delta: number) => void
  onRemove: () => void
}) {
  const model = models.find((candidate) => candidate.name === value.model)
  const family = reasoningFamilyForModel(model, families)
  const loras = (model?.loras ?? [])
    .map((lora) => lora.name)
    .filter((name): name is string => Boolean(name))
  const others = DECISION_MODEL_REF_SCHEMA.filter((field) => !SPECIAL_FIELDS.has(field.key))
  return (
    <li className="flex flex-col gap-3 rounded-xl border border-border bg-surface px-3.5 py-3 shadow-xs">
      <div className="flex items-center gap-3">
        <span className="w-4 shrink-0 text-center font-mono text-xs tabular text-subtle-foreground">
          {index + 1}
        </span>
        <ProviderLogo provider={modelLogo(model)} size={20} framed />
        <div className="min-w-0 flex-1">
          {readOnly ? (
            <div className="flex min-w-0 flex-col">
              <span className="truncate text-[13px] font-medium">
                {modelDisplayName(model, value.model)}
              </span>
              <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
                {value.model}
              </span>
            </div>
          ) : (
            <ModelPicker
              value={value.model}
              models={models}
              onChange={(name) => {
                const nextFamily = reasoningFamilyForModel(
                  models.find((candidate) => candidate.name === name),
                  families,
                )
                let next = withField(value, 'model', name)
                if (name !== value.model) {
                  next = withField(next, 'use_reasoning', defaultReasoningEnabled(nextFamily))
                  next = withField(
                    withField(next, 'reasoning_mode', undefined),
                    'reasoning_effort',
                    undefined,
                  )
                }
                onChange(next)
              }}
            />
          )}
        </div>
        {readOnly ? null : (
          <span className="flex shrink-0 items-center">
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              aria-label={`Move ${value.model || 'model'} up`}
              disabled={index === 0}
              onClick={() => onMove(-1)}
            >
              <ArrowUp />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              aria-label={`Move ${value.model || 'model'} down`}
              disabled={index === count - 1}
              onClick={() => onMove(1)}
            >
              <ArrowDown />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              aria-label={`Remove ${value.model || 'model reference'}`}
              onClick={onRemove}
            >
              <Trash2 />
            </Button>
          </span>
        )}
      </div>
      <div className="flex flex-col gap-3 pl-7">
        <ReasoningControls value={value} family={family} readOnly={readOnly} onChange={onChange} />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_140px]">
          <label className="flex min-w-0 flex-col gap-1.5">
            <span className="flex items-center justify-between text-[13px] font-medium">
              {fieldLabel('lora_name')}
              <span className="font-mono text-[11px] font-normal text-subtle-foreground">
                lora_name
              </span>
            </span>
            <SuggestInput
              ariaLabel="LoRA adapter"
              value={typeof value.lora_name === 'string' ? value.lora_name : ''}
              suggestions={loras}
              mono
              placeholder={loras.length ? 'Choose an adapter' : 'Optional adapter name'}
              disabled={readOnly}
              onChange={(next) => onChange(withField(value, 'lora_name', next))}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[13px] font-medium">{fieldLabel('weight')}</span>
            <NumberInput
              ariaLabel="Weight"
              value={value.weight}
              min={0}
              placeholder="none"
              disabled={readOnly}
              onChange={(next) => onChange(withField(value, 'weight', next))}
            />
          </label>
        </div>
        {others.length ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {others.map((field: FieldSchema) => (
              <SchemaField
                key={field.key}
                field={field}
                value={value[field.key]}
                readOnly={readOnly}
                path={['modelRefs', String(index), field.key]}
                onChange={(next) => onChange(withField(value, field.key, next))}
              />
            ))}
          </div>
        ) : null}
      </div>
    </li>
  )
}

interface EditorProps {
  value: DecisionModelRef[]
  onChange: (value: DecisionModelRef[]) => void
  models: NormalizedModel[]
  reasoningFamilies: Record<string, ReasoningFamily>
  readOnly?: boolean
}

/** A decision's candidate models in order, with every model reference option. */
export default function ConfigPageDecisionModelRefsEditor({
  value,
  onChange,
  models,
  reasoningFamilies,
  readOnly = false,
}: EditorProps) {
  const refs = (Array.isArray(value) ? value : []) as ModelRef[]
  const update = (index: number, next: ModelRef) =>
    onChange(refs.map((ref, position) => (position === index ? next : ref)))
  const move = (index: number, delta: number) => {
    const target = index + delta
    if (target < 0 || target >= refs.length) return
    const next = [...refs]
    ;[next[index], next[target]] = [next[target], next[index]]
    onChange(next)
  }
  return (
    <div className="flex flex-col gap-2">
      {refs.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border-strong px-4 py-4 text-[13px] text-muted-foreground">
          No candidate models. A decision without models must answer through a plugin, such as a
          fixed response.
        </p>
      ) : (
        <ol className="flex flex-col gap-2">
          {refs.map((ref, index) => (
            <ModelRefCard
              key={index}
              value={ref}
              index={index}
              count={refs.length}
              models={models}
              families={reasoningFamilies}
              readOnly={readOnly}
              onChange={(next) => update(index, next)}
              onMove={(delta) => move(index, delta)}
              onRemove={() => onChange(refs.filter((_, position) => position !== index))}
            />
          ))}
        </ol>
      )}
      {readOnly ? null : (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="w-fit"
          onClick={() => onChange([...refs, { model: '' } as ModelRef])}
        >
          <Plus />
          Add candidate model
        </Button>
      )}
    </div>
  )
}
