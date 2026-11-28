import { useId, useState, type ReactNode } from 'react'
import { ChevronDown, SlidersHorizontal } from 'lucide-react'

import NativeSelect from '../components/NativeSelect'
import {
  ChipInput,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Field,
  Input,
  InputGroup,
  SectionLabel,
  Textarea,
  cn,
} from '../ui'
import type { ConnectModelAdvancedValues } from './configPageConnectModelSupport'
import { PROVIDER_LB_POLICIES, PROVIDER_LB_POLICY_LABELS } from './configPageSupport'

interface Props {
  value: ConnectModelAdvancedValues
  reasoningFamilies: string[]
  onChange: (value: ConnectModelAdvancedValues) => void
}

type TextKey = {
  [Key in keyof ConnectModelAdvancedValues]: ConnectModelAdvancedValues[Key] extends string
    ? Key
    : never
}[keyof ConnectModelAdvancedValues]

const chipValues = (value: string) =>
  value
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)

export default function ConfigPageConnectModelAdvanced({
  value,
  reasoningFamilies,
  onChange,
}: Props) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const update = (key: TextKey, next: string) => onChange({ ...value, [key]: next })
  const fieldId = (key: TextKey) => `${id}-${key}`

  const text = (key: TextKey, label: string, placeholder: string, hint?: ReactNode) => (
    <Field label={label} htmlFor={fieldId(key)} optional hint={hint}>
      <Input
        id={fieldId(key)}
        value={value[key]}
        onChange={(event) => update(key, event.target.value)}
        placeholder={placeholder}
      />
    </Field>
  )
  const chips = (key: TextKey, label: string, placeholder: string, hint: string) => (
    <Field label={label} htmlFor={fieldId(key)} optional hint={hint} className="sm:col-span-2">
      <ChipInput
        id={fieldId(key)}
        value={chipValues(value[key])}
        onChange={(next) => update(key, next.join(', '))}
        placeholder={placeholder}
      />
    </Field>
  )
  const cost = (key: TextKey, label: string, placeholder = '0.00') => (
    <Field label={label} htmlFor={fieldId(key)} optional>
      <InputGroup
        id={fieldId(key)}
        inputMode="decimal"
        value={value[key]}
        onChange={(event) => update(key, event.target.value)}
        placeholder={placeholder}
        leading={<span className="text-[13px]">$</span>}
        className="tabular"
      />
    </Field>
  )

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-xl border border-border bg-card"
    >
      <CollapsibleTrigger className="flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40">
        <span className="grid size-7 shrink-0 place-items-center rounded-md bg-surface-subtle text-muted-foreground">
          <SlidersHorizontal className="size-3.5" />
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="text-[13px] font-medium">Advanced settings</span>
          <span className="text-xs text-muted-foreground">
            Identity, routing metadata, cost, and delivery
          </span>
        </span>
        <ChevronDown
          className={cn(
            'size-4 shrink-0 text-subtle-foreground transition-transform duration-200',
            open && 'rotate-180',
          )}
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="flex flex-col gap-6 border-t border-border px-4 pb-5 pt-4">
          <section className="flex flex-col gap-3">
            <SectionLabel>Identity</SectionLabel>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {text('namePrefix', 'Name prefix', 'team or environment')}
              <Field label="Reasoning family" htmlFor={fieldId('reasoningFamily')} optional>
                <NativeSelect
                  id={fieldId('reasoningFamily')}
                  value={value.reasoningFamily}
                  onChange={(event) => update('reasoningFamily', event.target.value)}
                >
                  <option value="">None</option>
                  {reasoningFamilies.map((family) => (
                    <option key={family} value={family}>
                      {family}
                    </option>
                  ))}
                </NativeSelect>
              </Field>
              <Field
                label="Description"
                htmlFor={fieldId('description')}
                optional
                className="sm:col-span-2"
              >
                <Textarea
                  id={fieldId('description')}
                  value={value.description}
                  onChange={(event) => update('description', event.target.value)}
                  placeholder="What this model is best at"
                  rows={2}
                />
              </Field>
            </div>
          </section>

          <section className="flex flex-col gap-3">
            <SectionLabel>Routing metadata</SectionLabel>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {chips('inputModalities', 'Input modalities', 'text, image', 'Text when empty.')}
              {chips('outputModalities', 'Output modalities', 'image', 'Text when empty.')}
              {text('parameterSize', 'Parameter size', 'e.g. 32B')}
              <Field label="Context window" htmlFor={fieldId('contextWindow')} optional>
                <Input
                  id={fieldId('contextWindow')}
                  type="number"
                  min="1"
                  value={value.contextWindow}
                  onChange={(event) => update('contextWindow', event.target.value)}
                  placeholder="e.g. 131072"
                  className="tabular"
                />
              </Field>
              {chips(
                'capabilities',
                'Capabilities',
                'reasoning, coding, tools, vision',
                'Press Enter or type a comma after each value.',
              )}
              {chips(
                'tags',
                'Tags',
                'fast, local, frontier',
                'Press Enter or type a comma after each value.',
              )}
            </div>
          </section>

          <section className="flex flex-col gap-3">
            <SectionLabel>Pricing · per one million tokens</SectionLabel>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {cost('inputCost', 'Input cost')}
              {cost('outputCost', 'Output cost')}
              {cost('cacheReadCost', 'Cache read cost', 'Defaults to input cost')}
              {cost('cacheWriteCost', 'Cache write cost', 'Defaults to input cost')}
            </div>
          </section>

          <section className="flex flex-col gap-3">
            <SectionLabel>Delivery · override platform defaults only when needed</SectionLabel>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field label="Max retries" htmlFor={fieldId('maxRetries')} hint="Between 0 and 5.">
                <Input
                  id={fieldId('maxRetries')}
                  type="number"
                  min="0"
                  max="5"
                  step="1"
                  value={value.maxRetries}
                  onChange={(event) => update('maxRetries', event.target.value)}
                  placeholder="Platform default"
                  className="tabular"
                />
              </Field>
              <Field label="Load balancing" htmlFor={fieldId('loadBalancing')} optional>
                <NativeSelect
                  id={fieldId('loadBalancing')}
                  value={value.loadBalancing}
                  onChange={(event) => update('loadBalancing', event.target.value)}
                >
                  <option value="">Platform default</option>
                  {PROVIDER_LB_POLICIES.map((policy) => (
                    <option key={policy} value={policy}>
                      {PROVIDER_LB_POLICY_LABELS[policy]}
                    </option>
                  ))}
                </NativeSelect>
              </Field>
              <div className="sm:col-span-2">
                {text('retryOn', 'Retry conditions', '5xx,reset,connect-failure')}
              </div>
              {text('healthCheckPath', 'Health check path', '/health')}
              {text('healthCheckInterval', 'Check interval', '10s')}
              {text('healthCheckTimeout', 'Check timeout', '2s')}
            </div>
          </section>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
