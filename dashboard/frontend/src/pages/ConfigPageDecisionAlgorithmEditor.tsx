import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Check } from 'lucide-react'

import { getAlgorithmFieldSchema, type AlgorithmType } from '../lib/dslAlgorithmSchemas'
import { Badge, SectionLabel, cn } from '../ui'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import {
  algorithmFields,
  algorithmType,
  mergeAlgorithmFields,
} from './configPageDecisionAlgorithmSupport'
import { ALGORITHMS, algorithmMeta } from './configPageDecisionPresentation'

interface ConfigPageDecisionAlgorithmEditorProps {
  value: unknown
  onChange?: (value: Record<string, unknown> | undefined) => void
  readOnly?: boolean
}

const ease = [0.16, 1, 0.3, 1] as const

function AlgorithmOption({
  label,
  description,
  badges,
  selected,
  disabled,
  onSelect,
}: {
  label: string
  description: string
  badges?: ReactNode
  selected: boolean
  disabled: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled && !selected}
      onClick={onSelect}
      className={cn(
        'relative flex min-h-[86px] flex-col gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none',
        selected
          ? 'border-chart-3/50 bg-chart-3/[0.07]'
          : 'border-border bg-surface hover:border-border-strong hover:bg-surface-subtle',
        disabled && !selected && 'cursor-not-allowed opacity-55',
      )}
    >
      {selected ? (
        <motion.span
          layoutId="algorithm-selected"
          className="absolute right-2.5 top-2.5 grid size-4 place-items-center rounded-full bg-chart-3 text-white"
          transition={{ duration: 0.25, ease }}
        >
          <Check className="size-3" strokeWidth={3} />
        </motion.span>
      ) : null}
      <span className="flex flex-wrap items-center gap-1.5 pr-5">
        <span className="text-[13px] font-medium">{label}</span>
        {badges}
      </span>
      <span className="text-xs leading-relaxed text-muted-foreground">{description}</span>
    </button>
  )
}

/** The selection algorithm: a type from the Router contract and every parameter of its schema. */
export default function ConfigPageDecisionAlgorithmEditor({
  value,
  onChange,
  readOnly = false,
}: ConfigPageDecisionAlgorithmEditorProps) {
  const configured = Boolean(value && typeof value === 'object')
  const type = algorithmType(value)
  const meta = algorithmMeta(type)
  const fields = algorithmFields(value)
  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2">
        <SectionLabel>Type</SectionLabel>
        <div
          role="radiogroup"
          aria-label="Selection algorithm"
          className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3"
        >
          <AlgorithmOption
            label="Not set"
            description="Leaves the algorithm block out of the decision."
            selected={!configured}
            disabled={readOnly}
            onSelect={() => onChange?.(undefined)}
          />
          {ALGORITHMS.map((algorithm) => (
            <AlgorithmOption
              key={algorithm.type}
              label={algorithm.label}
              description={algorithm.description}
              selected={configured && type === algorithm.type}
              disabled={readOnly}
              badges={
                <>
                  {algorithm.kind === 'executor' ? (
                    <Badge size="sm" tone="outline">
                      executor
                    </Badge>
                  ) : null}
                  {algorithm.tier !== 'supported' ? (
                    <Badge size="sm" tone="warning">
                      {algorithm.tier}
                    </Badge>
                  ) : null}
                </>
              }
              onSelect={() => {
                if (configured && type === algorithm.type) return
                onChange?.(mergeAlgorithmFields(undefined, algorithm.type as AlgorithmType, {}))
              }}
            />
          ))}
        </div>
      </section>
      {configured ? (
        <section className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between gap-3">
            <SectionLabel>{meta?.label ?? type} parameters</SectionLabel>
            <span className="font-mono text-[11px] text-subtle-foreground">algorithm.{type}</span>
          </div>
          <ConfigPageSchemaFieldsEditor
            schema={getAlgorithmFieldSchema(type)}
            value={fields}
            readOnly={readOnly}
            path={['algorithm', type]}
            onChange={(nextFields) => onChange?.(mergeAlgorithmFields(value, type, nextFields))}
          />
        </section>
      ) : null}
    </div>
  )
}
