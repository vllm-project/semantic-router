import * as React from 'react'

import type { FieldSchema } from '../lib/dslSchemas'
import type { Tone } from '../ui'

/** Configuration-wide hints the forms offer as suggestions; values stay free text. */
export interface SchemaFormHints {
  /** Suggestions for a string or string-list field at a path, such as model or deployment names. */
  suggest?: (field: FieldSchema, path: readonly string[]) => readonly string[] | undefined
  /** Declared signal names by canonical type, for rule conditions. */
  signalNames?: (type: string) => readonly string[]
  /** Labels a referenced signal can report, for labelled rule conditions. */
  signalLabels?: (type: string, name: string) => readonly string[]
  /** Entity tone for sliders and highlights: signal gold, decision violet, projection teal. */
  accent?: Tone
}

export const SchemaFormHintsContext = React.createContext<SchemaFormHints>({})

export function SchemaFormHintsProvider({
  value,
  children,
}: {
  value: SchemaFormHints
  children: React.ReactNode
}) {
  return React.createElement(SchemaFormHintsContext.Provider, { value }, children)
}

export function useSchemaFormHints(): SchemaFormHints {
  return React.useContext(SchemaFormHintsContext)
}

export const BOUND_KEYS = ['gt', 'gte', 'lt', 'lte'] as const
export type BoundKey = (typeof BOUND_KEYS)[number]
export const BOUND_LABEL: Record<BoundKey, string> = { gt: '>', gte: '≥', lt: '<', lte: '≤' }

/** An object of numeric bounds only, such as a predicate, count or density. */
export function isPredicateSchema(field: FieldSchema): boolean {
  const keys = (field.fields ?? []).map((nested) => nested.key)
  return (
    field.type === 'object' &&
    keys.length > 0 &&
    keys.every((key) => (BOUND_KEYS as readonly string[]).includes(key))
  )
}

const LONG_TEXT_KEY = /(prompt|instructions|message|template|description|contract)$/

export function isLongTextField(field: FieldSchema): boolean {
  return field.type === 'string' && LONG_TEXT_KEY.test(field.key)
}

/** A list whose entries may contain commas, such as regular expressions. */
export function isPatternList(field: FieldSchema, parent: Record<string, unknown>): boolean {
  return (
    field.type === 'string[]' &&
    (/patterns$/.test(field.key) || (field.key === 'keywords' && parent.method === 'regex'))
  )
}

/** A bounded number that reads better as a slider, such as a 0–1 threshold. */
export function isSliderField(field: FieldSchema): boolean {
  return (
    field.type === 'number' &&
    typeof field.min === 'number' &&
    typeof field.max === 'number' &&
    field.max > field.min &&
    field.max - field.min <= 100
  )
}

export function boundsSummary(value: unknown): string {
  if (!value || typeof value !== 'object') return ''
  const record = value as Record<string, unknown>
  return BOUND_KEYS.filter((bound) => typeof record[bound] === 'number')
    .map((bound) => `${BOUND_LABEL[bound]} ${record[bound]}`)
    .join(' and ')
}
